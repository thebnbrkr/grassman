"""Token mixers compared in the Grassmann ablation.

Every mixer maps (B, T, d) -> (B, T, d), is causal (the output at t sees only
positions <= t) and has roughly the parameters of one attention layer
(about 4 d^2), so the models differ only in how tokens are mixed.

PairMixer is the Causal Grassmann layer of arXiv 2512.19428 (Sec. 3), with
optional learned edge selection over the offsets, and the pair feature
swappable:

    antisym  a_i b_j - a_j b_i   (i < j)   the paper's Plucker coordinates
    sym      a_i b_j + a_j b_i   (i <= j)  same products, no orientation
    outer    a_i b_j             (all i,j) the full pairwise product
    concat   [a; b]                        no product at all

Each kind gets its own reduced width r so that every variant hands the
projection about 496 numbers per pair (see PAIR_R). The paper writes the
pairing as (t, t+delta), which would read future tokens; we pair (t, t-delta)
as the public reproduction does, and tests/test_mixers.py checks causality.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PAIR_KINDS = ("antisym", "sym", "outer", "concat")

# r per kind, chosen so pair_dim(kind, r) is about 496 = pair_dim("antisym", 32)
PAIR_R = {"antisym": 32, "sym": 31, "outer": 22, "concat": 248}


def pair_dim(kind: str, r: int) -> int:
    return {"antisym": r * (r - 1) // 2, "sym": r * (r + 1) // 2,
            "outer": r * r, "concat": 2 * r}[kind]


def pair_features(a: torch.Tensor, b: torch.Tensor, kind: str) -> torch.Tensor:
    """Features of the pair (a, b) = (z_t, z_{t-delta}); both (..., r)."""
    if kind == "concat":
        return torch.cat([a, b], dim=-1)
    r = a.shape[-1]
    outer = a.unsqueeze(-1) * b.unsqueeze(-2)          # outer[..., i, j] = a_i b_j
    if kind == "outer":
        return outer.flatten(-2)
    if kind == "antisym":
        i, j = torch.triu_indices(r, r, offset=1, device=a.device)
        return outer[..., i, j] - outer[..., j, i]
    if kind == "sym":
        i, j = torch.triu_indices(r, r, offset=0, device=a.device)
        return outer[..., i, j] + outer[..., j, i]
    raise ValueError(f"unknown pair kind {kind!r}")


def pair_norm(a: torch.Tensor, b: torch.Tensor, kind: str) -> torch.Tensor:
    """||pair_features(a, b, kind)|| in closed form, without building the features.
    antisym uses Lagrange's identity ||a ^ b||^2 = |a|^2 |b|^2 - (a.b)^2."""
    aa, bb, ab = (a * a).sum(-1), (b * b).sum(-1), (a * b).sum(-1)
    if kind == "antisym":
        sq = aa * bb - ab * ab
    elif kind == "sym":
        sq = aa * bb + ab * ab + 2 * ((a * a) * (b * b)).sum(-1)
    elif kind == "outer":
        sq = aa * bb
    elif kind == "concat":
        sq = aa + bb
    else:
        raise ValueError(f"unknown pair kind {kind!r}")
    return sq.clamp_min(0).sqrt()


def sin_angle(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """sin of the angle between a and b: 0 = same direction, 1 = orthogonal."""
    return pair_norm(a, b, "antisym") / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(eps)


class PairMixer(nn.Module):
    """Causal Grassmann mixing: reduce, pair t with t-delta for each offset,
    embed the pair, average over offsets, project back, gate against the input.

    The paper projects every offset's pair features and then averages. Both
    steps are linear, so we average first and project once (6x fewer
    multiply-adds), build the average from the r x r pair products so the
    upper-triangle gather happens once instead of once per offset, and take
    the normaliser from pair_norm. `reference_forward` is the paper's order;
    tests/test_speed.py checks both give the same output and gradients.

    With select=True the equal-weight average becomes a learned softmax over
    the offsets (a GATv2-style score from [z_t; z_{t-delta}], a per-offset
    bias and, with angle=True, sin of the pair's angle): graph attention over
    a causal graph whose edges are the offsets. Set `record = True` to keep
    the last edge weights and angles in `self.last`; set `drop_edge` to an
    offset (e.g. 1 for t-1) to remove that edge (ablation)."""

    def __init__(self, d: int, kind: str = "antisym", r: int | None = None,
                 windows=(1, 2, 4, 8, 12, 16), normalize: bool = True,
                 select: bool = False, angle: bool = False, score_dim: int = 32,
                 dropout: float = 0.1, eps: float = 1e-8):
        super().__init__()
        assert not angle or select, "angle only feeds the edge scores"
        self.kind, self.windows, self.normalize, self.eps = kind, tuple(windows), normalize, eps
        self.select, self.angle = select, angle
        self.r = PAIR_R[kind] if r is None else r
        self.red = nn.Linear(d, self.r)
        self.proj = nn.Linear(pair_dim(kind, self.r), d)
        self.gate = nn.Linear(2 * d, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        if select:
            self.score_in = nn.Linear(2 * self.r, score_dim)
            self.score_out = nn.Linear(score_dim, 1, bias=False)
            self.edge_bias = nn.Parameter(torch.zeros(len(self.windows)))
            if angle:
                self.angle_weight = nn.Parameter(torch.zeros(1))
        self.record, self.last, self.drop_edge = False, None, None
        self.use_reference = False   # benchmark switch: run the paper's (slow) order of operations

    @property
    def reach(self) -> int:
        return max(self.windows)

    def _edge_weights(self, z: torch.Tensor, deltas: list[int], sins: list) -> torch.Tensor:
        """(B, T, E) weights over the offsets; zero where t - delta < 0, and on
        rows with no valid offset (t = 0)."""
        B, T, _ = z.shape
        t = torch.arange(T, device=z.device)
        valid = t[:, None] >= torch.tensor(deltas, device=z.device)[None, :]      # (T, E)
        if self.drop_edge in deltas:
            valid[:, deltas.index(self.drop_edge)] = False
        if not self.select:
            w = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
            return w.expand(B, T, len(deltas))
        Wa, Wb = self.score_in.weight[:, :self.r], self.score_in.weight[:, self.r:]
        za, zb = z @ Wa.T + self.score_in.bias, z @ Wb.T
        scores = z.new_full((B, T, len(deltas)), float("-inf"))
        for e, delta in enumerate(deltas):
            s = self.score_out(F.leaky_relu(za[:, delta:] + zb[:, :-delta], 0.2)).squeeze(-1)
            s = s + self.edge_bias[e]
            if self.angle:
                s = s + self.angle_weight * sins[e]
            scores[:, delta:, e] = s
        scores = scores.masked_fill(~valid, float("-inf"))
        scores = torch.where(valid.any(-1)[None, :, None], scores, torch.zeros_like(scores))
        return torch.softmax(scores, dim=-1) * valid

    def _features(self, m: torch.Tensor) -> torch.Tensor:
        """Pair features of the averaged pair statistic (linear in it)."""
        if self.kind in ("concat", "outer"):
            return m.flatten(2)
        i, j = torch.triu_indices(self.r, self.r, offset=1 if self.kind == "antisym" else 0, device=m.device)
        sign = -1.0 if self.kind == "antisym" else 1.0
        return m[..., i, j] + sign * m[..., j, i]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.use_reference:
            return self.reference_forward(h)
        B, T, d = h.shape
        # Pair statistics in fp32: a_i b_j - a_j b_i cancels badly in bf16.
        z = self.red(h).float()
        deltas = [delta for delta in self.windows if delta < T]
        need_sin = self.angle or self.record
        sins = [sin_angle(z[:, delta:], z[:, :-delta]) for delta in deltas] if need_sin else []
        if not deltas:
            g = torch.zeros_like(h)
        else:
            w = self._edge_weights(z, deltas, sins)
            m = None
            for e, delta in enumerate(deltas):
                a, b = z[:, delta:], z[:, :-delta]
                x = torch.cat([a, b], -1) if self.kind == "concat" else a.unsqueeze(-1) * b.unsqueeze(-2)
                scale = w[:, delta:, e]
                if self.normalize:
                    scale = scale / pair_norm(a, b, self.kind).clamp_min(self.eps)
                x = x * scale.reshape(*scale.shape, *([1] * (x.dim() - 2)))
                x = F.pad(x, (0, 0) * (x.dim() - 2) + (delta, 0))
                m = x if m is None else m + x
            has_edge = (w.sum(-1, keepdim=True) > 0).to(h.dtype)
            g = self.proj(self._features(m).to(h.dtype)) * has_edge
            if self.record:
                full = [F.pad(s, (delta, 0)) for s, delta in zip(sins, deltas)]
                self.last = {"deltas": deltas, "weights": w.detach(), "sin": torch.stack(full, -1).detach()}
        alpha = torch.sigmoid(self.gate(torch.cat([h, g], dim=-1)))
        return self.drop(self.norm(alpha * h + (1 - alpha) * g))

    def reference_forward(self, h: torch.Tensor) -> torch.Tensor:
        """The paper's order of operations (project each offset, then average).
        Equal-weight averaging only; kept to test the fast path against."""
        assert not self.select
        B, T, d = h.shape
        z = self.red(h).float()
        g = torch.zeros_like(h)
        count = h.new_zeros(1, T, 1)
        for delta in self.windows:
            if delta >= T:
                continue
            p = pair_features(z[:, delta:], z[:, :-delta], self.kind)
            if self.normalize:
                p = p / p.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            g = g + F.pad(self.proj(p.to(h.dtype)), (0, 0, delta, 0))
            count = count + F.pad(count.new_ones(1, T - delta, 1), (0, 0, delta, 0))
        g = g / count.clamp_min(1)
        alpha = torch.sigmoid(self.gate(torch.cat([h, g], dim=-1)))
        return self.drop(self.norm(alpha * h + (1 - alpha) * g))


def causal_depthwise_conv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y[:, c, t] = sum_k w[c, k] x[:, c, t-k]; x (B, C, T), w (C, K)."""
    K = w.shape[-1]
    return F.conv1d(F.pad(x, (K - 1, 0)), w.flip(-1).unsqueeze(1), groups=x.shape[1])


class HyenaShortMixer(nn.Module):
    """Order-2 Hyena operator (Poli et al. 2023) with an explicit causal
    filter, like Evo 2's short/medium blocks: y = x2 * (h * (x1 * v)), where
    each projection first passes a width-3 causal depthwise conv. The reach
    (short_len - 1) + (filter_len - 1) is 16 by default, matching PairMixer."""

    def __init__(self, d: int, filter_len: int = 15, short_len: int = 3, dropout: float = 0.1):
        super().__init__()
        self.filter_len, self.short_len = filter_len, short_len
        self.in_proj = nn.Linear(d, 3 * d)
        self.short = nn.Parameter(torch.zeros(3 * d, short_len))
        self.filter = nn.Parameter(torch.zeros(d, filter_len))
        with torch.no_grad():  # start both convs as near-identity
            self.short[:, 0] = 1.0
            self.short.add_(0.02 * torch.randn_like(self.short))
            self.filter[:, 0] = 1.0
            self.filter.add_(0.02 * torch.randn_like(self.filter))
        self.out_proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    @property
    def reach(self) -> int:
        return (self.short_len - 1) + (self.filter_len - 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        u = causal_depthwise_conv(self.in_proj(h).transpose(1, 2), self.short)
        x1, x2, v = u.chunk(3, dim=1)
        y = x2 * causal_depthwise_conv(x1 * v, self.filter)
        return self.drop(self.out_proj(y.transpose(1, 2)))


class AttentionMixer(nn.Module):
    """Causal multi-head attention. window=None is full attention; window=w
    lets position t see t-w .. t (sliding-window attention)."""

    def __init__(self, d: int, n_heads: int = 4, window: int | None = None, dropout: float = 0.1):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads, self.window, self.p = n_heads, window, dropout
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    @property
    def reach(self) -> int | None:
        return self.window

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, d = h.shape
        q, k, v = self.qkv(h).view(B, T, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        p = self.p if self.training else 0.0
        if self.window is None:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        else:
            i = torch.arange(T, device=h.device)
            dist = i[:, None] - i[None, :]
            mask = (dist >= 0) & (dist <= self.window)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        return self.drop(self.out(y.transpose(1, 2).reshape(B, T, d)))
