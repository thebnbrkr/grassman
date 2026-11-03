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

import contextlib

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


def _wide(x: torch.Tensor) -> torch.Tensor:
    """bf16/fp16 -> fp32; fp32 and fp64 unchanged (the tests run in fp64)."""
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def _no_autocast(h: torch.Tensor):
    """Keep the folded projection in fp32 under CUDA autocast."""
    return torch.autocast("cuda", enabled=False) if h.is_cuda else contextlib.nullcontext()


def pair_norm(a: torch.Tensor, b: torch.Tensor, kind: str, eps: float = 0.0) -> torch.Tensor:
    """max(||pair_features(a, b, kind)||, eps) in closed form, without building the
    features. antisym uses Lagrange's identity ||a ^ b||^2 = |a|^2 |b|^2 - (a.b)^2.
    Clamping the square (not the root) keeps the gradient finite at zero pairs."""
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
    return sq.clamp_min(eps * eps).sqrt()


def sin_angle(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """sin of the angle between a and b: 0 = same direction, 1 = orthogonal."""
    return pair_norm(a, b, "antisym", eps) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(eps)


class PairMixer(nn.Module):
    """Causal Grassmann mixing: reduce, pair t with t-delta for each offset,
    embed the pair, average over offsets, project back, gate against the input.

    The paper projects every offset's pair features and then averages. Every
    pair shares z_t, and each pair feature is linear in its second vector, so

        sum_delta s_delta * pair(z_t, z_{t-delta}) = pair(z_t, b_t),
        b_t = sum_delta s_delta * z_{t-delta}

    with s_delta = edge weight / ||pair|| (the normaliser, from pair_norm):
    the layer is one pair of the current token with a weighted average of its
    past window (for concat the first half is z_t * sum_delta s_delta). We
    build that one pair and fold the upper-triangle selection into the
    projection weights (`_folded_weight`), so there is no per-token gather,
    and take all offsets at once from one strided window (`_past`), so there
    is no Python loop over offsets.
    `reference_forward` is the paper's order; tests/test_speed.py checks both
    give the same outputs and gradients.

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
        self.register_buffer("_deltas", torch.tensor(self.windows), persistent=False)
        D = max(self.windows)
        self.register_buffer("_win_idx", torch.tensor([D - delta for delta in self.windows]), persistent=False)
        if kind in ("antisym", "sym"):
            i, j = torch.triu_indices(self.r, self.r, offset=1 if kind == "antisym" else 0)
            self.register_buffer("_ij", i * self.r + j, persistent=False)
            self.register_buffer("_ji", j * self.r + i, persistent=False)

    @property
    def reach(self) -> int:
        return max(self.windows)

    def _past(self, z: torch.Tensor) -> torch.Tensor:
        """(B, T, E, r): z_{t - delta} for every offset, zero where t - delta < 0.
        One strided window plus one small index, instead of a loop over offsets."""
        D = max(self.windows)
        win = F.pad(z, (0, 0, D, 0)).unfold(1, D + 1, 1)          # (B, T, r, D+1); [..., k] = z_{t-D+k}
        return win.index_select(-1, self._win_idx).transpose(-1, -2)

    def _edge_weights(self, z: torch.Tensor, past: torch.Tensor, sin) -> torch.Tensor:
        """(B, T, E) weights over the offsets; zero where t - delta < 0, and on
        rows with no valid offset (t = 0)."""
        B, T, _ = z.shape
        valid = torch.arange(T, device=z.device)[:, None] >= self._deltas[None, :]   # (T, E)
        if self.drop_edge in self.windows:
            valid[:, self.windows.index(self.drop_edge)] = False
        if not self.select:
            w = valid.to(z.dtype) / valid.sum(-1, keepdim=True).clamp_min(1)
            return w.expand(B, T, len(self.windows))
        Wa, Wb = self.score_in.weight[:, :self.r], self.score_in.weight[:, self.r:]
        pre = (z @ Wa.T + self.score_in.bias).unsqueeze(2) + past @ Wb.T          # (B, T, E, score_dim)
        scores = self.score_out(F.leaky_relu(pre, 0.2)).squeeze(-1) + self.edge_bias
        if self.angle:
            scores = scores + self.angle_weight * sin
        scores = scores.masked_fill(~valid, float("-inf"))
        scores = torch.where(valid.any(-1)[None, :, None], scores, torch.zeros_like(scores))
        return torch.softmax(scores, dim=-1) * valid

    def _folded_weight(self) -> torch.Tensor:
        """proj.weight laid out over the r x r product a b^T, so that
        proj(pair_features(a, b)) = vec(a b^T) @ W.T + bias with no gather."""
        W = self.proj.weight
        if self.kind == "outer":
            return W
        sign = -1.0 if self.kind == "antisym" else 1.0
        full = W.new_zeros(W.shape[0], self.r * self.r)
        return full.index_add(1, self._ij, W).index_add(1, self._ji, sign * W)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.use_reference:
            return self.reference_forward(h)
        B, T, d = h.shape
        z = _wide(self.red(h))       # a_i b_j - a_j b_i cancels badly in bf16
        past = self._past(z)                                            # (B, T, E, r)
        a = z.unsqueeze(2)
        sin = sin_angle(a, past) if (self.angle or self.record) else None
        w = self._edge_weights(z, past, sin).to(z.dtype)               # (B, T, E)
        s_e = w / pair_norm(a, past, self.kind, self.eps) if self.normalize else w
        b_avg = torch.einsum("bte,bter->btr", s_e, past)                # weighted average of the past
        if self.kind == "concat":
            g = self.proj(torch.cat([z * s_e.sum(-1, keepdim=True), b_avg], -1).to(h.dtype))
        else:
            outer = (z.unsqueeze(-1) * b_avg.unsqueeze(-2)).flatten(2)      # one pair per token
            with _no_autocast(h):
                g = F.linear(outer, self._folded_weight().to(z.dtype), self.proj.bias.to(z.dtype))
            g = g.to(h.dtype)
        g = g * (w.sum(-1, keepdim=True) > 0).to(g.dtype)
        if self.record:
            keep = [e for e, delta in enumerate(self.windows) if delta < T]
            self.last = {"deltas": [self.windows[e] for e in keep],
                         "weights": w[..., keep].detach(), "sin": sin[..., keep].detach()}
        alpha = torch.sigmoid(self.gate(torch.cat([h, g], dim=-1)))
        return self.drop(self.norm(alpha * h + (1 - alpha) * g))

    def reference_forward(self, h: torch.Tensor) -> torch.Tensor:
        """The paper's order of operations (project each offset, then average).
        Equal-weight averaging only; kept to test the fast path against."""
        assert not self.select
        B, T, d = h.shape
        z = _wide(self.red(h))
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
