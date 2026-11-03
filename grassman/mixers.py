"""Token mixers compared in the Grassmann ablation.

Every mixer maps (B, T, d) -> (B, T, d), is causal (the output at t sees only
positions <= t) and has roughly the parameters of one attention layer
(about 4 d^2), so the models differ only in how tokens are mixed.

PairMixer is the Causal Grassmann layer of arXiv 2512.19428 (Sec. 3) with
the pair feature swappable:

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


class PairMixer(nn.Module):
    """Causal Grassmann mixing: reduce, pair with t-delta, embed the pair,
    project back, average over offsets, gate against the input."""

    def __init__(self, d: int, kind: str = "antisym", r: int | None = None,
                 windows=(1, 2, 4, 8, 12, 16), normalize: bool = True,
                 dropout: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.kind, self.windows, self.normalize, self.eps = kind, tuple(windows), normalize, eps
        self.r = PAIR_R[kind] if r is None else r
        self.red = nn.Linear(d, self.r)
        self.proj = nn.Linear(pair_dim(kind, self.r), d)
        self.gate = nn.Linear(2 * d, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    @property
    def reach(self) -> int:
        return max(self.windows)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, d = h.shape
        # Pair features in fp32: a_i b_j - a_j b_i cancels badly in bf16.
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
