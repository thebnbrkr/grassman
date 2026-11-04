"""Token mixers for the recall tests (Track B).

Every mixer maps (B, T, d) -> (B, T, d) and is causal. Next to each forward
pass, `state_numbers(T)` gives the numbers a mixer must keep per sequence to
generate token T+1 in streaming mode (its inference cache), so models can be
compared on recall against memory:

    HyenaLongMixer    order-2 Hyena with an implicit long filter (FFT conv)   d*T history (+ short-conv buffer)
    LinearAttention   causal linear attention, elu+1 feature map              h * dk * (dv + 1)
    DeltaRuleMixer    DeltaNet-style delta-rule memory (chunked, exact)       h * dk * dv
    MomentMemory      per-chunk summary slots, attended by later chunks       (T/C) * slots * 2d + C*d
    attention / swa   grassman.mixers.AttentionMixer                          2d*T / 2d*window

The Hyena history counts the input to its long convolution: without
distilling the filter into a recurrence (Massaroli et al. 2023), streaming
generation needs the whole past. That is stated, not hidden.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from grassman.mixers import AttentionMixer, causal_depthwise_conv


def _fp32(device_type: str):
    return torch.autocast(device_type=device_type, enabled=False)


def fft_causal_conv(u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """y[..., c, t] = sum_{s<=t} h[c, s] u[..., c, t-s]; u (B, C, T), h (C, T). fp32."""
    T = u.shape[-1]
    n = 2 * T
    y = torch.fft.irfft(torch.fft.rfft(u.float(), n=n) * torch.fft.rfft(h.float(), n=n), n=n)
    return y[..., :T]


class ImplicitFilter(nn.Module):
    """Hyena's implicit filter (Poli et al. 2023, Sec. 3.3): a small sine-MLP
    over a positional encoding of t, times an exponential decay window whose
    rate differs per channel (fast to slow)."""

    def __init__(self, d: int, max_len: int, emb_dim: int = 33, hidden: int = 64,
                 fast_decay_pct: float = 0.3, slow_decay_pct: float = 1.5, target: float = 1e-2):
        super().__init__()
        assert emb_dim % 2 == 1
        self.max_len = max_len
        t = torch.linspace(0, 1, max_len)[:, None]                       # (L, 1)
        bands = (emb_dim - 1) // 2
        f = torch.linspace(1e-4, bands - 1, bands)[None]                  # (1, bands)
        w = 2 * math.pi * torch.arange(max_len)[:, None] / max_len
        self.register_buffer("z", torch.cat([t, torch.cos(f * w), torch.sin(f * w)], -1), persistent=False)
        self.register_buffer("t", t, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(emb_dim, hidden), _Sin(hidden), nn.Linear(hidden, hidden),
                                 _Sin(hidden), nn.Linear(hidden, d, bias=False))
        lo, hi = math.log(target) / slow_decay_pct, math.log(target) / fast_decay_pct
        self.register_buffer("deltas", torch.linspace(lo, hi, d)[None].abs(), persistent=False)

    def forward(self, T: int) -> torch.Tensor:
        assert T <= self.max_len, f"sequence {T} > filter length {self.max_len}"
        h = self.mlp(self.z[:T]) * torch.exp(-self.t[:T] * self.deltas)  # (T, d)
        return h.transpose(0, 1)                                          # (d, T)


class _Sin(nn.Module):
    def __init__(self, dim: int, w: float = 1.0):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim))

    def forward(self, x):
        return torch.sin(self.freq * x)


class HyenaLongMixer(nn.Module):
    """Order-2 Hyena operator with one implicit long filter, as in the Hyena
    reference implementation: y = x0 * (h * (x1 * v)) + D-skip, where the three
    projections first pass a width-3 causal depthwise conv."""

    def __init__(self, d: int, max_len: int, short_len: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d, self.short_len = d, short_len
        self.in_proj = nn.Linear(d, 3 * d)
        self.short = nn.Parameter(torch.zeros(3 * d, short_len))
        with torch.no_grad():
            self.short[:, 0] = 1.0
            self.short.add_(0.02 * torch.randn_like(self.short))
        self.filter = ImplicitFilter(d, max_len)
        self.bias = nn.Parameter(torch.randn(d) * 0.02)
        self.out_proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def state_numbers(self, T: int) -> int:
        return self.d * T + 3 * self.d * (self.short_len - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        u = causal_depthwise_conv(self.in_proj(x).transpose(1, 2), self.short)
        x0, x1, v = u.chunk(3, dim=1)
        z = x1 * v
        with _fp32(x.device.type):
            y = fft_causal_conv(z, self.filter(T)) + z.float() * self.bias[:, None]
        y = (x0.float() * y).to(x.dtype)
        return self.drop(self.out_proj(y.transpose(1, 2)))


def _heads(t: torch.Tensor, n_heads: int) -> torch.Tensor:
    B, T, d = t.shape
    return t.view(B, T, n_heads, d // n_heads).transpose(1, 2)           # (B, h, T, dh)


def _merge(t: torch.Tensor) -> torch.Tensor:
    B, h, T, dh = t.shape
    return t.transpose(1, 2).reshape(B, T, h * dh)


def _chunks(t: torch.Tensor, C: int) -> torch.Tensor:
    B, h, T, e = t.shape
    assert T % C == 0, f"sequence {T} not a multiple of chunk {C}"
    return t.view(B, h, T // C, C, e)


def linear_attention(q, k, v, chunk: int = 64, eps: float = 1e-6):
    """Causal linear attention o_t = (sum_{s<=t} (q_t.k_s) v_s) / (sum_{s<=t} q_t.k_s),
    computed chunkwise; q, k already passed through a positive feature map.
    q, k (B, h, T, dk), v (B, h, T, dv)."""
    T = q.shape[2]
    C = min(chunk, T)
    qc, kc, vc = _chunks(q, C), _chunks(k, C), _chunks(v, C)
    kv = kc.transpose(-1, -2) @ vc                                        # (B, h, n, dk, dv)
    S = torch.cumsum(kv, dim=2) - kv                                      # state before each chunk
    z = torch.cumsum(kc.sum(3), dim=2) - kc.sum(3)                        # (B, h, n, dk)
    tri = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
    a = (qc @ kc.transpose(-1, -2)).masked_fill(~tri, 0.0)               # (B, h, n, C, C)
    num = qc @ S + a @ vc
    den = (qc * z.unsqueeze(3)).sum(-1, keepdim=True) + a.sum(-1, keepdim=True)
    return (num / (den + eps)).flatten(2, 3)


def delta_rule(q, k, v, beta, chunk: int = 64):
    """DeltaNet recurrence S_t = S_{t-1} + beta_t k_t (v_t - S_{t-1}^T k_t)^T,
    o_t = S_t^T q_t, with S_0 = 0, computed chunkwise and exactly (Yang et al.
    2024): inside a chunk, U = (I + diag(b) tril(K K^T, -1))^{-1} diag(b)(V - K S_0),
    O = Q S_0 + tril(Q K^T) U, S <- S_0 + K^T U.
    q, k (B, h, T, dk), v (B, h, T, dv), beta (B, h, T)."""
    B, h, T, dk = k.shape
    dv = v.shape[-1]
    C = min(chunk, T)
    qc, kc, vc = _chunks(q, C), _chunks(k, C), _chunks(v, C)
    bc = beta.view(B, h, T // C, C, 1)
    eye = torch.eye(C, device=k.device, dtype=k.dtype)
    strict = torch.tril(kc @ kc.transpose(-1, -2), diagonal=-1)
    A = eye + bc * strict                                                 # unit lower-triangular
    W = torch.linalg.solve_triangular(A, bc * kc, upper=False, unitriangular=True)
    U0 = torch.linalg.solve_triangular(A, bc * vc, upper=False, unitriangular=True)
    qk = torch.tril(qc @ kc.transpose(-1, -2))                            # includes the diagonal
    S = k.new_zeros(B, h, dk, dv)
    out = []
    for i in range(T // C):
        U = U0[:, :, i] - W[:, :, i] @ S
        out.append(qc[:, :, i] @ S + qk[:, :, i] @ U)
        S = S + kc[:, :, i].transpose(-1, -2) @ U
    return torch.stack(out, dim=2).flatten(2, 3)


def delta_rule_reference(q, k, v, beta):
    """Token-by-token recurrence, for tests."""
    B, h, T, dk = k.shape
    S = k.new_zeros(B, h, dk, v.shape[-1])
    out = []
    for t in range(T):
        kt, vt, bt = k[:, :, t], v[:, :, t], beta[:, :, t, None]
        pred = (S.transpose(-1, -2) @ kt.unsqueeze(-1)).squeeze(-1)
        S = S + kt.unsqueeze(-1) @ (bt * (vt - pred)).unsqueeze(-2)
        out.append((S.transpose(-1, -2) @ q[:, :, t].unsqueeze(-1)).squeeze(-1))
    return torch.stack(out, dim=2)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(dim)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class LinearAttention(nn.Module):
    """Multi-head causal linear attention (Katharopoulos et al. 2020): one
    fixed dk x dv matrix per head holds the whole past."""

    def __init__(self, d: int, n_heads: int = 4, chunk: int = 64, dropout: float = 0.1):
        super().__init__()
        self.h, self.chunk = n_heads, chunk
        self.qkv = nn.Linear(d, 3 * d)
        self.norm = _RMSNorm(d // n_heads)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def state_numbers(self, T: int) -> int:
        dh = self.out.in_features // self.h
        return self.h * dh * (dh + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = (_heads(t, self.h) for t in self.qkv(x).chunk(3, -1))
        with _fp32(x.device.type):
            o = linear_attention(F.elu(q.float()) + 1, F.elu(k.float()) + 1, v.float(), self.chunk)
        return self.drop(self.out(_merge(self.norm(o)).to(x.dtype)))


class DeltaRuleMixer(nn.Module):
    """Multi-head delta-rule memory (Schlag et al. 2021; DeltaNet, Yang et al.
    2024): writes overwrite what a key currently returns instead of adding to
    it. Keys and queries are SiLU + L2-normalised; beta = sigmoid(linear(x))."""

    def __init__(self, d: int, n_heads: int = 4, chunk: int = 64, dropout: float = 0.1):
        super().__init__()
        self.h, self.chunk = n_heads, chunk
        self.qkv = nn.Linear(d, 3 * d)
        self.beta = nn.Linear(d, n_heads)
        self.norm = _RMSNorm(d // n_heads)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def state_numbers(self, T: int) -> int:
        dh = self.out.in_features // self.h
        return self.h * dh * dh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = (_heads(t, self.h) for t in self.qkv(x).chunk(3, -1))
        with _fp32(x.device.type):
            q = F.normalize(F.silu(q.float()), dim=-1)
            k = F.normalize(F.silu(k.float()), dim=-1)
            beta = torch.sigmoid(self.beta(x).float()).transpose(1, 2)    # (B, h, T)
            o = delta_rule(q, k, v.float(), beta, self.chunk)
        return self.drop(self.out(_merge(self.norm(o)).to(x.dtype)))


class MomentMemory(nn.Module):
    """A lookup over summaries of past chunks ("moments").

    The sequence is cut into chunks of C tokens. Each chunk is summarised into
    `slots` vectors by learned attention pooling over its own tokens. A token
    in chunk j attends (softmax, multi-head) over the slots of chunks < j plus
    one always-visible null slot, so it can look up any earlier moment but not
    its own, unfinished chunk (the local Hyena layers cover that). The memory
    grows with T / C, `slots` entries per chunk instead of C."""

    def __init__(self, d: int, n_heads: int = 4, chunk: int = 64, slots: int = 4, dropout: float = 0.1):
        super().__init__()
        self.h, self.C, self.slots, self.d, self.p = n_heads, chunk, slots, d, dropout
        self.pool_q = nn.Parameter(torch.randn(slots, d) * 0.02)
        self.pool_kv = nn.Linear(d, 2 * d)
        self.mem_kv = nn.Linear(d, 2 * d)
        self.null_kv = nn.Parameter(torch.zeros(2, d))
        self.q = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def state_numbers(self, T: int) -> int:
        return (T // self.C) * self.slots * 2 * self.d + self.C * self.d

    def summaries(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d) -> (B, T // C, slots, d)."""
        B, T, d = x.shape
        assert T % self.C == 0, f"sequence {T} not a multiple of chunk {self.C}"
        pk, pv = self.pool_kv(x).view(B, T // self.C, self.C, 2, d).unbind(3)
        w = torch.softmax(torch.einsum("sd,bncd->bnsc", self.pool_q, pk) / math.sqrt(d), dim=-1)
        return torch.einsum("bnsc,bncd->bnsd", w, pv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        n = T // self.C
        mk, mv = self.mem_kv(self.summaries(x).reshape(B, n * self.slots, d)).chunk(2, -1)
        mk = torch.cat([self.null_kv[0].expand(B, 1, d).to(mk.dtype), mk], 1)
        mv = torch.cat([self.null_kv[1].expand(B, 1, d).to(mv.dtype), mv], 1)
        chunk_of_token = torch.arange(T, device=x.device) // self.C
        chunk_of_slot = torch.cat([torch.full((1,), -1, device=x.device),
                                   torch.arange(n, device=x.device).repeat_interleave(self.slots)])
        mask = chunk_of_slot[None, :] < chunk_of_token[:, None]           # (T, 1 + n*slots)
        p = self.p if self.training else 0.0
        o = F.scaled_dot_product_attention(_heads(self.q(x), self.h), _heads(mk, self.h), _heads(mv, self.h),
                                           attn_mask=mask, dropout_p=p)
        return self.drop(self.out(_merge(o)))


def attention_state(mixer: AttentionMixer, T: int) -> int:
    d = mixer.out.in_features
    return 2 * d * (T if mixer.window is None else min(T, mixer.window + 1))
