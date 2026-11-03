"""A small decoder-only LM whose token mixer is swappable.

Block layout follows the public Grassmann reproduction (pre-LN):
    x = x + mixer(ln1(x));  x = x + ffn(ln2(x))
with learned positions, tied input/output embeddings and N(0, 0.02) init.
Everything except the mixer is identical across variants.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixers import AttentionMixer, HyenaShortMixer, PairMixer

# mixer name -> PairMixer kind, for the four Grassmann-style variants
PAIR_MIXERS = {"grassmann": "antisym", "sym": "sym", "outer": "outer", "concat": "concat"}
MIXERS = (*PAIR_MIXERS, "hyena", "swa", "attention")


@dataclass
class Config:
    mixer: str = "grassmann"
    vocab_size: int = 30522        # bert-base-uncased WordPiece, as in the paper
    block_size: int = 128
    d: int = 256
    n_layers: int = 6
    d_ff: int = 1024
    n_heads: int = 4
    windows: tuple = (1, 2, 4, 8, 12, 16)
    normalize: bool = True         # L2-normalise pair features (paper: yes)
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


def make_mixer(cfg: Config) -> nn.Module:
    if cfg.mixer in PAIR_MIXERS:
        return PairMixer(cfg.d, kind=PAIR_MIXERS[cfg.mixer], windows=cfg.windows,
                         normalize=cfg.normalize, dropout=cfg.dropout)
    reach = max(cfg.windows)
    if cfg.mixer == "hyena":
        return HyenaShortMixer(cfg.d, filter_len=reach - 1, dropout=cfg.dropout)
    if cfg.mixer == "swa":
        return AttentionMixer(cfg.d, cfg.n_heads, window=reach, dropout=cfg.dropout)
    if cfg.mixer == "attention":
        return AttentionMixer(cfg.d, cfg.n_heads, window=None, dropout=cfg.dropout)
    raise ValueError(f"unknown mixer {cfg.mixer!r}; choose from {MIXERS}")


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d)
        self.mixer = make_mixer(cfg)
        self.ln2 = nn.LayerNorm(cfg.d)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d, cfg.d_ff), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d), nn.Dropout(cfg.dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class LM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d)
        self.pos = nn.Embedding(cfg.block_size, cfg.d)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab_size, bias=False)
        self.apply(self._init)
        self.head.weight = self.tok.weight

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def n_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied head counted once
        if non_embedding:
            n -= self.tok.weight.numel() + self.pos.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        T = idx.shape[1]
        assert T <= self.cfg.block_size, f"sequence {T} > block_size {self.cfg.block_size}"
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss
