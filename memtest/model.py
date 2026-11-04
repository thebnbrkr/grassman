"""Small recall models: a stack of pre-LN blocks (mixer + MLP), one mixer per
character of a layer pattern, with learned positions and tied embeddings
(the same block as grassman/model.py).

    H  HyenaLongMixer (implicit long filter)   L  LinearAttention
    A  full attention                          D  DeltaRuleMixer
    W  sliding-window attention                M  MomentMemory
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from grassman.mixers import AttentionMixer

from . import layers
from .tasks import VOCAB

# model name -> layer pattern (4 layers); the memory layer sits at layer 3 (index 2)
MODELS = {
    "hyena": "HHHH",
    "hyena+attn": "HHAH",
    "hyena+linear": "HHLH",
    "hyena+delta": "HHDH",
    "hyena+moment": "HHMH",
    "attention": "AAAA",
    "swa": "WWWW",
    # exploratory: where should the one attention layer go?
    "hyena+attn@0": "AHHH",
    "hyena+attn@3": "HHHA",
}


@dataclass
class Config:
    model: str = "hyena+attn"
    max_len: int = 512
    d: int = 128
    n_heads: int = 4
    d_ff: int = 512
    window: int = 128           # W layers
    chunk: int = 64             # L, D and M layers
    slots: int = 4              # M layers: summary vectors per chunk
    dropout: float = 0.0
    vocab_size: int = VOCAB

    @property
    def pattern(self) -> str:
        return MODELS[self.model]

    def to_dict(self) -> dict:
        return {**asdict(self), "pattern": self.pattern}


def make_mixer(kind: str, cfg: Config) -> nn.Module:
    d, h, p = cfg.d, cfg.n_heads, cfg.dropout
    return {
        "H": lambda: layers.HyenaLongMixer(d, cfg.max_len, dropout=p),
        "A": lambda: AttentionMixer(d, h, window=None, dropout=p),
        "W": lambda: AttentionMixer(d, h, window=cfg.window, dropout=p),
        "L": lambda: layers.LinearAttention(d, h, chunk=cfg.chunk, dropout=p),
        "D": lambda: layers.DeltaRuleMixer(d, h, chunk=cfg.chunk, dropout=p),
        "M": lambda: layers.MomentMemory(d, h, chunk=cfg.chunk, slots=cfg.slots, dropout=p),
    }[kind]()


class Block(nn.Module):
    def __init__(self, kind: str, cfg: Config):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d), nn.LayerNorm(cfg.d)
        self.mixer = make_mixer(kind, cfg)
        self.ffn = nn.Sequential(nn.Linear(cfg.d, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d),
                                 nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.mixer(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class RecallLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d)
        self.pos = nn.Embedding(cfg.max_len, cfg.d)
        self.blocks = nn.ModuleList(Block(k, cfg) for k in cfg.pattern)
        self.ln_f = nn.LayerNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab_size, bias=False)
        self.apply(self._init)
        self.head.weight = self.tok.weight

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state_numbers(self, T: int) -> int:
        """Numbers kept per sequence to generate the next token (all mixers)."""
        total = 0
        for b in self.blocks:
            m = b.mixer
            total += layers.attention_state(m, T) if isinstance(m, AttentionMixer) else m.state_numbers(T)
        return total

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        if targets is None:
            return self.head(x), None
        # only query positions carry a target: compute logits there only
        sel = targets != -100
        logits = self.head(x[sel])
        return logits, F.cross_entropy(logits.float(), targets[sel])
