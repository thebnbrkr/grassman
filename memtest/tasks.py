"""Synthetic recall tasks (Track B), generated on the device in batches.

Vocabulary layout (V = 8192): 0 unused, [1, 1024) filler noise, [1024, 4608)
keys, [4608, 8192) values. Each task returns
    x     (n, T) input tokens
    y     (n, T) target at query positions, -100 elsewhere (loss ignores it)
    dist  (n, T) query position minus the position of its key, -1 elsewhere

mqar      Multi-query associative recall as in Zoology (Arora et al. 2023): all
          key-value pairs first, then the rest of the sequence is filler with
          each key queried once at a random later position. Many pairs close
          together: tests how many items a memory holds.
far       Recall at distance: a few pairs scattered at random positions in a
          long filler sequence, all queried at the end. Distances span the
          whole sequence: tests how far back a memory reaches.

At a query position the model sees the key and must output its value.
"""
from __future__ import annotations

import torch

VOCAB = 8192
NOISE = (1, 1024)
KEYS = (1024, 4608)
VALUES = (4608, 8192)


def _sample_distinct(n: int, low: int, high: int, k: int, g: torch.Generator, device) -> torch.Tensor:
    """(n, k) distinct integers from [low, high) per row."""
    return torch.rand(n, high - low, generator=g, device=device).argsort(dim=1)[:, :k] + low


def _base(n: int, T: int, g: torch.Generator, device):
    x = torch.randint(*NOISE, (n, T), generator=g, device=device)
    y = torch.full((n, T), -100, dtype=torch.long, device=device)
    dist = torch.full((n, T), -1, dtype=torch.long, device=device)
    return x, y, dist


def mqar(n: int, T: int, pairs: int, g: torch.Generator, device="cpu"):
    assert 3 * pairs <= T, "need room for the pairs and one query per pair"
    x, y, dist = _base(n, T, g, device)
    keys = _sample_distinct(n, *KEYS, pairs, g, device)
    vals = torch.randint(*VALUES, (n, pairs), generator=g, device=device)
    rows = torch.arange(n, device=device)[:, None]
    kpos = 2 * torch.arange(pairs, device=device).expand(n, pairs)
    x[rows, kpos], x[rows, kpos + 1] = keys, vals
    qpos = _sample_distinct(n, 2 * pairs, T, pairs, g, device)
    x[rows, qpos], y[rows, qpos], dist[rows, qpos] = keys, vals, qpos - kpos
    return x, y, dist


def far(n: int, T: int, pairs: int, g: torch.Generator, device="cpu"):
    assert 3 * pairs <= T
    x, y, dist = _base(n, T, g, device)
    keys = _sample_distinct(n, *KEYS, pairs, g, device)
    vals = torch.randint(*VALUES, (n, pairs), generator=g, device=device)
    rows = torch.arange(n, device=device)[:, None]
    slots = (T - pairs) // 2                       # pair slots (2 tokens each) before the queries
    kpos = 2 * _sample_distinct(n, 0, slots, pairs, g, device)
    x[rows, kpos], x[rows, kpos + 1] = keys, vals
    order = torch.rand(n, pairs, generator=g, device=device).argsort(dim=1)
    qpos = (T - pairs + torch.arange(pairs, device=device)).expand(n, pairs)
    qk, qv, qkpos = keys.gather(1, order), vals.gather(1, order), kpos.gather(1, order)
    x[rows, qpos], y[rows, qpos], dist[rows, qpos] = qk, qv, qpos - qkpos
    return x, y, dist


TASKS = {"mqar": mqar, "far": far}

# distance buckets for reporting: [0, 16), [16, 64), ..., [4096, 16384)
BUCKETS = (0, 16, 64, 256, 1024, 4096, 16384)


def make(task: str, n: int, T: int, pairs: int, seed: int, device="cpu"):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return TASKS[task](n, T, pairs, g, device)
