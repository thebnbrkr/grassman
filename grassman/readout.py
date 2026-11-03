"""Read what the edge-selecting layers do (notebooks/readouts.ipynb).

  edge_usage      share of each offset per layer, and how peaked the choice is
  edge_ablation   loss increase when one edge of one layer is removed
  sin_by_boundary sin(angle) between t and t-1 after a sentence end vs elsewhere
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .mixers import PairMixer
from .model import LM, Config


def load(path: str, device: torch.device) -> LM:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["config"])
    cfg["windows"], cfg["attn_layers"] = tuple(cfg["windows"]), tuple(cfg.get("attn_layers", ()))
    model = LM(Config(**cfg))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def pair_layers(model: LM) -> list[tuple[int, PairMixer]]:
    return [(i, b.mixer) for i, b in enumerate(model.blocks) if isinstance(b.mixer, PairMixer)]


@torch.no_grad()
def _run(model: LM, x: torch.Tensor, batch: int, device: torch.device, fn) -> None:
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(device)
        model(xb)
        fn(xb)


@torch.no_grad()
def edge_usage(model: LM, x: torch.Tensor, batch: int = 32) -> dict:
    """Per layer: mean weight on each offset (over positions that have every
    offset available) and the mean weight of each token's top edge."""
    device = next(model.parameters()).device
    layers = pair_layers(model)
    for _, m in layers:
        m.record = True
    sums = {i: None for i, _ in layers}
    top = {i: 0.0 for i, _ in layers}
    n = {i: 0 for i, _ in layers}

    def collect(_):
        for i, m in layers:
            w = m.last["weights"][:, max(m.last["deltas"]):]       # positions with all edges
            flat = w.reshape(-1, w.shape[-1])
            sums[i] = flat.sum(0) if sums[i] is None else sums[i] + flat.sum(0)
            top[i] += flat.max(-1).values.sum().item()
            n[i] += flat.shape[0]

    _run(model, x, batch, device, collect)
    for _, m in layers:
        m.record = False
    return {i: {"deltas": m.last["deltas"], "share": (sums[i] / n[i]).tolist(), "top_weight": top[i] / n[i]}
            for i, m in layers}


@torch.no_grad()
def mean_nll(model: LM, x: torch.Tensor, y: torch.Tensor, batch: int = 32) -> float:
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for i in range(0, len(x), batch):
        logits, _ = model(x[i:i + batch].to(device))
        yb = y[i:i + batch].to(device)
        total += F.cross_entropy(logits.float().transpose(1, 2), yb, reduction="sum").item()
        count += yb.numel()
    return total / count


@torch.no_grad()
def edge_ablation(model: LM, x: torch.Tensor, y: torch.Tensor, batch: int = 32) -> dict:
    """Increase in mean NLL (nats) when each edge (offset) of each layer is
    removed, one at a time; the remaining edges are renormalised."""
    base = mean_nll(model, x, y, batch)
    out = {"base_nll": base, "base_ppl": math.exp(base), "delta_nll": {}}
    for i, m in pair_layers(model):
        row = {}
        for delta in (dl for dl in m.windows if dl < x.shape[1]):
            m.drop_edge = delta
            row[delta] = mean_nll(model, x, y, batch) - base
            m.drop_edge = None
        out["delta_nll"][i] = row
    return out


@torch.no_grad()
def sin_by_boundary(model: LM, x: torch.Tensor, boundary_ids: list[int], batch: int = 32) -> dict:
    """Per layer, mean sin(angle) between z_t and z_{t-1} when token t-1 ends a
    sentence, against all other positions. A readable 'something changed'
    signal should be higher after a boundary."""
    device = next(model.parameters()).device
    layers = pair_layers(model)
    for _, m in layers:
        m.record = True
    ids = torch.tensor(boundary_ids, device=device)
    acc = {i: [0.0, 0, 0.0, 0] for i, _ in layers}

    def collect(xb):
        after = torch.isin(xb[:, :-1], ids)                         # token t-1 ends a sentence
        for i, m in layers:
            e = m.last["deltas"].index(1)
            s = m.last["sin"][:, 1:, e]
            acc[i][0] += s[after].sum().item(); acc[i][1] += int(after.sum())
            acc[i][2] += s[~after].sum().item(); acc[i][3] += int((~after).sum())

    _run(model, x, batch, device, collect)
    for _, m in layers:
        m.record = False
    return {i: {"after_boundary": a[0] / max(a[1], 1), "elsewhere": a[2] / max(a[3], 1),
                "n_boundary": a[1]} for i, a in acc.items()}
