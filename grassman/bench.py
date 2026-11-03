"""Speed and equivalence checks for the fast PairMixer (notebooks/speed.ipynb)."""
from __future__ import annotations

import time

import torch

from .mixers import PAIR_KINDS, PairMixer
from .model import LM, Config


def set_reference(model: torch.nn.Module, on: bool) -> None:
    """Switch every PairMixer between the fast path and the paper's order."""
    for m in model.modules():
        if isinstance(m, PairMixer):
            m.use_reference = on


def throughput(cfg: Config, device: torch.device, steps: int = 50, warmup: int = 10,
               batch: int = 32, reference: bool = False, compile: bool = False) -> dict:
    """Training tokens per second on random tokens (forward, backward, AdamW),
    with the same precision rule as training. warmup_s includes compilation."""
    from .train import pick_amp
    torch.manual_seed(0)
    model = LM(cfg).to(device)
    set_reference(model, reference)
    fwd = torch.compile(model) if compile else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    amp = pick_amp(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp is torch.float16)
    x = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size), device=device)
    t_start, t0 = time.time(), None
    for step in range(warmup + steps):
        if step == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            _, loss = fwd(x, x)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"tok_per_s": steps * batch * cfg.block_size / (time.time() - t0), "warmup_s": t0 - t_start,
            "amp": str(amp).replace("torch.", "") if amp else "fp32"}


def fast_vs_reference(device: torch.device, d: int = 256, T: int = 128, batch: int = 8) -> list[dict]:
    """Largest difference between the fast path and the paper's order, in
    outputs and in every parameter gradient, for each pair kind (fp32)."""
    rows = []
    for kind in PAIR_KINDS:
        for normalize in (True, False):
            torch.manual_seed(0)
            m = PairMixer(d, kind=kind, normalize=normalize, dropout=0.0).to(device)
            h = torch.randn(batch, T, d, device=device)
            out = {}
            for ref in (False, True):
                m.zero_grad()
                m.use_reference = ref
                y = m(h)
                y.square().mean().backward()
                out[ref] = (y.detach(), {n: p.grad.clone() for n, p in m.named_parameters()})
            grad_diff = max((out[False][1][n] - out[True][1][n]).abs().max().item() for n in out[False][1])
            rows.append({"kind": kind, "normalize": normalize,
                         "max_out_diff": (out[False][0] - out[True][0]).abs().max().item(),
                         "max_grad_diff": grad_diff})
    return rows
