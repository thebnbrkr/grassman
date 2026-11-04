"""Train one recall model on fresh synthetic data and append a record to a
JSONL file.

    python -m memtest.train --model hyena+moment --task far --T 1024 --pairs 8 \
        --steps 8000 --lr 1e-3 --seed 0 --tag main --out /content/results/recall.jsonl

Training batches are drawn fresh each step from a stream seeded by --seed.
The dev and test sets are fixed (seeds DEV_SEED, TEST_SEED), the same for
every model and seed, so comparisons are paired by sequence. --no-test skips
the test set (LR sweep and pilots).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from grassman.train import git_commit, lr_at, pick_amp, pick_device, seed_all

from . import tasks
from .model import MODELS, Config, RecallLM

DEV_SEED, TEST_SEED = 1_000_003, 2_000_003


@torch.no_grad()
def evaluate(model, x, y, dist, batch: int, amp) -> dict:
    """Accuracy over query positions; correct and total counts per sequence and
    distance bucket (tasks.BUCKETS), for bootstraps over sequences."""
    model.eval()
    device = next(model.parameters()).device
    edges = torch.tensor(tasks.BUCKETS, device=device)
    nb = len(edges) - 1
    correct, total = [], []
    for i in range(0, len(x), batch):
        xb, yb, db = x[i:i + batch], y[i:i + batch], dist[i:i + batch]
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            logits, _ = model(xb, yb)                       # logits at query positions only, row-major
        sel = yb != -100
        hit = torch.zeros_like(sel)
        hit[sel] = logits.argmax(-1) == yb[sel]
        bucket = (torch.bucketize(db.clamp(min=0), edges, right=True) - 1).clamp(max=nb - 1)
        bucket[~sel] = nb                                   # non-query positions go to a dropped column
        zeros = torch.zeros(len(xb), nb + 1, device=device)
        correct.append(zeros.scatter_add(1, bucket, hit.float())[:, :nb])
        total.append(zeros.scatter_add(1, bucket, sel.float())[:, :nb])
    model.train()
    correct, total = torch.cat(correct), torch.cat(total)
    return {"acc": float(correct.sum() / total.sum().clamp(min=1)),
            "bucket_correct": correct.sum(0).tolist(), "bucket_total": total.sum(0).tolist(),
            "seq_bucket_correct": correct.int().tolist(), "seq_bucket_total": total.int().tolist()}


def run_key(r: dict) -> tuple:
    c = r["config"]
    return (r["tag"], c["model"], r["task"], r["T"], r["pairs"], c["d"], c["slots"], c["chunk"],
            r["seed"], r["lr"], r["steps"])


def already_done(out: Path, key: tuple) -> bool:
    if not out.exists():
        return False
    with out.open() as f:
        return any(run_key(json.loads(line)) == key for line in f if line.strip())


def main(argv=None) -> dict | None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="hyena+attn", choices=list(MODELS))
    ap.add_argument("--task", default="far", choices=list(tasks.TASKS))
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-n", type=int, default=2000, help="sequences in the dev and the test set")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--out", default="results/recall.jsonl")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"])
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args(argv)

    cfg = Config(model=a.model, max_len=a.T, d=a.d, n_heads=4, d_ff=4 * a.d, window=a.window,
                 chunk=a.chunk, slots=a.slots)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stub = {"tag": a.tag, "config": cfg.to_dict(), "task": a.task, "T": a.T, "pairs": a.pairs,
            "seed": a.seed, "lr": a.lr, "steps": a.steps}
    if a.skip_existing and already_done(out, run_key(stub)):
        print("skip (already in", out, "):", run_key(stub))
        return None

    seed_all(a.seed)
    device = pick_device(a.device)
    amp = pick_amp(device, a.amp)
    model = RecallLM(cfg).to(device)
    fwd = torch.compile(model) if a.compile else model
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": rest, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)

    dev = tasks.make(a.task, a.eval_n, a.T, a.pairs, DEV_SEED, device)
    g = torch.Generator(device=device)
    g.manual_seed(a.seed)
    warmup = max(1, a.steps // 20)
    history, t0, tokens = [], time.time(), 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f">> {a.model} ({cfg.pattern}) task={a.task} T={a.T} pairs={a.pairs} lr={a.lr} seed={a.seed} "
          f"params={model.n_params():,} amp={amp}")
    for step in range(a.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, a.steps, a.lr, warmup)
        x, y, _ = tasks.TASKS[a.task](a.batch, a.T, a.pairs, g, device)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            _, loss = fwd(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        tokens += x.numel()
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            d = evaluate(model, *dev, a.eval_batch, amp)
            history.append({"step": step + 1, "loss": float(loss), "dev_acc": d["acc"]})
            print(f"   step {step + 1:6d}  loss {float(loss):.3f}  dev acc {d['acc']:.4f}", flush=True)
    wall = time.time() - t0
    rec = {**stub, "params": model.n_params(), "state_numbers": model.state_numbers(a.T),
           "dev_acc": history[-1]["dev_acc"],
           "dev_detail": {k: d[k] for k in ("bucket_correct", "bucket_total")},
           "history": history, "tokens_per_s": tokens / wall, "wall_s": wall,
           "peak_mem_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
           "device": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
           "amp": str(amp).replace("torch.", "") if amp else "fp32", "compile": a.compile,
           "torch": torch.__version__, "commit": git_commit(), "test_acc": None}
    if not a.no_test:
        test = tasks.make(a.task, a.eval_n, a.T, a.pairs, TEST_SEED, device)
        t = evaluate(model, *test, a.eval_batch, amp)
        rec["test_acc"], rec["test_detail"] = t["acc"], t
        print(f"   test acc {t['acc']:.4f}")
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


if __name__ == "__main__":
    main()
