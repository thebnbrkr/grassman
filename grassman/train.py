"""Train one LM on Wikitext-2 and append one JSON line of results.

    python -m grassman.train --mixer grassmann --seed 0 --lr 1e-3 --tag main

Model selection is on validation only: test perplexity is computed at the
epoch with the best validation perplexity, never chosen by test. With
--no-test (the LR sweep) the test set is never evaluated at all. At that
epoch the record also stores per-article test NLL for the cluster bootstrap,
NLL by position in the block and by target-token frequency.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data
from .model import LM, MIXERS, Config

REPO = Path(__file__).resolve().parent.parent


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step: int, total: int, lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    """Linear warm-up, then cosine decay to min_ratio * lr."""
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def make_optimizer(model: LM, lr: float, weight_decay: float) -> torch.optim.AdamW:
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95))


@torch.no_grad()
def evaluate(model: LM, x: torch.Tensor, y: torch.Tensor, batch_size: int,
             device: torch.device, amp_dtype) -> float:
    """Mean next-token NLL (nats) over every token of the split."""
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(x), batch_size):
        xb, yb = x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            _, loss = model(xb, yb)
        total += loss.item() * yb.numel()
        count += yb.numel()
    model.train()
    return total / count


FREQ_BUCKETS = (("top-1k", 0, 1_000), ("1k-10k", 1_000, 10_000), ("rare", 10_000, None))


def frequency_rank(train_stream: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """rank[token] = 0 for the most frequent training token, 1 for the next, ..."""
    order = torch.argsort(torch.bincount(train_stream, minlength=vocab_size), descending=True, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order))
    return rank


@torch.no_grad()
def evaluate_detail(model: LM, x: torch.Tensor, y: torch.Tensor, doc: torch.Tensor,
                    rank: torch.Tensor, batch_size: int, device: torch.device, amp_dtype) -> dict:
    """Per-token NLL, summed per article (for the cluster bootstrap), averaged
    per position in the block (local mixers only see 16 tokens back) and per
    target-frequency bucket. doc holds the article of each target in y."""
    model.eval()
    nll = []
    for i in range(0, len(x), batch_size):
        xb, yb = x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, _ = model(xb)
        nll.append(F.cross_entropy(logits.float().transpose(1, 2), yb, reduction="none").cpu())
    model.train()
    nll = torch.cat(nll).double()                       # (n_blocks, block_size)
    n_docs = int(doc.max()) + 1
    doc_nll = torch.zeros(n_docs, dtype=torch.float64).index_add_(0, doc.flatten(), nll.flatten())
    doc_tokens = torch.bincount(doc.flatten(), minlength=n_docs)
    r = rank[y]
    buckets = {}
    for name, lo, hi in FREQ_BUCKETS:
        mask = (r >= lo) & (r < hi) if hi else (r >= lo)
        buckets[name] = {"nll": nll[mask].mean().item(), "tokens": int(mask.sum())}
    return {"nll": nll.mean().item(),
            "doc_nll": doc_nll.tolist(), "doc_tokens": doc_tokens.tolist(),
            "pos_nll": nll.mean(0).tolist(), "buckets": buckets}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def run_key(r: dict) -> tuple:
    return (r["tag"], r["config"]["mixer"], r["config"]["normalize"], r["seed"], r["lr"], r["epochs"])


def already_done(out: Path, key: tuple) -> bool:
    if not out.exists():
        return False
    for line in out.read_text().splitlines():
        if line.strip() and run_key(json.loads(line)) == key:
            return True
    return False


def main(argv=None) -> dict | None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mixer", choices=MIXERS, default="grassmann")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-normalize", action="store_true", help="skip L2-normalising pair features")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results/runs.jsonl")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-test", action="store_true",
                    help="never touch the test set (the LR sweep uses this)")
    args = ap.parse_args(argv)

    cfg = Config(mixer=args.mixer, block_size=args.block_size, d=args.d, d_ff=4 * args.d,
                 n_layers=args.n_layers, normalize=not args.no_normalize, dropout=args.dropout)
    out = Path(args.out)
    key = (args.tag, cfg.mixer, cfg.normalize, args.seed, args.lr, args.epochs)
    if args.skip_existing and already_done(out, key):
        print(f"skip {key}: already in {out}")
        return None

    device = pick_device(args.device)
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    seed_all(args.seed)

    splits = data.load(args.data_dir)
    train_x, train_y = data.blocks(splits["train"], cfg.block_size)
    val = data.blocks(splits["validation"], cfg.block_size)
    test = data.blocks(splits["test"], cfg.block_size)
    test_doc = data.blocks(data.load_docs(args.data_dir)["test"], cfg.block_size)[1]
    rank = frequency_rank(splits["train"], cfg.vocab_size)

    model = LM(cfg).to(device)
    fwd = torch.compile(model) if args.compile else model
    opt = make_optimizer(model, args.lr, args.weight_decay)
    steps_per_epoch = math.ceil(len(train_x) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    print(f"{cfg.mixer}: {model.n_params():,} params "
          f"({model.n_params(non_embedding=True):,} non-embedding), "
          f"{total_steps} steps on {device}, amp={amp_dtype}")

    gen = torch.Generator().manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, best = [], {"val_ppl": float("inf")}
    step, train_time, train_tokens = 0, 0.0, 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(len(train_x), generator=gen)
        losses, t0 = [], time.time()
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = train_x[idx].to(device), train_y[idx].to(device)
            for group in opt.param_groups:
                group["lr"] = lr_at(step, total_steps, args.lr, args.warmup)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                _, loss = fwd(xb, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            train_tokens += yb.numel()
            step += 1
            if step >= total_steps:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time += time.time() - t0

        val_ppl = math.exp(evaluate(model, *val, 64, device, amp_dtype))
        row = {"epoch": epoch, "step": step, "train_loss": float(np.mean(losses)), "val_ppl": val_ppl}
        if val_ppl < best["val_ppl"]:
            best = {"val_ppl": val_ppl, "epoch": epoch, "test_ppl": None, "test_detail": None}
            if not args.no_test:
                detail = evaluate_detail(model, *test, test_doc, rank, 64, device, amp_dtype)
                best["test_ppl"], best["test_detail"] = math.exp(detail["nll"]), detail
                row["test_ppl"] = best["test_ppl"]
        history.append(row)
        print(f"epoch {epoch:>3}  train {row['train_loss']:.3f}  val ppl {val_ppl:8.2f}"
              + (f"  test ppl {row['test_ppl']:8.2f}" if "test_ppl" in row else ""))
        if step >= total_steps:
            break

    result = {
        "tag": args.tag, "seed": args.seed, "lr": args.lr, "epochs": args.epochs,
        "config": cfg.to_dict(),
        "params": model.n_params(), "params_non_embedding": model.n_params(non_embedding=True),
        "best_val_ppl": best["val_ppl"], "best_epoch": best["epoch"], "test_ppl": best["test_ppl"],
        "steps": step, "tokens_per_s": train_tokens / train_time,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
        "wall_s": time.time() - t_start,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else device.type,
        "torch": torch.__version__, "commit": git_commit(), "history": history,
        "test_detail": best["test_detail"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(result) + "\n")
    test_msg = "test not run" if best["test_ppl"] is None else f"test ppl {best['test_ppl']:.2f}"
    print(f"done: best val ppl {best['val_ppl']:.2f} (epoch {best['epoch']}), "
          f"{test_msg}, {result['tokens_per_s']:,.0f} tok/s -> {out}")
    return result


if __name__ == "__main__":
    main()
