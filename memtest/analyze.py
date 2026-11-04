"""Tables for the recall runs (Track B).

    python -m memtest.analyze results/recall.jsonl --tag sweep --best-lr --task far --T 1024
    python -m memtest.analyze results/recall.jsonl --tag main --task far --T 4096 --confirmatory

Accuracy is over query tokens. "far" accuracy counts only queries whose key is
at least --min-dist tokens back. Intervals over seeds are 95% t-intervals;
model comparisons are paired cluster bootstraps over test sequences
(seed-averaged counts, 10,000 resamples), with Holm correction for the
confirmatory family in memtest/PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from grassman.analyze import N_RESAMPLES, holm, mean_ci

from .tasks import BUCKETS

# (a, b): difference reported as b - a, in accuracy points
CONFIRMATORY = [("hyena", "hyena+moment"), ("hyena+attn", "hyena+moment"), ("hyena+attn", "hyena+delta")]
REFERENCE = "hyena+attn"
ORDER = ["hyena", "hyena+attn", "hyena+linear", "hyena+delta", "hyena+moment", "attention", "swa",
         "hyena+attn@0", "hyena+attn@3"]


def load(path, tag=None, task=None, T=None, pairs=None) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return [r for r in rows if (tag is None or r["tag"] == tag) and (task is None or r["task"] == task)
            and (T is None or r["T"] == T) and (pairs is None or r["pairs"] == pairs)]


def name(r: dict) -> str:
    return r["config"]["model"]


def best_lr(rows: list[dict]) -> dict[str, float]:
    """Per model, the learning rate with the highest final dev accuracy."""
    best = {}
    for r in rows:
        n = name(r)
        if n not in best or r["dev_acc"] > best[n][0]:
            best[n] = (r["dev_acc"], r["lr"])
    return {n: lr for n, (_, lr) in best.items()}


def _cols(min_dist: int) -> list[int]:
    return [i for i in range(len(BUCKETS) - 1) if BUCKETS[i] >= min_dist]


def seq_counts(r: dict, min_dist: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per test sequence: correct and total queries with distance >= min_dist."""
    c = np.asarray(r["test_detail"]["seq_bucket_correct"], dtype=float)[:, _cols(min_dist)]
    t = np.asarray(r["test_detail"]["seq_bucket_total"], dtype=float)[:, _cols(min_dist)]
    return c.sum(1), t.sum(1)


def acc(r: dict, min_dist: int = 0) -> float:
    c, t = seq_counts(r, min_dist)
    return float(c.sum() / max(1.0, t.sum()))


def paired_bootstrap(rows_a, rows_b, min_dist: int = 0, n: int = N_RESAMPLES, seed: int = 0) -> dict:
    """Accuracy difference b - a (points) over shared seeds, resampling test sequences."""
    seeds = sorted({r["seed"] for r in rows_a} & {r["seed"] for r in rows_b})
    by_a, by_b = {r["seed"]: r for r in rows_a}, {r["seed"]: r for r in rows_b}
    ca = np.mean([seq_counts(by_a[s], min_dist)[0] for s in seeds], axis=0)
    cb = np.mean([seq_counts(by_b[s], min_dist)[0] for s in seeds], axis=0)
    tot = seq_counts(by_a[seeds[0]], min_dist)[1]
    keep = tot > 0
    ca, cb, tot = ca[keep], cb[keep], tot[keep]
    idx = np.random.default_rng(seed).integers(0, len(tot), size=(n, len(tot)))
    t = tot[idx].sum(1)
    diff = 100 * (cb[idx].sum(1) - ca[idx].sum(1)) / t
    observed = 100 * (cb.sum() - ca.sum()) / tot.sum()
    tail = min((diff <= 0).sum(), (diff >= 0).sum())
    p = 1.0 if observed == 0 and np.all(diff == 0) else min(1.0, 2 * (tail + 1) / (n + 1))
    return {"diff": float(observed), "lo": float(np.percentile(diff, 2.5)), "hi": float(np.percentile(diff, 97.5)),
            "p": p, "seeds": seeds, "sequences": int(len(tot))}


def _group(rows):
    by = defaultdict(list)
    for r in rows:
        by[name(r)].append(r)
    return dict(sorted(by.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))


def _fmt(m, h, scale=100):
    return f"{scale * m:.1f}" + ("" if np.isnan(h) else f" ± {scale * h:.1f}")


def main_table(by: dict, min_dist: int) -> list[str]:
    heads = [f"{BUCKETS[i]}–{BUCKETS[i + 1]}" for i in range(len(BUCKETS) - 1)]
    lines = ["| model | layers | seeds | lr | params | state (MB, bf16) | test acc % | "
             f"far acc % (≥{min_dist}) | " + " | ".join(heads) + " | tok/s |",
             "|" + "---|" * (9 + len(heads))]
    for n, rs in by.items():
        r0 = rs[0]
        bc = np.sum([r["test_detail"]["bucket_correct"] for r in rs], axis=0)
        bt = np.sum([r["test_detail"]["bucket_total"] for r in rs], axis=0)
        buckets = [f"{100 * c / t:.1f}" if t else "—" for c, t in zip(bc, bt)]
        lines.append(
            f"| {n} | {r0['config']['pattern']} | {len(rs)} | {r0['lr']:g} | {r0['params'] / 1e6:.2f}M | "
            f"{2 * r0['state_numbers'] / 1e6:.2f} | {_fmt(*mean_ci([acc(r) for r in rs]))} | "
            f"{_fmt(*mean_ci([acc(r, min_dist) for r in rs]))} | " + " | ".join(buckets) +
            f" | {np.mean([r['tokens_per_s'] for r in rs]):,.0f} |")
    return lines


def _row(a, b, boot, by, min_dist, adj=None):
    seeds = boot["seeds"]
    sa, sb = {r["seed"]: r for r in by[a]}, {r["seed"]: r for r in by[b]}
    per = ", ".join(f"{100 * (acc(sb[s], min_dist) - acc(sa[s], min_dist)):+.1f}" for s in seeds)
    adj_s = "—" if adj is None else f"{adj:.4f}"
    return (f"| {b} − {a} | {boot['diff']:+.2f} [{boot['lo']:+.2f}, {boot['hi']:+.2f}] | {boot['p']:.4f} | "
            f"{adj_s} | {per} |")


def comparison_tables(by: dict, min_dist: int, confirmatory: bool) -> list[str]:
    head = ["| comparison | difference, points (95% cluster CI) | p | Holm p | per-seed differences |",
            "|---|---|---|---|---|"]
    lines = []
    fam = [(a, b) for a, b in CONFIRMATORY if a in by and b in by]
    if confirmatory and fam:
        boots = [paired_bootstrap(by[a], by[b], min_dist) for a, b in fam]
        adj = holm([b["p"] for b in boots])
        lines += [f"**Confirmatory family (PROTOCOL.md), far accuracy (distance ≥ {min_dist}), Holm-corrected:**",
                  "", *head, *(_row(a, b, bt, by, min_dist, h) for (a, b), bt, h in zip(fam, boots, adj)), ""]
    others = [n for n in by if n != REFERENCE and (not confirmatory or all(n != b for _, b in fam))]
    if REFERENCE in by and others:
        title = "Exploratory" if confirmatory else "All comparisons exploratory (not corrected)"
        lines += [f"**{title}: each model − {REFERENCE}, far accuracy (distance ≥ {min_dist}):**", "", *head]
        lines += [_row(REFERENCE, n, paired_bootstrap(by[REFERENCE], by[n], min_dist), by, min_dist)
                  for n in others]
    return lines


def dev_table(rows: list[dict]) -> str:
    """LR sweep: final dev accuracy per model and learning rate."""
    lrs = sorted({r["lr"] for r in rows})
    by = _group(rows)
    lines = ["| model | " + " | ".join(f"lr {lr:g}" for lr in lrs) + " |", "|---|" + "---|" * len(lrs)]
    for n, rs in by.items():
        got = {r["lr"]: r["dev_acc"] for r in rs}
        lines.append(f"| {n} | " + " | ".join(f"{100 * got[lr]:.1f}" if lr in got else "—" for lr in lrs) + " |")
    return "\n".join(lines)


def summary(rows: list[dict], min_dist: int = 1024, confirmatory: bool = False) -> str:
    rows = [r for r in rows if r.get("test_detail")]
    if not rows:
        return "no runs with a test evaluation"
    by = _group(rows)
    return "\n".join(main_table(by, min_dist) + [""] + comparison_tables(by, min_dist, confirmatory))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs")
    ap.add_argument("--tag")
    ap.add_argument("--task")
    ap.add_argument("--T", type=int)
    ap.add_argument("--pairs", type=int)
    ap.add_argument("--min-dist", type=int, default=1024)
    ap.add_argument("--best-lr", action="store_true", help="print the best LR per model as JSON")
    ap.add_argument("--dev", action="store_true", help="print the dev-accuracy table of the LR sweep")
    ap.add_argument("--confirmatory", action="store_true", help="Holm-corrected PROTOCOL.md family")
    a = ap.parse_args(argv)
    rows = load(a.runs, a.tag, a.task, a.T, a.pairs)
    if a.best_lr:
        print(json.dumps(best_lr(rows)))
    elif a.dev:
        print(dev_table(rows))
    else:
        print(summary(rows, a.min_dist, a.confirmatory))


if __name__ == "__main__":
    main()
