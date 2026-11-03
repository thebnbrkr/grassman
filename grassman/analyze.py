"""Summarise runs.jsonl: mean and 95% t-interval per mixer, plus paired
(same-seed) differences against the paper's Grassmann layer.

    python -m grassman.analyze results/runs.jsonl --tag main
    python -m grassman.analyze results/runs.jsonl --tag sweep --best-lr
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

# two-sided 95% Student-t critical values by degrees of freedom
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}
BASELINE = "grassmann"
ORDER = ("grassmann", "sym", "outer", "concat", "hyena", "swa", "attention")


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    for k in sorted(_T95):
        if df <= k:
            return _T95[k]
    return 1.96


def mean_ci(xs: list[float]) -> tuple[float, float]:
    """Mean and half-width of the 95% t-interval (nan with one value)."""
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, t95(n - 1) * sd / math.sqrt(n)


def load(path: str | Path, tag: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [r for r in rows if tag is None or r["tag"] == tag]


def name(r: dict) -> str:
    c = r["config"]
    return c["mixer"] + ("" if c["normalize"] else "-raw")


def best_lr(rows: list[dict]) -> dict[str, float]:
    """Per mixer, the learning rate with the lowest mean validation perplexity."""
    by = defaultdict(list)
    for r in rows:
        by[(name(r), r["lr"])].append(r["best_val_ppl"])
    best: dict[str, tuple[float, float]] = {}
    for (mixer, lr), ppls in by.items():
        m = sum(ppls) / len(ppls)
        if mixer not in best or m < best[mixer][1]:
            best[mixer] = (lr, m)
    return {k: v[0] for k, v in best.items()}


def _fmt(m: float, h: float) -> str:
    return f"{m:.1f}" if math.isnan(h) else f"{m:.1f} ± {h:.1f}"


def summary(rows: list[dict]) -> str:
    by = defaultdict(list)
    for r in rows:
        by[name(r)].append(r)
    names = sorted(by, key=lambda n: (ORDER.index(n.split("-")[0]) if n.split("-")[0] in ORDER else 99, n))
    ref = mean_ci([r["test_ppl"] for r in by["attention"]])[0] if "attention" in by else None

    lines = ["| mixer | seeds | lr | params | val ppl | test ppl | vs attention | tok/s |",
             "|---|---|---|---|---|---|---|---|"]
    for n in names:
        rs = by[n]
        val, test = mean_ci([r["best_val_ppl"] for r in rs]), mean_ci([r["test_ppl"] for r in rs])
        gap = f"{100 * (test[0] / ref - 1):+.1f}%" if ref else "—"
        lrs = ",".join(sorted({f"{r['lr']:g}" for r in rs}))
        tps = sum(r["tokens_per_s"] for r in rs) / len(rs)
        lines.append(f"| {n} | {len(rs)} | {lrs} | {rs[0]['params'] / 1e6:.2f}M | {_fmt(*val)} "
                     f"| {_fmt(*test)} | {gap} | {tps:,.0f} |")

    if BASELINE in by:
        lines += ["", f"Paired test-ppl difference, other − {BASELINE} (same seeds; "
                  "positive means the paper's Grassmann layer is better):", "",
                  "| other | pairs | difference | 95% CI excludes 0? |", "|---|---|---|---|"]
        base = {r["seed"]: r["test_ppl"] for r in by[BASELINE]}
        for n in names:
            if n == BASELINE:
                continue
            diffs = [r["test_ppl"] - base[r["seed"]] for r in by[n] if r["seed"] in base]
            if not diffs:
                continue
            m, h = mean_ci(diffs)
            sig = "n/a (1 pair)" if math.isnan(h) else ("yes" if abs(m) > h else "no")
            lines.append(f"| {n} | {len(diffs)} | {m:+.1f}" + ("" if math.isnan(h) else f" ± {h:.1f}")
                         + f" | {sig} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--best-lr", action="store_true", help="print {mixer: lr} as JSON")
    args = ap.parse_args(argv)
    rows = load(args.runs, args.tag)
    if not rows:
        raise SystemExit(f"no runs with tag {args.tag!r} in {args.runs}")
    print(json.dumps(best_lr(rows)) if args.best_lr else summary(rows))


if __name__ == "__main__":
    main()
