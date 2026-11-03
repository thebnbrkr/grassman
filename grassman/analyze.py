"""Summarise runs.jsonl following PROTOCOL.md.

    python -m grassman.analyze results/runs.jsonl --tag main
    python -m grassman.analyze results/runs.jsonl --tag sweep --best-lr

Two kinds of uncertainty are reported:
  - training noise: mean and 95% Student-t interval over seeds;
  - test-set sampling: a paired cluster bootstrap over Wikipedia articles
    (tokens of one article are not independent), on the seed-averaged
    per-article NLL, both systems on the same resamples.
The comparisons fixed in PROTOCOL.md are the confirmatory family and are
Holm-corrected; every other comparison is exploratory.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# two-sided 95% Student-t critical values by degrees of freedom
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}
BASELINE = "grassmann"
CONFIRMATORY = ("sym", "outer", "concat", "attention")   # fixed in PROTOCOL.md
ORDER = ("grassmann", "sym", "outer", "concat", "concat32", "hyena", "swa", "attention")
N_RESAMPLES = 10_000
LOCAL = 16   # every windowed mixer sees 16 tokens back


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


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted, running = [0.0] * m, 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * pvals[i]))
        adjusted[i] = running
    return adjusted


def _doc_arrays(rows: list[dict], seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Seed-averaged per-article NLL sums and the article token counts."""
    by_seed = {r["seed"]: r["test_detail"] for r in rows}
    nll = np.mean([by_seed[s]["doc_nll"] for s in seeds], axis=0)
    return nll, np.asarray(by_seed[seeds[0]]["doc_tokens"], dtype=float)


def paired_bootstrap(rows_a: list[dict], rows_b: list[dict], n: int = N_RESAMPLES,
                     seed: int = 0) -> dict:
    """Perplexity difference b − a with a 95% cluster-bootstrap interval and a
    two-sided p-value (Phipson & Smyth 2010), resampling whole articles."""
    seeds = sorted({r["seed"] for r in rows_a} & {r["seed"] for r in rows_b})
    nll_a, tok = _doc_arrays(rows_a, seeds)
    nll_b, _ = _doc_arrays(rows_b, seeds)
    keep = tok > 0
    nll_a, nll_b, tok = nll_a[keep], nll_b[keep], tok[keep]
    idx = np.random.default_rng(seed).integers(0, len(tok), size=(n, len(tok)))
    t = tok[idx].sum(1)
    diff = np.exp(nll_b[idx].sum(1) / t) - np.exp(nll_a[idx].sum(1) / t)
    observed = math.exp(nll_b.sum() / tok.sum()) - math.exp(nll_a.sum() / tok.sum())
    tail = min((diff <= 0).sum(), (diff >= 0).sum())
    return {"diff": observed, "lo": float(np.percentile(diff, 2.5)), "hi": float(np.percentile(diff, 97.5)),
            "p": min(1.0, 2 * (tail + 1) / (n + 1)), "seeds": seeds, "articles": int(len(tok))}


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


def _sorted_names(names) -> list[str]:
    base = lambda n: n.split("-")[0]
    return sorted(names, key=lambda n: (ORDER.index(base(n)) if base(n) in ORDER else 99, n))


def _main_table(by: dict, names: list[str]) -> list[str]:
    has_test = all(r["test_ppl"] is not None for rs in by.values() for r in rs)
    ref = mean_ci([r["test_ppl"] for r in by["attention"]])[0] if has_test and "attention" in by else None
    lines = ["| mixer | seeds | lr | params | val ppl | test ppl | vs attention | tok/s |",
             "|---|---|---|---|---|---|---|---|"]
    for n in names:
        rs = by[n]
        val = mean_ci([r["best_val_ppl"] for r in rs])
        test = mean_ci([r["test_ppl"] for r in rs]) if has_test else None
        gap = f"{100 * (test[0] / ref - 1):+.1f}%" if ref else "—"
        lrs = ",".join(sorted({f"{r['lr']:g}" for r in rs}))
        tps = sum(r["tokens_per_s"] for r in rs) / len(rs)
        lines.append(f"| {n} | {len(rs)} | {lrs} | {rs[0]['params'] / 1e6:.2f}M | {_fmt(*val)} "
                     f"| {_fmt(*test) if test else '—'} | {gap} | {tps:,.0f} |")
    return lines


def _comparison_tables(by: dict, names: list[str]) -> list[str]:
    base = by[BASELINE]
    base_by_seed = {r["seed"]: r["test_ppl"] for r in base}
    results = {}
    for n in names:
        if n == BASELINE:
            continue
        seed_diffs = [r["test_ppl"] - base_by_seed[r["seed"]] for r in by[n] if r["seed"] in base_by_seed]
        if seed_diffs:
            results[n] = {"boot": paired_bootstrap(base, by[n]), "seed": seed_diffs}
    confirm = [n for n in CONFIRMATORY if n in results]
    adjusted = dict(zip(confirm, holm([results[n]["boot"]["p"] for n in confirm])))

    head = ["| other | difference (95% cluster CI) | p | {p_col} | per-seed differences | wins |",
            "|---|---|---|---|---|---|"]

    def row(n, padj):
        b, sd = results[n]["boot"], results[n]["seed"]
        wins = sum(d > 0 for d in sd)
        return (f"| {n} | {b['diff']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}] | {b['p']:.4f} | {padj} "
                f"| {', '.join(f'{d:+.1f}' for d in sd)} | {wins}/{len(sd)} |")

    lines = ["", f"Test-ppl difference, other − {BASELINE}; positive means the paper's layer is better. "
             f"Cluster bootstrap over test articles, {N_RESAMPLES:,} resamples, seed-averaged NLL. "
             "'wins' counts seeds where the paper's layer is better.", "",
             "**Confirmatory family (PROTOCOL.md), Holm-corrected:**", "",
             head[0].format(p_col="Holm p"), head[1]]
    lines += [row(n, f"{adjusted[n]:.4f}") for n in confirm]
    exploratory = [n for n in results if n not in confirm]
    if exploratory:
        lines += ["", "**Exploratory (not corrected, not used for claims):**", "",
                  head[0].format(p_col="—"), head[1]]
        lines += [row(n, "—") for n in exploratory]
    return lines


def _breakdown_tables(by: dict, names: list[str]) -> list[str]:
    lines = ["", f"Test perplexity by position in the 128-token block (positions < {LOCAL} have the same "
             "context for every mixer) and by how frequent the target token is in training "
             "(mean over seeds):", "",
             f"| mixer | pos 0–{LOCAL - 1} | pos {LOCAL}–127 | top-1k tokens | 1k–10k | rare |",
             "|---|---|---|---|---|---|"]
    for n in names:
        ds = [r["test_detail"] for r in by[n]]
        pos = np.mean([d["pos_nll"] for d in ds], axis=0)
        cells = [math.exp(pos[:LOCAL].mean()), math.exp(pos[LOCAL:].mean())]
        cells += [math.exp(np.mean([d["buckets"][b]["nll"] for d in ds])) for b in ("top-1k", "1k-10k", "rare")]
        lines.append(f"| {n} | " + " | ".join(f"{c:.1f}" for c in cells) + " |")
    return lines


def summary(rows: list[dict]) -> str:
    by = defaultdict(list)
    for r in rows:
        by[name(r)].append(r)
    names = _sorted_names(by)
    lines = _main_table(by, names)
    if all(r.get("test_detail") for rs in by.values() for r in rs):
        if BASELINE in by:
            lines += _comparison_tables(by, names)
        lines += _breakdown_tables(by, names)
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
