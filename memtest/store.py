"""Track A: is linear attention's associative matrix any good as a store?

Task: store N items (key vector -> item id), then look items up with a query
vector (the key itself plus noise, or a paraphrase). Every method returns a
ranking of item ids; we report recall@1, recall@10 and MRR@10, and the bytes
the method keeps.

Methods
    matrix   S (dv x dk), item i written as code(i) k_i^T; read r = S q, then rank
             ids by r . code(j). Codes are pseudo-random +-1 vectors regenerated
             from the item id (no storage), so S is all the memory there is.
             Writes: add (S += c k^T), delta (S += (c - S k) k^T, 1 or 3 passes),
             lstsq (best linear map, needs all items at once: an upper bound).
             +c = keys and queries centred on a calibration mean, then re-normalised.
    pq       FAISS product quantisation with the largest code that fits the
             matrix's byte budget (codebooks counted). The fair competitor.
    fifo     the most recent items that fit the budget, fp16, exact search.
    flat     exact FAISS search over all items (no budget: the ceiling).
    hnsw     FAISS HNSW graph (no budget: what a vector DB does).

Experiments (each writes JSONL rows; see memtest/PROTOCOL.md)
    capacity   A1: random keys, d in {64, 256, 1024}, N/d from 1/8 to 8
    real       A2: Quora duplicate questions (key = question, query = its
               paraphrase), MiniLM-L6 embeddings, budgets 98 KB / 590 KB / 2.4 MB
    updates    A3: overwrite and delete items; stale answers; float drift
    cost       A5: write and read time against N
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from grassman.analyze import N_RESAMPLES, holm, mean_ci

TOPK = 10
STREAMING = ("add", "add+c", "delta", "delta+c", "delta3+c")
MATRIX = STREAMING + ("lstsq+c",)
BUDGET_DV = (64, 384, 1536)          # real data: dv for dk = 384 -> 98 KB, 590 KB, 2.36 MB
CONFIRM_DV, CONFIRM_N = 384, (192, 384, 768, 1536)


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def codes(n: int, dv: int, seed: int = 12345) -> np.ndarray:
    """Pseudo-random +-1/sqrt(dv) codes for ids 0..n-1, regenerable from (n, seed)."""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=(n, dv), dtype=np.int8) * 2 - 1).astype(np.float32) / math.sqrt(dv)


# ---------------------------------------------------------------- the matrix store

class MatrixStore:
    def __init__(self, dk: int, dv: int, rule: str = "add", passes: int = 1, mean: np.ndarray | None = None,
                 dtype=np.float32):
        self.rule, self.passes, self.mean = rule, passes, mean
        self.S = np.zeros((dv, dk), dtype=dtype)

    @property
    def nbytes(self) -> int:
        return self.S.nbytes + (0 if self.mean is None else self.mean.nbytes)

    def prep(self, x: np.ndarray) -> np.ndarray:
        return unit(x if self.mean is None else x - self.mean).astype(self.S.dtype)

    def write(self, K: np.ndarray, C: np.ndarray) -> None:
        K, C = self.prep(K), C.astype(self.S.dtype)
        if self.rule == "add":
            self.S += C.T @ K
        elif self.rule == "delta":
            for _ in range(self.passes):
                for k, c in zip(K, C):
                    self.S += np.outer(c - self.S @ k, k)
        elif self.rule == "lstsq":
            dk = K.shape[1]
            self.S = (C.T @ K) @ np.linalg.inv(K.T @ K + 1e-3 * np.eye(dk, dtype=K.dtype))
        else:
            raise ValueError(self.rule)

    def overwrite(self, K: np.ndarray, C_old: np.ndarray, C_new: np.ndarray) -> None:
        """Point keys at new values: add subtracts the old value (it must be
        known); delta writes the new value over whatever the key returns."""
        K = self.prep(K)
        if self.rule == "add":
            self.S += (C_new - C_old).astype(self.S.dtype).T @ K
        else:
            for k, c in zip(K, C_new.astype(self.S.dtype)):
                self.S += np.outer(c - self.S @ k, k)

    def delete(self, K: np.ndarray, C: np.ndarray) -> None:
        self.overwrite(K, C, np.zeros_like(C))

    def read(self, Q: np.ndarray) -> np.ndarray:
        return self.prep(Q) @ self.S.T


def rank_ids(R: np.ndarray, C: np.ndarray, k: int = TOPK) -> np.ndarray:
    """Top-k ids by R . C^T, best first; R (nq, dv), C (N, dv)."""
    scores = R @ C.T
    k = min(k, scores.shape[1])
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, top, 1), axis=1)
    return np.take_along_axis(top, order, 1)


def metrics(ids: np.ndarray, target: np.ndarray) -> dict:
    hit = ids == target[:, None]
    rank = np.where(hit.any(1), hit.argmax(1) + 1, 0)
    return {"recall1": float((rank == 1).mean()), "recall10": float((rank > 0).mean()),
            "mrr10": float(np.where(rank > 0, 1.0 / np.maximum(rank, 1), 0).mean()),
            "hits1": (rank == 1).astype(int).tolist()}


# ---------------------------------------------------------------- baselines

def _faiss():
    import faiss  # noqa: PLC0415  (only needed on Colab: pip install faiss-cpu)
    return faiss


def index_bytes(index) -> int:
    return int(_faiss().serialize_index(index).nbytes)


def pq_shape(N: int, d: int, budget: int) -> tuple[int, int] | None:
    """Largest PQ code (m sub-vectors x nbits) whose codes + codebooks fit the budget."""
    best = None
    for nbits in (8, 6, 4):
        for m in (x for x in range(1, d + 1) if d % x == 0):
            size = N * math.ceil(m * nbits / 8) + d * (2 ** nbits) * 4 + 256
            if size <= budget and (best is None or m * nbits > best[0] * best[1]):
                best = (m, nbits)
    return best


def build_index(kind: str, K: np.ndarray, train: np.ndarray | None = None, budget: int | None = None):
    """A FAISS index over K (inner product on unit vectors), or (None, reason) if PQ cannot fit the budget."""
    faiss = _faiss()
    N, d = K.shape
    ip = faiss.METRIC_INNER_PRODUCT
    if kind == "flat":
        index, info = faiss.IndexFlatIP(d), ""
    elif kind == "hnsw":
        index, info = faiss.IndexHNSWFlat(d, 32, ip), "M=32"
        index.hnsw.efConstruction, index.hnsw.efSearch = 80, 64
    elif kind == "pq":
        shape = pq_shape(N, d, budget)
        if shape is None:
            return None, "does not fit"
        index, info = faiss.IndexPQ(d, shape[0], shape[1], ip), f"m={shape[0]} nbits={shape[1]}"
        index.train(np.ascontiguousarray(train, dtype=np.float32))
    else:
        raise ValueError(kind)
    index.add(np.ascontiguousarray(K, dtype=np.float32))
    return index, info


def fifo_search(K: np.ndarray, Q: np.ndarray, keep: int) -> np.ndarray:
    """Exact search over the last `keep` items, stored in fp16 (a recent-items cache)."""
    N = len(K)
    if keep == 0:
        return np.full((len(Q), 1), -1)
    kept = K[N - keep:].astype(np.float16).astype(np.float32)
    return rank_ids(Q, kept) + (N - keep)


# ---------------------------------------------------------------- one evaluation

def evaluate_all(K: np.ndarray, qsets: dict, dv: int, mean: np.ndarray | None, train: np.ndarray,
                 methods: tuple, ceilings: bool) -> list[dict]:
    """Build every store once over K (N, dk), then search each query set
    {name: (Q (nq, dk), target item indices (nq,))}. One row per method and set."""
    N, dk = K.shape
    C = codes(N, dv)
    rows = []

    def emit(method, nbytes, info, search):
        for q, (Q, target) in qsets.items():
            ids = search(Q)
            res = metrics(ids, target) if ids is not None else {"recall1": None, "recall10": None,
                                                                "mrr10": None, "hits1": None}
            rows.append({"method": method, "qset": q, "bytes": nbytes, "info": info, **res})

    for m in methods:
        st = MatrixStore(dk, dv, m.split("+")[0].rstrip("3"), passes=3 if m.startswith("delta3") else 1,
                         mean=mean if "+c" in m else None)
        st.write(K, C)
        emit(m, st.nbytes, "", lambda Q: rank_ids(st.read(Q), C))
    budget = dk * dv * 4
    kinds = [("pq", budget)] + ([("flat", None), ("hnsw", None)] if ceilings else [])
    for kind, b in kinds:
        index, info = build_index(kind, K, train, b)
        emit(kind, index_bytes(index) if index is not None else None, info,
             lambda Q: None if index is None else index.search(np.ascontiguousarray(Q, dtype=np.float32),
                                                                min(TOPK, N))[1])
    keep = min(N, budget // (2 * dk))
    emit("fifo", keep * dk * 2, f"keeps last {keep}", lambda Q: fifo_search(K, Q, keep))
    return rows


def _append(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------- A1 capacity

def noisy(K: np.ndarray, cos: float, rng) -> np.ndarray:
    """Unit queries at cosine `cos` to their keys."""
    if cos >= 1.0:
        return K.copy()
    noise = rng.standard_normal(K.shape).astype(np.float32)
    noise = unit(noise - (noise * K).sum(1, keepdims=True) * K)
    return unit(cos * K + math.sqrt(1 - cos ** 2) * noise)


def run_capacity(out: Path, seeds=range(5), dims=(64, 256, 1024), ratios=(0.125, 0.25, 0.5, 1, 2, 4, 8),
                 coss=(1.0, 0.9, 0.7), n_query: int = 1000) -> None:
    for d in dims:
        for seed in seeds:
            rng = np.random.default_rng([1, d, seed])
            train = unit(rng.standard_normal((10000, d)).astype(np.float32))   # PQ training sample
            for ratio in ratios:
                N = max(1, int(ratio * d))
                K = unit(rng.standard_normal((N, d)).astype(np.float32))
                target = rng.choice(N, size=min(N, n_query), replace=False)
                qsets = {cos: (noisy(K[target], cos, rng), target) for cos in coss}
                rows = evaluate_all(K, qsets, d, None, train, ("add", "delta", "delta3", "lstsq"), True)
                _append(out, [{"exp": "capacity", "d": d, "dv": d, "N": N, "ratio": ratio, "cos": r.pop("qset"),
                               "seed": seed, **{k: v for k, v in r.items() if k != "hits1"}} for r in rows])
            print(f"capacity d={d} seed={seed} done", flush=True)


# ---------------------------------------------------------------- A2 real data

def embed_quora(out: Path, n_calib: int = 20000, n_split: int = 20000, model: str = "all-MiniLM-L6-v2") -> None:
    """Dedupe Quora duplicate pairs, split them (calibration / dev / test) and
    save normalised MiniLM embeddings of keys (questions) and queries
    (their duplicates)."""
    from datasets import load_dataset  # noqa: PLC0415
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    ds = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train")
    seen, pairs = set(), []
    for a, p in zip(ds["anchor"], ds["positive"]):
        a, p = a.strip(), p.strip()
        if a.lower() == p.lower() or a in seen or p in seen:
            continue
        seen.update((a, p))
        pairs.append((a, p))
    rng = np.random.default_rng(0)
    pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    need = n_calib + 2 * n_split
    assert len(pairs) >= need, f"only {len(pairs)} distinct pairs"
    enc = SentenceTransformer(model)
    emb = lambda xs: enc.encode(xs, batch_size=256, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    calib = pairs[:n_calib]
    dev, test = pairs[n_calib:n_calib + n_split], pairs[n_calib + n_split:need]
    np.savez(out, calib=emb([a for a, _ in calib]),
             dev_key=emb([a for a, _ in dev]), dev_query=emb([p for _, p in dev]),
             test_key=emb([a for a, _ in test]), test_query=emb([p for _, p in test]),
             n_pairs=len(pairs), model=model)
    print(f"{len(pairs):,} distinct pairs; saved {out}")


def run_real(emb_path: Path, out: Path, split: str, seeds=range(5), Ns=(96, 192, 384, 768, 1536, 3072, 6144),
             dvs=BUDGET_DV, n_query: int = 1000) -> None:
    e = np.load(emb_path)
    keys, queries, calib = e[f"{split}_key"], e[f"{split}_query"], e["calib"]
    mean = calib.mean(0)
    for seed in seeds:
        for N in Ns:
            rng = np.random.default_rng([2, 0 if split == "dev" else 1, seed, N])
            items = rng.choice(len(keys), size=N, replace=False)
            target = rng.choice(N, size=min(N, n_query), replace=False)
            K, qsets = keys[items], {"paraphrase": (queries[items[target]], target)}
            for dv in dvs:
                rows = evaluate_all(K, qsets, dv, mean, calib, MATRIX, ceilings=dv == dvs[0])
                _append(out, [{"exp": "real", "split": split, "dv": dv, "budget": 384 * dv * 4, "N": N,
                               "seed": seed, **{k: v for k, v in r.items()
                                                if k != "hits1" or dv == CONFIRM_DV}} for r in rows])
        print(f"real {split} seed={seed} done", flush=True)


# ---------------------------------------------------------------- A3 updates and deletes

def run_updates(emb_path: Path, out: Path, split: str = "test", seeds=range(5), Ns=(384, 768),
                fracs=(0.25, 1.0), dv: int = CONFIRM_DV) -> None:
    e = np.load(emb_path)
    keys, queries, mean = e[f"{split}_key"], e[f"{split}_query"], e["calib"].mean(0)
    for seed in seeds:
        for N in Ns:
            rng = np.random.default_rng([3, seed, N])
            items = rng.choice(len(keys), size=N, replace=False)
            K, Q = keys[items], queries[items]
            C = codes(2 * N, dv)                            # ids N..2N-1 are the new values
            for frac in fracs:
                changed = rng.choice(N, size=int(frac * N), replace=False)
                same = np.setdiff1d(np.arange(N), changed)
                for m in STREAMING:
                    st = MatrixStore(384, dv, m.split("+")[0].rstrip("3"), 3 if m.startswith("delta3") else 1,
                                     mean if "+c" in m else None)
                    st.write(K, C[:N])
                    st.overwrite(K[changed], C[changed], C[N + changed])
                    ids = rank_ids(st.read(Q), C)
                    top = ids[:, 0]
                    row = {"exp": "updates", "op": "overwrite", "method": m, "N": N, "frac": frac, "seed": seed,
                           "fresh": float((top[changed] == N + changed).mean()),
                           "stale": float((top[changed] == changed).mean()),
                           "untouched_recall1": float((top[same] == same).mean()) if len(same) else None}
                    st2 = MatrixStore(384, dv, st.rule, st.passes, st.mean)
                    st2.write(K, C[:N])
                    st2.delete(K[changed], C[changed])
                    top2 = rank_ids(st2.read(Q), C[:N])[:, 0]
                    row |= {"deleted_still_found": float((top2[changed] == changed).mean()),
                            "after_delete_untouched_recall1": float((top2[same] == same).mean()) if len(same) else None}
                    _append(out, [row])
        print(f"updates seed={seed} done", flush=True)
    # float drift: write and delete the same random items many times, compare with the exact store
    rng = np.random.default_rng(4)
    K = unit(rng.standard_normal((1000, 384)).astype(np.float32))
    C = codes(1000, dv)
    for dtype in (np.float32, np.float64):
        st = MatrixStore(384, dv, "add", dtype=dtype)
        st.write(K, C)
        ref = st.S.copy()
        for cycle in range(100):
            j = rng.choice(1000, size=100, replace=False)
            st.delete(K[j], C[j])
            st.write(K[j], C[j])
        _append(out, [{"exp": "drift", "dtype": np.dtype(dtype).name, "cycles": 100, "ops": 2 * 100 * 100,
                       "rel_error": float(np.linalg.norm(st.S - ref) / np.linalg.norm(ref))}])


# ---------------------------------------------------------------- A5 cost

def run_cost(out: Path, Ns=(1000, 10000, 100000, 300000), d: int = 384, n_query: int = 1000) -> None:
    rng = np.random.default_rng(5)
    for N in Ns:
        K = unit(rng.standard_normal((N, d)).astype(np.float32))
        Q = noisy(K[:n_query], 0.9, rng)
        C = codes(N, d)
        st = MatrixStore(d, d, "add")
        t0 = time.time(); st.write(K, C); t_write = time.time() - t0
        t0 = time.time(); R = st.read(Q); t_read = time.time() - t0
        t0 = time.time(); rank_ids(R, C); t_decode = time.time() - t0
        rows = [{"exp": "cost", "method": "matrix-add", "N": N, "write_s": t_write, "read_s": t_read,
                 "decode_s": t_decode, "bytes": st.nbytes}]
        faiss = _faiss()
        for kind in ("flat", "hnsw", "pq"):
            if kind == "flat":
                index = faiss.IndexFlatIP(d)
            elif kind == "hnsw":
                index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
            else:
                index = faiss.IndexPQ(d, 48, 8, faiss.METRIC_INNER_PRODUCT)
                index.train(K[:min(N, 50000)])
            t0 = time.time(); index.add(K); t_write = time.time() - t0
            t0 = time.time(); index.search(Q, TOPK); t_read = time.time() - t0
            rows.append({"exp": "cost", "method": kind, "N": N, "write_s": t_write, "read_s": t_read,
                         "decode_s": 0.0, "bytes": index_bytes(index)})
        _append(out, rows)
        print(f"cost N={N} done", flush=True)


# ---------------------------------------------------------------- analysis

def load(path, exp=None) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return [r for r in rows if exp is None or r["exp"] == exp]


def _ci(xs) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    m, h = mean_ci(xs)
    return f"{100 * m:.1f}" + ("" if math.isnan(h) else f" ± {100 * h:.1f}")


def grid_table(rows: list[dict], key: str, methods: list[str], metric: str = "recall1") -> str:
    by = defaultdict(list)
    for r in rows:
        by[(r[key], r["method"])].append(r[metric])
    keys = sorted({r[key] for r in rows})
    lines = [f"| {key} | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    lines += [f"| {k} | " + " | ".join(_ci(by.get((k, m), [])) for m in methods) + " |" for k in keys]
    return "\n".join(lines)


def best_variant(dev_rows: list[dict], dv: int = CONFIRM_DV) -> str:
    """The streaming matrix variant with the best mean dev recall@1 at budget dv, over all N."""
    score = defaultdict(list)
    for r in dev_rows:
        if r["dv"] == dv and r["method"] in STREAMING:
            score[r["method"]].append(r["recall1"])
    return max(score, key=lambda m: np.mean(score[m]))


def bootstrap_hits(a: list[list[int]], b: list[list[int]], n: int = N_RESAMPLES, seed: int = 0) -> dict:
    """recall@1 difference b - a (points), paired over queries pooled across seeds."""
    a, b = np.concatenate([np.asarray(x) for x in a]), np.concatenate([np.asarray(x) for x in b])
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n, len(a)))
    diff = 100 * (b[idx].mean(1) - a[idx].mean(1))
    tail = min((diff <= 0).sum(), (diff >= 0).sum())
    obs = 100 * (b.mean() - a.mean())
    p = 1.0 if obs == 0 and np.all(diff == 0) else min(1.0, 2 * (tail + 1) / (n + 1))
    return {"diff": float(obs), "lo": float(np.percentile(diff, 2.5)), "hi": float(np.percentile(diff, 97.5)),
            "p": p, "queries": int(len(a))}


def confirmatory(test_rows: list[dict], variant: str) -> str:
    lines = [f"**Confirmatory family (PROTOCOL.md): {variant} − pq at {CONFIRM_DV * 384 * 4 / 1e3:.0f} KB, "
             "test split, recall@1, Holm-corrected:**", "",
             "| N | difference, points (95% CI) | p | Holm p | per-seed differences |", "|---|---|---|---|---|"]
    res = []
    for N in CONFIRM_N:
        rs = [r for r in test_rows if r["N"] == N and r["dv"] == CONFIRM_DV]
        by = {(r["method"], r["seed"]): r for r in rs}
        seeds = sorted({s for (m, s) in by if m == variant} & {s for (m, s) in by if m == "pq"
                                                                and by[(m, s)]["hits1"] is not None})
        if not seeds:
            continue
        boot = bootstrap_hits([by[("pq", s)]["hits1"] for s in seeds], [by[(variant, s)]["hits1"] for s in seeds])
        per = ", ".join(f"{100 * (by[(variant, s)]['recall1'] - by[('pq', s)]['recall1']):+.1f}" for s in seeds)
        res.append((N, boot, per))
    adj = holm([b["p"] for _, b, _ in res])
    lines += [f"| {N} | {b['diff']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}] | {b['p']:.4f} | {h:.4f} | {per} |"
              for (N, b, per), h in zip(res, adj)]
    return "\n".join(lines)


def summary(path) -> str:
    out = []
    cap = load(path, "capacity")
    if cap:
        out.append("## A1 capacity (random keys): recall@1 %, mean ± 95% CI over seeds\n")
        for d in sorted({r["d"] for r in cap}):
            for cos in sorted({r["cos"] for r in cap}, reverse=True):
                rs = [r for r in cap if r["d"] == d and r["cos"] == cos]
                out += [f"**d = {d}, query cosine {cos}** (budget {d * d * 4 / 1e3:.0f} KB for matrix, pq, fifo)\n",
                        grid_table(rs, "N", ["add", "delta", "delta3", "lstsq", "pq", "fifo", "flat", "hnsw"]), ""]
    real = load(path, "real")
    if real:
        dev, test = [r for r in real if r["split"] == "dev"], [r for r in real if r["split"] == "test"]
        out.append("## A2 real data (Quora paraphrases, MiniLM-L6): recall@1 %, mean ± 95% CI over seeds\n")
        for split, rs in (("dev", dev), ("test", test)):
            for dv in sorted({r["dv"] for r in rs}):
                sub = [r for r in rs if r["dv"] == dv]
                pq_info = sorted({f"N={r['N']}: {r['info']}" for r in sub if r["method"] == "pq" and r["info"]})
                out += [f"**{split}, budget {384 * dv * 4 / 1e3:.0f} KB**\n",
                        grid_table(sub, "N", list(MATRIX) + ["pq", "fifo"] + (["flat", "hnsw"] if dv == BUDGET_DV[0] else [])),
                        "", "PQ shapes: " + "; ".join(pq_info), ""]
        if dev and test:
            v = best_variant(dev)
            out += [f"Best streaming matrix variant on dev (chosen before looking at test): **{v}**\n",
                    confirmatory(test, v), ""]
    up = load(path, "updates")
    if up:
        out.append("## A3 updates and deletes (test split, 590 KB): %, mean ± 95% CI over seeds\n")
        cols = ["fresh", "stale", "untouched_recall1", "deleted_still_found", "after_delete_untouched_recall1"]
        out += ["| method | N | fraction changed | " + " | ".join(cols) + " |", "|---|---|---|" + "---|" * len(cols)]
        by = defaultdict(list)
        for r in up:
            by[(r["method"], r["N"], r["frac"])].append(r)
        for (m, N, f), rs in sorted(by.items()):
            out.append(f"| {m} | {N} | {f} | " + " | ".join(_ci([r[c] for r in rs]) for c in cols) + " |")
        out.append("")
    for r in load(path, "drift"):
        out.append(f"Float drift ({r['dtype']}, {r['ops']:,} deletes and re-writes): relative error {r['rel_error']:.2e}")
    cost = load(path, "cost")
    if cost:
        out += ["", "## A5 cost (CPU, d = 384, 1,000 queries)\n",
                "| method | N | write s | read s | decode s | MB |", "|---|---|---|---|---|---|"]
        out += [f"| {r['method']} | {r['N']:,} | {r['write_s']:.3f} | {r['read_s']:.3f} | {r['decode_s']:.3f} | "
                f"{r['bytes'] / 1e6:.2f} |" for r in cost]
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", choices=["embed", "capacity", "real", "updates", "cost", "summary"])
    ap.add_argument("--out", default="results/store.jsonl")
    ap.add_argument("--emb", default="data/quora_emb.npz")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args(argv)
    out, seeds = Path(a.out), range(a.seeds)
    if a.exp == "embed":
        Path(a.emb).parent.mkdir(parents=True, exist_ok=True)
        embed_quora(Path(a.emb))
    elif a.exp == "capacity":
        run_capacity(out, seeds)
    elif a.exp == "real":
        run_real(Path(a.emb), out, a.split, seeds)
    elif a.exp == "updates":
        run_updates(Path(a.emb), out, seeds=seeds)
    elif a.exp == "cost":
        run_cost(out)
    else:
        print(summary(out))


if __name__ == "__main__":
    main()
