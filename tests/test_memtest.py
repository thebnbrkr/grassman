"""Tests for the memory experiments (memtest/)."""
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from memtest import analyze, layers, store, tasks
from memtest import train as mtrain
from memtest.model import MODELS, Config, RecallLM

torch.manual_seed(0)


# ---------------------------------------------------------------- exact maths

def test_fft_conv_matches_direct():
    u, h = torch.randn(2, 3, 40), torch.randn(3, 40)
    direct = torch.zeros_like(u)
    for t in range(40):
        for s in range(t + 1):
            direct[..., t] += h[:, s] * u[..., t - s]
    assert torch.allclose(layers.fft_causal_conv(u, h), direct, atol=1e-4)


@pytest.mark.parametrize("chunk", [1, 8, 16, 48])
def test_delta_rule_chunked_equals_recurrence(chunk):
    B, h, T, dk, dv = 2, 3, 48, 8, 5
    q = F.normalize(torch.randn(B, h, T, dk, dtype=torch.float64), dim=-1)
    k = F.normalize(torch.randn(B, h, T, dk, dtype=torch.float64), dim=-1)
    v = torch.randn(B, h, T, dv, dtype=torch.float64)
    beta = torch.rand(B, h, T, dtype=torch.float64)
    ref = layers.delta_rule_reference(q, k, v, beta)
    assert torch.allclose(layers.delta_rule(q, k, v, beta, chunk), ref, atol=1e-10)


@pytest.mark.parametrize("chunk", [1, 8, 32])
def test_linear_attention_chunked_equals_quadratic(chunk):
    B, h, T, dk, dv = 2, 2, 32, 6, 4
    q = F.elu(torch.randn(B, h, T, dk, dtype=torch.float64)) + 1
    k = F.elu(torch.randn(B, h, T, dk, dtype=torch.float64)) + 1
    v = torch.randn(B, h, T, dv, dtype=torch.float64)
    a = torch.tril(q @ k.transpose(-1, -2))
    ref = (a @ v) / a.sum(-1, keepdim=True)
    assert torch.allclose(layers.linear_attention(q, k, v, chunk, eps=0.0), ref, atol=1e-10)


# ---------------------------------------------------------------- causality

@pytest.mark.parametrize("kind", ["H", "A", "W", "L", "D", "M"])
def test_mixers_are_causal(kind):
    cfg = Config(max_len=64, d=16, n_heads=2, window=8, chunk=16, slots=2)
    from memtest.model import make_mixer
    m = make_mixer(kind, cfg).eval()
    x = torch.randn(2, 64, 16)
    y = m(x)
    for t in (5, 17, 40):
        x2 = x.clone()
        x2[:, t:] = torch.randn_like(x2[:, t:])
        assert torch.allclose(m(x2)[:, :t], y[:, :t], atol=1e-5), f"{kind} leaks the future at {t}"


def test_moment_memory_only_sees_finished_chunks():
    m = layers.MomentMemory(16, n_heads=2, chunk=8, slots=2).eval()
    x = torch.randn(1, 32, 16)
    y = m(x)
    x2 = x.clone()
    x2[:, 8:15] = torch.randn(1, 7, 16)            # change chunk 1, except its last token
    y2 = m(x2)
    assert torch.allclose(y2[:, :8], y[:, :8], atol=1e-6)     # chunk 0 cannot see chunk 1
    assert torch.allclose(y2[:, 15], y[:, 15], atol=1e-6)     # nor can chunk 1 itself (unfinished)
    assert not torch.allclose(y2[:, 16:], y[:, 16:])          # chunks 2-3 can


def test_models_build_and_count_state():
    for name in MODELS:
        cfg = Config(model=name, max_len=128, d=32)
        m = RecallLM(cfg)
        x, y, _ = tasks.make("far", 2, 128, 4, seed=0)
        logits, loss = m(x, y)
        assert logits.shape == (8, tasks.VOCAB) and torch.isfinite(loss)
        assert m.state_numbers(128) > 0
    att, hyb = RecallLM(Config(model="attention", max_len=128, d=32)), RecallLM(Config(model="hyena+moment", max_len=128, d=32))
    assert att.state_numbers(128) == 4 * 2 * 32 * 128
    assert hyb.state_numbers(4096) < att.state_numbers(4096)


# ---------------------------------------------------------------- tasks

@pytest.mark.parametrize("task", ["mqar", "far"])
def test_task_labels_point_at_the_right_value(task):
    T, pairs = 256, 16
    x, y, dist = tasks.make(task, 50, T, pairs, seed=1)
    assert ((x > 0) & (x < tasks.VOCAB)).all()
    for i in range(50):
        q = (y[i] != -100).nonzero().flatten()
        assert len(q) == pairs
        for p in q.tolist():
            kpos = p - int(dist[i, p])
            assert x[i, kpos] == x[i, p]                       # same key
            assert x[i, kpos + 1] == y[i, p]                   # its value follows it
            assert tasks.KEYS[0] <= x[i, p] < tasks.KEYS[1]
            assert tasks.VALUES[0] <= y[i, p] < tasks.VALUES[1]
        keys = x[i, q]
        assert len(set(keys.tolist())) == pairs                # distinct keys
    if task == "far":
        assert (y[:, T - pairs:] != -100).all()


def test_tasks_are_reproducible():
    a, b = tasks.make("far", 4, 128, 4, seed=7), tasks.make("far", 4, 128, 4, seed=7)
    assert all(torch.equal(u, v) for u, v in zip(a, b))


# ---------------------------------------------------------------- training and analysis

def test_train_writes_record_and_skips(tmp_path):
    out = tmp_path / "r.jsonl"
    args = ["--model", "hyena+moment", "--task", "far", "--T", "64", "--pairs", "4", "--d", "32", "--chunk", "16",
            "--steps", "3", "--batch", "4", "--eval-n", "8", "--eval-every", "2", "--out", str(out),
            "--device", "cpu", "--skip-existing"]
    rec = mtrain.main(args)
    assert rec["test_acc"] is not None and len(rec["test_detail"]["seq_bucket_correct"]) == 8
    assert sum(rec["test_detail"]["bucket_total"]) == 8 * 4
    assert mtrain.main(args) is None
    assert len(out.read_text().splitlines()) == 1


def _fake(model, seed, correct, lr=1e-3):
    n, nb = 10, len(tasks.BUCKETS) - 1
    c = np.zeros((n, nb), int)
    t = np.zeros((n, nb), int)
    t[:, 4], c[:, 4] = 4, correct
    return {"tag": "main", "config": {"model": model, "pattern": MODELS[model], "d": 32, "slots": 4, "chunk": 64},
            "task": "far", "T": 4096, "pairs": 4, "seed": seed, "lr": lr, "steps": 1, "params": 1, "state_numbers": 10,
            "tokens_per_s": 1.0, "dev_acc": correct / 4,
            "test_detail": {"seq_bucket_correct": c.tolist(), "seq_bucket_total": t.tolist(),
                            "bucket_correct": c.sum(0).tolist(), "bucket_total": t.sum(0).tolist()}}


def test_analyze_tables_and_bootstrap(tmp_path):
    rows = [_fake(m, s, k) for s in range(3) for m, k in
            [("hyena", 1), ("hyena+attn", 4), ("hyena+moment", 3), ("hyena+delta", 2), ("attention", 4)]]
    b = analyze.paired_bootstrap([r for r in rows if r["config"]["model"] == "hyena"],
                                 [r for r in rows if r["config"]["model"] == "hyena+attn"], 1024)
    assert b["diff"] == pytest.approx(75.0) and b["p"] < 0.01
    same = [r for r in rows if r["config"]["model"] == "attention"]
    assert analyze.paired_bootstrap(same, same, 1024)["p"] == 1.0
    text = analyze.summary(rows, 1024, confirmatory=True)
    assert "Confirmatory" in text and "hyena+moment − hyena" in text
    assert analyze.best_lr([_fake("hyena", 0, 1, 1e-3), _fake("hyena", 0, 3, 3e-3)]) == {"hyena": 3e-3}
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    analyze.main([str(p), "--tag", "main", "--confirmatory"])


# ---------------------------------------------------------------- the matrix store

def test_matrix_add_is_exact_for_orthonormal_keys():
    d = 32
    K = np.linalg.qr(np.random.default_rng(0).standard_normal((d, d)))[0].astype(np.float32)
    C = store.codes(d, d)
    st = store.MatrixStore(d, d, "add")
    st.write(K, C)
    ids = store.rank_ids(st.read(K), C)
    assert (ids[:, 0] == np.arange(d)).all()


def test_matrix_delete_and_overwrite():
    rng = np.random.default_rng(1)
    d, N = 64, 40
    K = store.unit(rng.standard_normal((N, d)).astype(np.float32))
    C = store.codes(2 * N, d)
    st = store.MatrixStore(d, d, "add", dtype=np.float64)
    st.write(K, C[:N])
    before = st.S.copy()
    st.delete(K[:5], C[:5])
    st.write(K[:5], C[:5])
    assert np.allclose(st.S, before)                              # subtraction is exact
    for rule in ("add", "delta"):
        st = store.MatrixStore(d, d, rule, dtype=np.float64)
        st.write(K, C[:N])
        st.overwrite(K[:5], C[:5], C[N:N + 5])
        top = store.rank_ids(st.read(K[:5]), C)[:, 0]
        assert (top == np.arange(N, N + 5)).all()


def test_lstsq_and_delta_beat_add_on_correlated_keys():
    rng = np.random.default_rng(2)
    d, N = 64, 48
    K = store.unit(rng.standard_normal((N, d)) + 2.0).astype(np.float32)   # strongly correlated keys
    C = store.codes(N, d)
    recall = {}
    for rule, passes in (("add", 1), ("delta", 3), ("lstsq", 1)):
        st = store.MatrixStore(d, d, rule, passes)
        st.write(K, C)
        recall[rule] = (store.rank_ids(st.read(K), C)[:, 0] == np.arange(N)).mean()
    assert recall["lstsq"] == 1.0 and recall["delta"] > recall["add"]


def test_rank_metrics_and_fifo():
    ids = np.array([[3, 1, 2], [0, 2, 1], [5, 6, 7]])
    m = store.metrics(ids, np.array([3, 2, 1]))
    assert m["recall1"] == pytest.approx(1 / 3) and m["recall10"] == pytest.approx(2 / 3)
    assert m["mrr10"] == pytest.approx((1 + 0.5) / 3)
    K = store.unit(np.random.default_rng(3).standard_normal((10, 8)).astype(np.float32))
    assert (store.fifo_search(K, K, 4)[6:, 0] == np.arange(6, 10)).all()
    assert (store.fifo_search(K, K, 0) == -1).all()


def test_pq_shape_fits_budget():
    m, nbits = store.pq_shape(1000, 384, 384 * 384 * 4)
    assert 384 % m == 0 and 1000 * -(-m * nbits // 8) + 384 * 2 ** nbits * 4 <= 384 * 384 * 4
    assert store.pq_shape(10**6, 384, 1000) is None


def test_store_end_to_end_with_faiss(tmp_path):
    pytest.importorskip("faiss")
    rng = np.random.default_rng(4)
    K = store.unit(rng.standard_normal((200, 64)).astype(np.float32))
    train = store.unit(rng.standard_normal((10000, 64)).astype(np.float32))
    rows = store.evaluate_all(K, {"x": (K[:50], np.arange(50))}, 64, None, train, ("add", "delta"), True)
    got = {r["method"]: r for r in rows}
    assert got["flat"]["recall1"] == 1.0 and set(got) == {"add", "delta", "pq", "flat", "hnsw", "fifo"}
