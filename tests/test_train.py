import json
import math

import numpy as np
import pytest
import torch

from grassman import analyze, data, train


def test_blocks_shift_targets_by_one():
    stream = torch.arange(23)
    x, y = data.blocks(stream, 5)
    assert x.shape == y.shape == (4, 5)
    torch.testing.assert_close(y, x + 1)
    assert x[0, 0] == 0 and x[-1, -1] == 19


def test_article_boundaries_follow_level_one_headings():
    lines = [" = Valkyria Chronicles III = \n", " Senjō no Valkyria 3 ... \n",
             " = = Gameplay = = \n", " As with previous games ... \n",
             " = Tower Building of the Little Rock Arsenal = \n", " The building ... \n",
             " = = = Early history = = = \n"]
    assert data.article_ids(lines) == [0, 0, 0, 0, 1, 1, 1]
    assert not data.is_article_start(" = = Gameplay = = \n")


def test_lr_schedule_warms_up_then_decays_to_floor():
    lrs = [train.lr_at(s, 1000, 1e-3, 100) for s in range(1000)]
    assert lrs[0] == pytest.approx(1e-5)
    assert lrs[99] == pytest.approx(1e-3)
    assert all(a >= b for a, b in zip(lrs[100:], lrs[101:]))
    assert lrs[-1] == pytest.approx(1e-4, rel=1e-3)


def test_evaluate_uniform_model_gives_vocab_perplexity():
    class Uniform(torch.nn.Module):
        def forward(self, x, y):
            logits = torch.zeros(*x.shape, 50)
            return logits, torch.nn.functional.cross_entropy(logits.view(-1, 50), y.reshape(-1))

    x = torch.randint(0, 50, (10, 8))
    nll = train.evaluate(Uniform(), x, x, 3, torch.device("cpu"), None)
    assert math.exp(nll) == pytest.approx(50, rel=1e-5)


def test_mean_ci_known_values():
    m, h = analyze.mean_ci([1.0, 2.0, 3.0])
    assert m == 2.0
    assert h == pytest.approx(4.303 * 1.0 / math.sqrt(3), rel=1e-4)
    assert math.isnan(analyze.mean_ci([5.0])[1])


@pytest.fixture
def tiny_data(tmp_path):
    g = torch.Generator().manual_seed(0)
    for split, n in (("train", 4000), ("validation", 600), ("test", 600)):
        torch.save(torch.randint(0, 200, (n,), generator=g), tmp_path / f"{split}.pt")
        torch.save(torch.arange(n) // 100, tmp_path / f"{split}_doc.pt")   # 100-token "articles"
    return tmp_path


def _run(tiny_data, tmp_path, mixer, seed=0, lr=1e-3, extra=()):
    return train.main(["--mixer", mixer, "--seed", str(seed), "--lr", str(lr), "--epochs", "2",
                       "--block-size", "16", "--d", "32", "--n-layers", "2", "--batch-size", "8",
                       "--warmup", "5", "--tag", "t", "--data-dir", str(tiny_data),
                       "--out", str(tmp_path / "runs.jsonl"), "--device", "cpu", *extra])


def test_end_to_end_run_writes_a_complete_record(tiny_data, tmp_path):
    r = _run(tiny_data, tmp_path, "grassmann")
    line = json.loads((tmp_path / "runs.jsonl").read_text())
    assert line["config"]["mixer"] == "grassmann" and line["seed"] == 0
    assert len(line["history"]) == 2 and line["best_epoch"] in (1, 2)
    assert math.isfinite(line["test_ppl"]) and line["tokens_per_s"] > 0
    assert r["best_val_ppl"] == min(h["val_ppl"] for h in line["history"])
    d = line["test_detail"]
    assert len(d["pos_nll"]) == 16 and len(d["doc_nll"]) == 6
    # the per-article sums and the per-position means describe the same tokens
    assert math.exp(sum(d["doc_nll"]) / sum(d["doc_tokens"])) == pytest.approx(line["test_ppl"], rel=1e-6)
    assert sum(b["tokens"] for b in d["buckets"].values()) == sum(d["doc_tokens"])


def test_no_test_flag_never_scores_the_test_set(tiny_data, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("test set evaluated")
    monkeypatch.setattr(train, "evaluate_detail", boom)
    r = _run(tiny_data, tmp_path, "grassmann", extra=["--no-test"])
    assert r["test_ppl"] is None and r["test_detail"] is None
    assert all("test_ppl" not in h for h in r["history"])


def test_evaluate_detail_matches_evaluate(tiny_data):
    from grassman.model import LM, Config
    torch.manual_seed(0)
    model = LM(Config(vocab_size=30522, block_size=16, d=32, d_ff=128, n_layers=2))
    stream = torch.load(tiny_data / "test.pt")
    x, y = data.blocks(stream, 16)
    doc = data.blocks(torch.load(tiny_data / "test_doc.pt"), 16)[1]
    rank = train.frequency_rank(torch.load(tiny_data / "train.pt"), 30522)
    cpu = torch.device("cpu")
    detail = train.evaluate_detail(model, x, y, doc, rank, 7, cpu, None)
    assert detail["nll"] == pytest.approx(train.evaluate(model, x, y, 7, cpu, None), rel=1e-5)


def test_frequency_rank():
    rank = train.frequency_rank(torch.tensor([5, 5, 5, 2, 2, 9]), 10)
    assert rank[5] == 0 and rank[2] == 1 and rank[9] == 2
    assert sorted(rank.tolist()) == list(range(10))


def test_same_seed_reproduces_exactly(tiny_data, tmp_path):
    a = _run(tiny_data, tmp_path / "a", "grassmann", seed=1)
    b = _run(tiny_data, tmp_path / "b", "grassmann", seed=1)
    assert a["test_ppl"] == pytest.approx(b["test_ppl"], rel=1e-6)


def test_skip_existing(tiny_data, tmp_path):
    _run(tiny_data, tmp_path, "sym")
    assert _run(tiny_data, tmp_path, "sym", extra=["--skip-existing"]) is None
    assert _run(tiny_data, tmp_path, "sym", seed=1, extra=["--skip-existing"]) is not None


def test_analyze_summary_and_best_lr(tiny_data, tmp_path):
    for mixer in ("grassmann", "attention"):
        for seed in (0, 1):
            _run(tiny_data, tmp_path, mixer, seed=seed)
    _run(tiny_data, tmp_path, "attention", seed=0, lr=3e-3)
    rows = analyze.load(tmp_path / "runs.jsonl", "t")
    table = analyze.summary([r for r in rows if r["lr"] == 1e-3])
    assert "| grassmann | 2 |" in table and "| attention | 2 |" in table
    assert "Confirmatory" in table and "pos 0–15" in table
    best = analyze.best_lr(rows)
    assert set(best) == {"grassmann", "attention"} and best["grassmann"] == 1e-3


def test_holm_known_values():
    # p = (0.01, 0.04, 0.03): sorted 0.01*3, 0.03*2, 0.04*1 -> monotone 0.03, 0.06, 0.06
    assert analyze.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert analyze.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def _fake_run(seed, doc_nll, doc_tokens):
    return {"seed": seed, "test_detail": {"doc_nll": doc_nll, "doc_tokens": doc_tokens}}


def test_bootstrap_identical_systems_give_zero_and_p_one():
    rng = np.random.default_rng(0)
    tok = rng.integers(100, 1000, 50).tolist()
    nll = (np.array(tok) * rng.uniform(4, 6, 50)).tolist()
    a = [_fake_run(0, nll, tok)]
    b = analyze.paired_bootstrap(a, a, n=2000)
    assert b["diff"] == 0 and b["lo"] == 0 and b["hi"] == 0 and b["p"] == 1.0


def test_bootstrap_detects_a_clear_difference_and_its_sign():
    rng = np.random.default_rng(0)
    tok = np.array(rng.integers(100, 1000, 50))
    per_tok = rng.uniform(4, 6, 50)
    a = [_fake_run(0, (tok * per_tok).tolist(), tok.tolist())]
    worse = [_fake_run(0, (tok * (per_tok + 0.1)).tolist(), tok.tolist())]
    r = analyze.paired_bootstrap(a, worse, n=2000)
    assert r["diff"] > 0 and r["lo"] > 0 and r["p"] < 0.001
    r2 = analyze.paired_bootstrap(worse, a, n=2000)
    assert r2["diff"] < 0 and r2["hi"] < 0


def test_bootstrap_only_pairs_shared_seeds():
    tok = [100] * 10
    a = [_fake_run(0, [500.0] * 10, tok), _fake_run(1, [520.0] * 10, tok)]
    b = [_fake_run(1, [520.0] * 10, tok)]
    assert analyze.paired_bootstrap(a, b, n=500)["seeds"] == [1]


def test_pick_amp(monkeypatch):
    cuda = torch.device("cuda")
    assert train.pick_amp(torch.device("cpu")) is None
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (7, 5))   # T4
    assert train.pick_amp(cuda) is torch.float16
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (8, 0))   # A100
    assert train.pick_amp(cuda) is torch.bfloat16
    assert train.pick_amp(cuda, "fp16") is torch.float16 and train.pick_amp(cuda, "off") is None
