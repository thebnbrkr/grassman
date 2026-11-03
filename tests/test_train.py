import json
import math

import pytest
import torch

from grassman import analyze, data, train


def test_blocks_shift_targets_by_one():
    stream = torch.arange(23)
    x, y = data.blocks(stream, 5)
    assert x.shape == y.shape == (4, 5)
    torch.testing.assert_close(y, x + 1)
    assert x[0, 0] == 0 and x[-1, -1] == 19


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
    assert "| grassmann | 2 |" in table and "| attention | 2 |" in table and "Paired" in table
    best = analyze.best_lr(rows)
    assert set(best) == {"grassmann", "attention"} and best["grassmann"] == 1e-3
