"""Edge selection (choose vs reach), hybrids, checkpoints and readouts."""
import math

import pytest
import torch

from grassman import data, readout, train
from grassman.mixers import AttentionMixer, PairMixer
from grassman.model import LM, LONG, MIXERS, PAIR_MIXERS, Config

D = 32
TINY = dict(vocab_size=101, block_size=32, d=32, d_ff=128, n_layers=2, n_heads=4)


def test_select_weights_are_a_distribution_over_valid_edges():
    torch.manual_seed(0)
    m = PairMixer(D, r=8, windows=LONG, select=True, angle=True, dropout=0.0).eval()
    m.record = True
    m(torch.randn(2, 128, D))
    w, deltas = m.last["weights"], m.last["deltas"]
    assert w.shape == (2, 128, len(deltas))
    assert w[:, 0].abs().max() == 0                      # t = 0 has no earlier token
    torch.testing.assert_close(w[:, 1:].sum(-1), torch.ones(2, 127))
    for e, delta in enumerate(deltas):
        assert w[:, :delta, e].abs().max() == 0          # no edge before it exists
    assert m.last["sin"].shape == w.shape and (m.last["sin"] >= 0).all() and (m.last["sin"] <= 1 + 1e-5).all()


def test_dropping_an_edge_removes_it():
    torch.manual_seed(0)
    m = PairMixer(D, r=8, windows=LONG, select=True, dropout=0.0).eval()
    m.record, m.drop_edge = True, 1          # remove the t-1 edge
    m(torch.randn(1, 64, D))
    w = m.last["weights"]
    assert w[..., 0].abs().max() == 0
    torch.testing.assert_close(w[:, 2:].sum(-1), torch.ones(1, 62))


def test_select_starts_as_the_equal_average():
    """Zero-initialised edge bias and score output: before training, learned
    selection equals the paper's equal-weight average."""
    torch.manual_seed(0)
    m = PairMixer(D, r=8, select=True, dropout=0.0).eval()
    torch.nn.init.zeros_(m.score_out.weight)
    plain = PairMixer(D, r=8, select=False, dropout=0.0).eval()
    plain.load_state_dict({k: v for k, v in m.state_dict().items() if k in plain.state_dict()})
    h = torch.randn(2, 40, D)
    torch.testing.assert_close(m(h), plain(h))


@pytest.mark.parametrize("mixer", ["g-dilated", "g-both", "g-both-angle"])
def test_dilated_reach_is_127_and_causal(mixer):
    torch.manual_seed(0)
    m = LM(Config(mixer=mixer, **{**TINY, "block_size": 160})).blocks[0].mixer.eval()
    assert m.reach == 127
    h = torch.randn(1, 160, D)
    h2 = h.clone()
    h2[:, 10] += 1.0
    changed = (m(h) - m(h2)).abs().amax(dim=(0, 2)) > 1e-6
    assert not changed[:10].any() and changed[10 + 127] and not changed[10 + 128:].any()


def test_hybrid_puts_attention_where_asked():
    model = LM(Config(mixer="concat32", attn_layers=(1,), **TINY))
    assert isinstance(model.blocks[0].mixer, PairMixer)
    assert isinstance(model.blocks[1].mixer, AttentionMixer) and model.blocks[1].mixer.window is None


def test_new_variants_stay_size_matched():
    ref = LM(Config(mixer="attention")).n_params()
    for m in ("g-select", "g-dilated", "g-both", "g-both-angle"):
        assert abs(LM(Config(mixer=m)).n_params() / ref - 1) < 0.03, m


@pytest.fixture
def tiny_data(tmp_path):
    g = torch.Generator().manual_seed(0)
    for split, n in (("train", 4000), ("validation", 600), ("test", 600)):
        torch.save(torch.randint(0, 200, (n,), generator=g), tmp_path / f"{split}.pt")
        torch.save(torch.arange(n) // 100, tmp_path / f"{split}_doc.pt")
    return tmp_path


def _run(tiny_data, out_dir, mixer, extra=()):
    return train.main(["--mixer", mixer, "--epochs", "2", "--block-size", "32", "--d", "32", "--n-layers", "2",
                       "--batch-size", "8", "--warmup", "5", "--tag", "t", "--data-dir", str(tiny_data),
                       "--out", str(out_dir / "runs.jsonl"), "--device", "cpu", "--no-test", *extra])


def test_hybrid_runs_are_kept_apart_by_skip_existing(tiny_data, tmp_path):
    _run(tiny_data, tmp_path, "concat32")
    assert _run(tiny_data, tmp_path, "concat32", ["--attn-layers", "1", "--skip-existing"]) is not None
    assert _run(tiny_data, tmp_path, "concat32", ["--attn-layers", "1", "--skip-existing"]) is None
    from grassman import analyze
    names = {analyze.name(r) for r in analyze.load(tmp_path / "runs.jsonl")}
    assert names == {"concat32", "concat32+attn@1"}


def test_checkpoint_and_readouts(tiny_data, tmp_path):
    ckpt = tmp_path / "m.pt"
    r = _run(tiny_data, tmp_path, "g-both-angle", ["--save", str(ckpt)])
    model = readout.load(str(ckpt), torch.device("cpu"))
    x, y = data.blocks(torch.load(tiny_data / "validation.pt", weights_only=True), 32)
    # the saved weights reproduce the best validation perplexity
    assert math.exp(readout.mean_nll(model, x, y)) == pytest.approx(r["best_val_ppl"], rel=1e-4)
    usage = readout.edge_usage(model, x)
    assert set(usage) == {0, 1}
    for u in usage.values():
        assert sum(u["share"]) == pytest.approx(1.0, abs=1e-4) and 0 < u["top_weight"] <= 1
    abl = readout.edge_ablation(model, x[:4], y[:4])
    assert list(abl["delta_nll"][0]) == [dl for dl in LONG if dl < 32]
    sb = readout.sin_by_boundary(model, x, boundary_ids=[5, 7])
    assert all(0 <= v["elsewhere"] <= 1 for v in sb.values())


def test_val_table():
    from grassman import analyze
    rows = [{"seed": s, "best_val_ppl": v, "config": {"mixer": m, "normalize": True, "attn_layers": a}}
            for m, a, vals in (("grassmann", [], (270.0, 269.0)), ("concat32", [5], (210.0, 212.0)))
            for s, v in zip((0, 1), vals)]
    t = analyze.val_table(rows)
    assert "| grassmann | 270.0 | 269.0 | 269.5 | 1.0 | — |" in t
    assert "| concat32+attn@5 | 210.0 | 212.0 | 211.0 | 2.0 | -22.2%, -21.2% |" in t
