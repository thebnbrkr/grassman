"""The fast PairMixer must compute exactly what the paper's order computes."""
import itertools

import pytest
import torch

from grassman.bench import fast_vs_reference, set_reference
from grassman.mixers import PAIR_KINDS, PairMixer, pair_features, pair_norm, sin_angle
from grassman.model import LM, LONG, Config


@pytest.mark.parametrize("kind", PAIR_KINDS)
def test_closed_form_norm_matches_the_features(kind):
    torch.manual_seed(0)
    a, b = torch.randn(4, 7, 9, dtype=torch.float64), torch.randn(4, 7, 9, dtype=torch.float64)
    torch.testing.assert_close(pair_norm(a, b, kind), pair_features(a, b, kind).norm(dim=-1))


def test_sin_angle():
    a = torch.tensor([1.0, 0.0, 0.0])
    assert sin_angle(a, torch.tensor([0.0, 3.0, 0.0])).item() == pytest.approx(1.0)
    assert sin_angle(a, torch.tensor([2.0, 0.0, 0.0])).item() == pytest.approx(0.0, abs=1e-6)
    assert sin_angle(a, torch.tensor([1.0, 1.0, 0.0])).item() == pytest.approx(2 ** -0.5, rel=1e-5)


@pytest.mark.parametrize("kind,normalize,windows,T", itertools.product(
    PAIR_KINDS, (True, False), ((1, 2, 4, 8, 12, 16), LONG), (5, 40, 128)))
def test_fast_path_equals_paper_order(kind, normalize, windows, T):
    torch.manual_seed(0)
    m = PairMixer(32, kind=kind, r=8, windows=windows, normalize=normalize, dropout=0.0).double()
    h = torch.randn(2, T, 32, dtype=torch.float64)
    outs, grads = [], []
    for ref in (False, True):
        m.zero_grad()
        m.use_reference = ref
        y = m(h)
        y.square().mean().backward()
        outs.append(y.detach())
        grads.append({n: p.grad.clone() for n, p in m.named_parameters()})
    torch.testing.assert_close(outs[0], outs[1])
    for n in grads[0]:
        torch.testing.assert_close(grads[0][n], grads[1][n], msg=n)


def test_whole_model_fast_equals_reference():
    torch.manual_seed(0)
    model = LM(Config(mixer="grassmann", vocab_size=101, block_size=32, d=32, d_ff=64, n_layers=2,
                      dropout=0.0)).double().eval()
    idx = torch.randint(0, 101, (2, 32))
    fast = model(idx)[0]
    set_reference(model, True)
    torch.testing.assert_close(fast, model(idx)[0])


def test_benchmark_helper_reports_tiny_differences():
    rows = fast_vs_reference(torch.device("cpu"), d=32, T=20, batch=2)
    assert len(rows) == 2 * len(PAIR_KINDS)
    assert all(r["max_out_diff"] < 1e-4 and r["max_grad_diff"] < 1e-4 for r in rows), rows
