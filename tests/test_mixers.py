import itertools

import pytest
import torch

from grassman.mixers import (PAIR_KINDS, PAIR_R, AttentionMixer, HyenaShortMixer, PairMixer,
                             causal_depthwise_conv, pair_dim, pair_features)

D = 32


def all_mixers():
    for kind in PAIR_KINDS:
        yield f"pair-{kind}", PairMixer(D, kind=kind, r=8)
    yield "pair-antisym-raw", PairMixer(D, kind="antisym", r=8, normalize=False)
    yield "hyena", HyenaShortMixer(D)
    yield "swa", AttentionMixer(D, n_heads=4, window=16)
    yield "attention", AttentionMixer(D, n_heads=4)


MIXERS = list(all_mixers())
IDS = [n for n, _ in MIXERS]


# ---------- Plucker / pair-feature maths ----------

def test_antisym_matches_definition():
    torch.manual_seed(0)
    a, b = torch.randn(5), torch.randn(5)
    expected = torch.tensor([a[i] * b[j] - a[j] * b[i] for i, j in itertools.combinations(range(5), 2)])
    torch.testing.assert_close(pair_features(a, b, "antisym"), expected)


def test_antisym_in_3d_is_the_cross_product():
    torch.manual_seed(0)
    a, b = torch.randn(3), torch.randn(3)
    p12, p13, p23 = pair_features(a, b, "antisym")
    c = torch.linalg.cross(a, b)
    torch.testing.assert_close(torch.stack([p23, -p13, p12]), c)


def test_antisym_flips_sign_when_order_swaps():
    torch.manual_seed(0)
    a, b = torch.randn(32), torch.randn(32)
    torch.testing.assert_close(pair_features(b, a, "antisym"), -pair_features(a, b, "antisym"))


def test_antisym_is_zero_for_parallel_vectors():
    a = torch.randn(32)
    assert pair_features(a, 2.5 * a, "antisym").abs().max() < 1e-5


def test_antisym_depends_only_on_the_plane():
    """Any invertible mix of (a, b) inside their plane scales the Plucker
    vector by det(M): after normalising, the same point up to sign."""
    torch.manual_seed(0)
    a, b = torch.randn(32), torch.randn(32)
    M = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    a2, b2 = M[0, 0] * a + M[0, 1] * b, M[1, 0] * a + M[1, 1] * b
    p, p2 = pair_features(a, b, "antisym"), pair_features(a2, b2, "antisym")
    torch.testing.assert_close(p2, torch.det(M) * p, rtol=1e-4, atol=1e-4)


def test_sym_is_symmetric_under_swap():
    torch.manual_seed(0)
    a, b = torch.randn(16), torch.randn(16)
    torch.testing.assert_close(pair_features(b, a, "sym"), pair_features(a, b, "sym"))


def test_antisym_plus_sym_recover_the_outer_product():
    """antisym and sym split the full product into its two halves."""
    torch.manual_seed(0)
    a, b = torch.randn(6), torch.randn(6)
    outer = torch.outer(a, b)
    anti = (outer - outer.T) / 2
    sym = (outer + outer.T) / 2
    torch.testing.assert_close(anti + sym, outer)
    i, j = torch.triu_indices(6, 6, offset=1)
    torch.testing.assert_close(pair_features(a, b, "antisym"), 2 * anti[i, j])


@pytest.mark.parametrize("kind", PAIR_KINDS)
def test_pair_dims(kind):
    r = PAIR_R[kind]
    a, b = torch.randn(2, 3, r), torch.randn(2, 3, r)
    assert pair_features(a, b, kind).shape == (2, 3, pair_dim(kind, r))


def test_feature_budgets_are_matched():
    """Every variant hands the projection about 496 numbers per pair."""
    dims = {k: pair_dim(k, PAIR_R[k]) for k in PAIR_KINDS}
    assert dims["antisym"] == 496
    assert all(abs(v - 496) <= 12 for v in dims.values()), dims


def test_causal_depthwise_conv_matches_loop():
    torch.manual_seed(0)
    x, w = torch.randn(2, 3, 10), torch.randn(3, 4)
    y = causal_depthwise_conv(x, w)
    for t in range(10):
        expected = sum(w[:, k] * x[:, :, t - k] for k in range(4) if t - k >= 0)
        torch.testing.assert_close(y[:, :, t], expected)


# ---------- every mixer: shape, causality, reach ----------

@pytest.mark.parametrize("name,mixer", MIXERS, ids=IDS)
def test_shape(name, mixer):
    h = torch.randn(2, 40, D)
    assert mixer.eval()(h).shape == h.shape


@pytest.mark.parametrize("name,mixer", MIXERS, ids=IDS)
def test_no_future_leak(name, mixer):
    """Changing positions after t0 must not change outputs at or before t0."""
    torch.manual_seed(0)
    mixer.eval()
    h = torch.randn(2, 48, D)
    h2 = h.clone()
    t0 = 20
    h2[:, t0 + 1:] = torch.randn_like(h2[:, t0 + 1:])
    torch.testing.assert_close(mixer(h)[:, :t0 + 1], mixer(h2)[:, :t0 + 1])


@pytest.mark.parametrize("name,mixer", [m for m in MIXERS if m[0] != "attention"],
                         ids=[n for n in IDS if n != "attention"])
def test_reach_is_exactly_16(name, mixer):
    """A change at position s reaches s + 16 but not s + 17."""
    torch.manual_seed(0)
    mixer.eval()
    assert mixer.reach == 16
    h = torch.randn(1, 64, D)
    h2 = h.clone()
    s = 10
    h2[:, s] += 1.0
    changed = (mixer(h) - mixer(h2)).abs().amax(dim=(0, 2)) > 1e-6
    assert changed[s + 16]
    assert not changed[s + 17:].any()


@pytest.mark.parametrize("name,mixer", MIXERS, ids=IDS)
def test_every_parameter_gets_gradient(name, mixer):
    torch.manual_seed(0)
    mixer.train()
    mixer(torch.randn(2, 32, D)).square().mean().backward()
    for pname, p in mixer.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, pname


def test_pair_mixer_handles_sequences_shorter_than_windows():
    m = PairMixer(D, kind="antisym", r=8).eval()
    for T in (1, 2, 5):
        out = m(torch.randn(1, T, D))
        assert out.shape == (1, T, D) and torch.isfinite(out).all()


def test_pair_mixer_finite_when_pair_vanishes():
    """Identical consecutive frames give an all-zero Plucker vector; the
    normalisation must not produce NaN."""
    m = PairMixer(D, kind="antisym", r=8).eval()
    h = torch.randn(1, 1, D).expand(1, 20, D).contiguous()
    assert torch.isfinite(m(h)).all()
