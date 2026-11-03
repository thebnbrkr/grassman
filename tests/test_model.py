import math

import pytest
import torch

from grassman.model import LM, MIXERS, UNMATCHED, Config

TINY = dict(vocab_size=101, block_size=32, d=32, d_ff=128, n_layers=2, n_heads=4)


@pytest.mark.parametrize("mixer", MIXERS)
def test_full_model_is_causal(mixer):
    torch.manual_seed(0)
    model = LM(Config(mixer=mixer, **TINY)).eval()
    idx = torch.randint(0, 101, (2, 32))
    idx2 = idx.clone()
    idx2[:, 17:] = torch.randint(0, 101, (2, 15))
    torch.testing.assert_close(model(idx)[0][:, :17], model(idx2)[0][:, :17])


@pytest.mark.parametrize("mixer", MIXERS)
def test_same_seed_same_model(mixer):
    torch.manual_seed(3)
    a = LM(Config(mixer=mixer, **TINY))
    torch.manual_seed(3)
    b = LM(Config(mixer=mixer, **TINY))
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(p, q), n


@pytest.mark.parametrize("mixer", MIXERS)
def test_initial_loss_is_near_uniform(mixer):
    """With N(0, 0.02) init the first loss should be close to log(vocab)."""
    torch.manual_seed(0)
    model = LM(Config(mixer=mixer, **TINY)).eval()
    idx = torch.randint(0, 101, (4, 32))
    _, loss = model(idx, idx)
    assert abs(loss.item() - math.log(101)) < 0.5


@pytest.mark.parametrize("mixer", MIXERS)
def test_can_memorise_a_batch(mixer):
    """Gradients flow end to end: 150 steps should fit one fixed batch."""
    torch.manual_seed(0)
    model = LM(Config(mixer=mixer, dropout=0.0, **TINY))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 101, (4, 32))
    y = torch.roll(x, -1, dims=1)
    first = None
    for _ in range(150):
        _, loss = model(x, y)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3 * first, (first, loss.item())


def test_head_is_tied_to_embedding():
    model = LM(Config(**TINY))
    assert model.head.weight is model.tok.weight


def test_paper_scale_parameter_counts_match():
    """At the paper's 6-layer size every variant is within 3% of the
    attention baseline, which itself matches the paper's 12.59M."""
    counts = {m: LM(Config(mixer=m)).n_params() for m in MIXERS if m not in UNMATCHED}
    ref = counts["attention"]
    assert abs(ref / 1e6 - 12.59) < 0.05, ref
    for m, n in counts.items():
        assert abs(n / ref - 1) < 0.03, (m, n, ref)


def test_concat32_keeps_the_papers_token_compression():
    """concat32 reduces tokens to r = 32 like grassmann, so it sees 64 pair
    features instead of ~496, and is smaller than the matched models."""
    from grassman.mixers import pair_dim
    m = LM(Config(mixer="concat32")).blocks[0].mixer
    g = LM(Config(mixer="grassmann")).blocks[0].mixer
    assert m.r == g.r == 32 and pair_dim(m.kind, m.r) == 64
    assert LM(Config(mixer="concat32")).n_params() < LM(Config(mixer="attention")).n_params()


def test_padded_output_layer_is_exact():
    torch.manual_seed(0)
    model = LM(Config(**TINY)).double()          # vocab 101 -> padded to 128 for the matmul
    x = torch.randn(2, 5, 32, dtype=torch.float64)
    torch.testing.assert_close(model.logits(x), model.head(x))
    assert model.logits(x).shape[-1] == 101


@pytest.mark.parametrize("mixer", MIXERS)
def test_torch_compile_traces_without_graph_breaks(mixer):
    """fullgraph=True fails on any graph break; backend='eager' checks tracing
    only, so this runs on CPU without a compiler. Outputs must match."""
    torch._dynamo.reset()
    torch.manual_seed(0)
    model = LM(Config(mixer=mixer, dropout=0.0, **TINY)).eval()
    idx = torch.randint(0, 101, (2, 32))
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(idx, idx)[1], model(idx, idx)[1])
