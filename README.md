# grassman

Does the Grassmann geometry matter? An ablation of *Attention Is Not What You Need: Grassmann Flows as an
Attention-Free Alternative for Sequence Modeling* (Zhang Chong, arXiv 2512.19428).

The paper's layer pairs each token with tokens 1–16 steps back and feeds the pair's **Plücker coordinates**
(the antisymmetric product a_i b_j − a_j b_i) into the model. It never tests whether that geometry is what helps.
This repo trains small language models that differ **only** in the token mixer:

| mixer | pair feature / mixing | reach |
|---|---|---|
| `grassmann` | antisymmetric product (the paper), r = 32 → 496 features | 16 |
| `sym` | symmetric product a_i b_j + a_j b_i, r = 31 → 496 | 16 |
| `outer` | full product a_i b_j, r = 22 → 484 | 16 |
| `concat` | [a; b], no product, r = 248 → 496 | 16 |
| `concat32` | [a; b] at the paper's r = 32 → 64 (exploratory, smaller by design) | 16 |
| `hyena` | order-2 Hyena, explicit 15-tap filter + width-3 short convs | 16 |
| `swa` | sliding-window attention | 16 |
| `attention` | full causal attention (baseline) | all |

Everything else is shared: 6 layers, d = 256, d_ff = 1024, pre-LN blocks, learned positions, tied embeddings,
bert-base-uncased WordPiece, block 128, batch 32, 30 epochs. All models are within 3% of the attention baseline's
12.59M parameters, which matches the paper's Table 1.

## Run it

**Quick pilot on an L4 (~2.5 h):** `notebooks/pilot_l4.ipynb`. It runs 5 models × 2 seeds × 5 epochs on validation only,
to see whether there is any effect before the full run.

**Follow-up tests** (one notebook each, validation only; rebuild them with `python notebooks/make_notebooks.py`):

| notebook | question | A100 time (estimate) |
|---|---|---|
| `speed.ipynb` | does the fast Grassmann layer give identical numbers, how much faster is it, where does time go? | ~5 min |
| `choose_vs_reach.ipynb` | is the gap to attention about *choosing* which token (learned edge weights) or *reach* (dilated edges)? | ~45 min |
| `hybrid.ipynb` | how much of the gap does one attention layer recover, and does the local mixer type matter? | ~1 h |
| `readouts.ipynb` | are the learned edge choices readable, used and meaningful? | ~15 min |

**Full experiment:**

Open `notebooks/grassman_colab.ipynb` in Colab with an L4 GPU and run the cells in order. It clones this repo,
runs the tests, builds the data, measures speed, sweeps the learning rate per mixer (1 seed, test set untouched),
runs 3 seeds at each mixer's best LR and prints the results described in [PROTOCOL.md](PROTOCOL.md). Results are
saved under `/content/results`; finished runs are skipped on re-run, and the last cell downloads them.

Locally (CPU or CUDA):

```bash
pip install -r requirements.txt
python -m pytest -q
python -m grassman.data --data-dir data
python -m grassman.train --mixer grassmann --seed 0 --lr 1e-3 --tag main
python -m grassman.analyze results/runs.jsonl --tag main
```

Training on Apple MPS currently fails at `torch.autocast` (PyTorch 2.4); use Colab or CPU.

## Fairness choices

- **Matched feature budget.** Each pair variant gets its own reduced width r so the projection always receives
  about 496 numbers per pair. A difference between `grassmann` and `sym` is then about the geometry, not size.
- **Matched reach.** Every windowed mixer sees exactly 16 tokens back; a test proves s → s+16 is reached and
  s+17 is not.
- **Per-model learning rate.** The paper does not give its optimizer settings, so each mixer picks its LR from
  {3e-4, 1e-3, 3e-3} on validation perplexity.
- **No test-set selection.** Test perplexity is read at the epoch with the best validation perplexity.
- **Seeds and paired comparisons.** 3 seeds per mixer; differences are computed seed by seed.

## Evaluation

The protocol in [PROTOCOL.md](PROTOCOL.md) was fixed before any run. In short:

- every choice (learning rate, reported epoch) is made on validation; the LR sweep never evaluates the test set;
- training noise: mean and 95% t-interval over 3 seeds, plus per-seed wins;
- test-set sampling: paired cluster bootstrap over test articles (10,000 resamples, same resamples for both
  systems), two-sided p-values;
- four comparisons are declared confirmatory and Holm-corrected (`grassmann` vs `sym`, `outer`, `concat`,
  `attention`); everything else is exploratory;
- breakdowns by position in the block (inside or beyond the 16-token window) and by target-token frequency;
- known limitations are listed in advance.

## Things to know about the paper

- Its equations pair position t with **t+Δ**, a future token, which would leak the answer in next-token
  prediction. The public reproduction (github.com/Infatoshi/grassmann-flows, `src/models/grassmann_v4.py`) pairs
  with t−Δ. So does this repo, and `test_no_future_leak` enforces it.
- The paper omits optimizer, LR, schedule, weight decay and dropout. We use AdamW (β = 0.9, 0.95), weight decay
  0.1, 500 warm-up steps, cosine decay to 10%, dropout 0.1 and grad-clip 1.0, mostly following the reproduction.
- The reproduction found a 22.6% perplexity gap to a transformer, against the paper's 10–15%.

## Tests

`python -m pytest -q`: 187 tests in about 90 s on CPU. They cover:

- **Plücker maths:** the definition, equality with the cross product in 3-D, sign flip under swap, zero for
  parallel vectors, scaling by det(M) within the plane.
- **Every mixer:** no future leak, exact reach, gradients reach every parameter, short sequences, zero pairs.
- **Full model:** causality, determinism, initial loss ≈ log(vocab), can memorise a batch, parameter matching.
- **Training and analysis:** article boundaries, block shifting, LR schedule, perplexity of a uniform model, a
  complete record from a 2-epoch run (per-article, per-position and per-frequency numbers agree with the overall
  perplexity), `--no-test` never touching the test set, exact reproducibility by seed, skip-existing, CI and Holm
  maths, the bootstrap (zero for identical systems, detects a known difference with the right sign, pairs only
  shared seeds), best-LR selection.

Planted bugs (pairing with the future, the wrong Plücker sign, an off-by-one attention window) each make a test
fail.

## Layout

```
grassman/mixers.py   PairMixer (4 pair kinds, fast path, optional edge selection), HyenaShortMixer, AttentionMixer
grassman/model.py    the LM and Config
grassman/data.py     Wikitext-2 download, tokenisation, blocks
grassman/train.py    one run -> one JSON line
grassman/analyze.py  tables, CIs, paired differences, best LR, pilot tables
grassman/bench.py    speed and fast-vs-paper-order checks
grassman/readout.py  edge usage, edge ablation, sin(angle) at sentence boundaries
notebooks/           the Colab notebook
PROTOCOL.md          evaluation protocol, fixed before any run
RESEARCH_LOG.md      what we tried and what happened
```
