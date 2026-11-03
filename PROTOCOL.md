# Evaluation protocol

Fixed on 2026-11-03, **before any real training run**. The git history shows this file was committed before the
first results. Any later change is recorded at the end of this file with its date and reason.

## Questions

- **Q1 (geometry).** At a matched feature budget (~496 numbers per pair), does the paper's antisymmetric
  (Plücker) pair feature give lower test perplexity than the symmetric product, the full product, or plain
  concatenation?
- **Q2 (reproduction).** How far is the paper's layer behind a size-matched transformer? The paper says 10–15%;
  the public reproduction found 22.6%.
- **Q3 (exploratory).** Is it better than other mixers that see the same 16 tokens: Hyena with a 16-step filter,
  and sliding-window attention? Does L2 normalisation matter?

## Data and splits

- Wikitext-2-raw with its standard train, validation and test splits, which come from different Wikipedia
  articles. Tokenised with bert-base-uncased WordPiece, as in the paper.
- **Every choice is made on validation:** the learning rate, and the epoch whose test score is reported.
- The LR sweep runs with `--no-test`, so it never evaluates the test set. The main runs evaluate test only at
  their best validation epoch.

## Selection rule

- Each mixer's learning rate is the one in {3e-4, 1e-3, 3e-3} with the lowest validation perplexity after
  10 epochs, seed 0.
- No other setting is tuned per mixer.
- The settings are fixed before the main runs; nothing is re-chosen after seeing test numbers.

## Runs

- 7 mixers, plus `grassmann` without normalisation, × 3 seeds (0, 1, 2) × 30 epochs.
- The seed changes initialisation, data order and dropout together.

## Uncertainty

- **Training noise:** mean and 95% Student-t interval over the 3 seeds; per-seed paired differences and how many
  seeds each side wins.
- **Test-set sampling:** paired **cluster bootstrap** over test articles, because tokens of one article are not
  independent.
  - 10,000 resamples of whole articles.
  - Per-article NLL averaged over seeds, and both systems scored on the same resamples.
  - 95% percentile interval; two-sided p-value with the +1 correction (Phipson & Smyth, 2010).
- **Limitation:** the bootstrap does not include training noise, and 3 seeds estimate that noise only roughly.
  Both are reported, and a claim needs both to agree (decision rule).

## Confirmatory family (Holm-corrected, α = 0.05)

The paper's layer (`grassmann`) against:

1. `sym`
2. `outer`
3. `concat`
4. `attention`

Comparisons 1–3 answer Q1 and comparison 4 answers Q2. **Everything else is exploratory:** `hyena`, `swa`,
`grassmann-raw`, the position and frequency breakdowns, and any comparison not on this list. Exploratory results
are reported but not used for claims.

## Decision rules

- **"The geometry matters" (Q1)** requires all three of `sym`, `outer` and `concat` to be worse than `grassmann`,
  with a Holm-adjusted p < 0.05 **and** `grassmann` winning in every seed.
- **"It does not matter"** is claimed if at least one of the three is not worse (adjusted p ≥ 0.05, or the
  difference favours the alternative). Then we report the size of the difference and its interval, not just the
  p-value.
- **Q2** is answered by the gap in percent with its interval, compared against 10–15% and 22.6%. It is not framed
  as a win or a loss.

## Breakdowns (exploratory)

- **Position in the 128-token block.** Positions 0–15 give every mixer the same context; positions 16–127 are
  where full attention can see further than the windowed mixers.
- **Frequency of the target token in training:** top 1k, 1k–10k, rare.

## Known limitations, stated in advance

- One small dataset, one model size (~12.6M) and one tokenizer. The result may not hold at scale.
- 3 seeds. Differences of about one seed standard deviation are within training noise.
- The test set has only about 60 articles, so the bootstrap intervals are coarse.
- The LR grid has 3 points, and other hyperparameters follow the reproduction rather than the paper (which omits
  them). They may suit some mixers better than others.
- The bf16 autocast on the L4 affects every mixer equally, but it is not the paper's precision.

## Changes after fixing this protocol

- **2026-11-03: pilot added (`notebooks/pilot_l4.ipynb`).**
  - What: `grassmann`, `sym`, `outer`, `concat` and `attention`; 2 seeds × 5 epochs at LR 1e-3.
  - **Validation only (`--no-test`)**, logged with tag `pilot` in a separate file.
  - Why: to see cheaply on an L4 whether there is any effect before spending ~9 A100-hours (or ~43 L4-hours) on the full runs.
  - The pilot does not change any rule above, and its results are exploratory and not used for claims.
  - It can inform only whether the full run is worth doing, not its settings.
- **2026-11-03: `concat32` added as an exploratory model.**
  - What: `concat` at the paper's r = 32, so 64 features per pair and the same token compression as `grassmann`.
  - **Added after seeing the pilot's validation results** (no test-set numbers existed).
  - Why: in the pilot `concat` matched `grassmann`, but it keeps 248 numbers per token instead of 32. `concat32`
    separates "no geometry" from "less compression".
  - Not part of the confirmatory family, and not parameter-matched (it is smaller). No confirmatory rule changes.
