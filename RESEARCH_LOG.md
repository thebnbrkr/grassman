# Research log

A running diary of what we tried, what happened, what went wrong, and what we changed. Entries are appended, never
rewritten.

---

## 2026-11-03: Set-up of the geometry ablation

**Status:** code and tests written; no real training run yet. Every claim below about results is a hypothesis.

### Question

Grassmann flows (arXiv 2512.19428) feed the Plücker coordinates of token pairs (t, t−Δ) into a language model. Is
the antisymmetric geometry what helps, or would any pairwise feature of the same size do as well? The paper does not
test this. One independent reproduction found the layer 22.6% behind a transformer, not the claimed 10–15%.

### What we built

Seven mixers in one LM (see README): the paper's layer, three pair-feature controls with a matched ~496-feature
budget (sym, outer, concat), and three standard mixers (Hyena with a 16-step filter, sliding-window attention with
window 16, full attention). Colab L4 notebook: LR sweep (1 seed, 10 epochs, 3 LRs) then 3 seeds × 30 epochs.

### Found while reading

- The paper's equations pair t with t+Δ (the future). The reproduction uses t−Δ. We use t−Δ and test for it.
- The paper omits optimizer settings, so every mixer gets its own LR from a sweep.

### Testing

84 tests pass on CPU (Mac, PyTorch 2.4). Three planted bugs each caused a failure: future pairing, the wrong Plücker
sign, and an off-by-one window. Wikitext-2 tokenises to 2,303,695 train, 238,656 validation and 273,178 test tokens.

### What went wrong

Training on Apple MPS fails at `torch.autocast` with autocast disabled (PyTorch 2.4). Not fixed. Colab uses CUDA,
where this path is not taken.

### How we'll know

- `grassmann` vs `sym` / `outer` / `concat`, paired by seed: if the 95% CI of the difference excludes 0 in favour of
  `grassmann`, the geometry matters at this scale. If not, it doesn't.
- `grassmann` vs `attention`: check the gap against the paper (10–15%) and the reproduction (22.6%).
- Next, if the geometry matters: the synthetic recall and bracket tasks, then larger models and the Hyena hybrid.

---

## 2026-11-03 (later): Evaluation brought up to the IWCLUL 2026 ASR paper's standard

**Status:** still no real training run.

### Why

The user's Hungarian ASR submission (IWCLUL 2026) set the bar: choices on dev only, a declared confirmatory family
with Holm correction, a cluster bootstrap with paired resamples, explicit seed-noise reporting, error breakdowns,
and limitations and test-informed decisions disclosed.

### What changed

- `PROTOCOL.md`, committed before any result: questions, selection rule, confirmatory family (`grassmann` vs `sym`,
  `outer`, `concat`, `attention`), decision rules, known limitations.
- The LR sweep runs with `--no-test`, so the test set is never evaluated during selection.
- Every main run stores per-article test NLL (Wikitext-2 article boundaries are the clusters), NLL by position in
  the block, and NLL by target-token frequency.
- `analyze.py`: paired cluster bootstrap (10,000 resamples), Holm-corrected confirmatory table, exploratory table,
  per-seed wins, position and frequency breakdowns.
- The notebook clones from GitHub and saves results under `/content/results`, with no Google Drive.
- 92 tests pass on CPU.

---

## 2026-11-03 (later): Full read of arXiv 2512.19428v1 and its references; measured speeds; L4 pilot

**Status:** still no real training run. The speeds below come from 200-step checks run by the user on Colab.

### What the paper says, checked against its own text

- **The pairing is written as non-causal.** §3.2 step 2: pairs are (z_t, z_{t+Δ}) "with t + Δ ≤ L", and "we only pair
  t with positions strictly to its right (future)". No code was released, so we cannot tell whether the runs leaked.
  Their Grassmann model is *worse* than the transformer, which suggests a writing slip rather than a leak.
- **The block differs from the reproduction and from ours.** The paper's block is post-LN: ĥ = LN(α⊙h + (1−α)⊙g),
  then h' = LN(ĥ + FFN(ĥ)). The gate replaces the residual around mixing. The reproduction and this repo use pre-LN
  (x + mixer(LN x), x + FFN(LN x)) for every mixer. This does not affect Q1 (all pair variants share the block), but
  it may affect Q2 (the gap to attention).
- **No training settings.** Only "the same optimizer and learning rate schedule in a shared script", 30 epochs, and
  batch 32 (L=128) or 16 (L=256). No LM dropout rate (0.1 is given only for the SNLI head).
- **It reports the best validation perplexity**, selected and reported on the same split. There is no test number
  for language modelling.
- **Table 1 vs the text.**
  - Table 1 pairs 275.7 with 248.4 and 282.3 with 253.6. The text says "275.7, compared to 241.0–253.6" (241.0 is in
    no table) and pairs 282.3 with 248.4.
  - The transformer has no window schedule, yet appears twice with different values (248.4 and 253.6). That is a
    2.1% difference between runs of the same configuration: a free estimate of run-to-run noise.
- **Gaps:** 11.0% and 11.3% (6 layers), 11.0% (12 layers, block 256). The claim that "the gap appears to narrow as
  depth increases" is not supported: it is 11.0% in both.
- **Parameters:** GrassmannLM has 68–70k more parameters per layer than its transformer (6 and 12 layers). That is
  close to one 256×256 linear layer (65,792), which our layer does not have. Unexplained; no code.
- **Compute.** The paper says the Plücker projection is "absorbed into O(L d²)". At its own sizes the Grassmann
  mixer is ~901k multiply-adds per token against ~295k for attention at L=128, about 3× more. The cause is the
  projection: m·C(r,2)·d = 6·496·256. Because projecting and averaging are both linear, averaging the six
  normalised Plücker vectors *before* one projection is mathematically the same and costs ~266k, about attention's
  cost. Not implemented.
- **SNLI:** one run each. A 0.0027 test-accuracy difference is about 27 of ~9.8k examples, with no variance
  reported.

### References (14)

- **[1]–[5]:** Transformer, BERT, GPT-2, GPT-3, ViT.
- **[6], [8]:** WikiText (the same paper twice). **[4], [7]:** GPT-3 (twice). **[9]:** SNLI.
- **[10]:** Image Transformer.
- **[11]–[14]:** textbooks and a survey on algebraic geometry, Riemannian manifolds and optimisation on
  Grassmann/Stiefel manifolds (Hartshorne; Lee; Edelman, Arias & Smith 1998; Absil, Mahony & Sepulchre 2008).
- Cited in the text: [1, 2, 4, 5] (introduction), [11–14] (Grassmann/Plücker background), [6, 9] (datasets).
  [3], [7], [8] and [10] are never cited.
- **Related Work (§7.1–7.4) cites nothing.** No efficient-attention, state-space, Hyena, geometric deep learning
  or attention-interpretability papers are referenced. No reference contains any experiment on Grassmann
  features in sequence models, so the empirical claims rest only on the paper's own tables.

### Measured speed (200-step checks, user's Colab)

| mixer | A100 tok/s | L4 tok/s |
|---|---|---|
| grassmann | 37,832 | 8,778 |
| sym | 40,676 | 8,818 |
| outer | 69,249 | 16,405 |
| concat | 74,727 | 16,261 |
| hyena | 143,736 | 23,839 |
| swa | 186,352 | 23,993 |
| attention | 207,660 | 23,839 |

- Full protocol: ~9 h on the A100, ~43 h on the L4.
- On the L4, hyena, swa and attention run at the same speed, so the time goes to a part all models share
  (unprofiled).
- `grassmann` and `sym` are ~2× slower than `outer`, which has about the same feature count. The upper-triangle
  gather is the likely cause.

### Pilot

`notebooks/pilot_l4.ipynb`: 5 models × 2 seeds × 5 epochs, validation only, ~2.5 h on the L4. It is recorded in
PROTOCOL.md as an exploratory addition that changes none of the confirmatory rules.

---

## 2026-11-03 (later): Pilot results

**Status:** pilot finished. Validation only, 5 epochs, so these are exploratory, not confirmatory.

### What ran

`notebooks/pilot_l4.ipynb`, but on an **A100** (the notebook's GPU check shows A100-SXM4-80GB), commit `3c8a1ea`.
5 models × 2 seeds × 5 epochs, LR 1e-3 for all. Total ~38 min. The test set was never evaluated.

### Results (best validation perplexity; the best epoch was 5 for every run)

| mixer | seed 0 | seed 1 | vs grassmann |
|---|---|---|---|
| concat (r = 248) | 268.0 | 269.1 | −0.5% |
| grassmann (Plücker) | 269.9 | 269.8 | — |
| outer | 271.8 | 271.7 | +0.7% |
| sym | 280.2 | 280.2 | +3.8% |
| attention | 199.1 | 201.1 | −25.9% |

The seed spread is 0.0–1.1 perplexity for the pair mixers and 2.0 for attention. The seeds differ at epoch 1
(e.g. grassmann 467.9 vs 461.0) and converge by epoch 5.

### What it suggests

- **The Plücker geometry does not beat plain concatenation.** Under the protocol's decision rule, "the geometry
  matters" would fail.
- **Order is what matters.** `sym`, the only variant that cannot tell the two tokens apart, is the only clearly
  worse one. Antisymmetry helps relative to `sym` because it encodes order, which `concat` and `outer` also do.
- **Attention is ~26% better at 5 epochs.** The paper says 11%, Arledge 22.6%, Norman 43%.
- **The paper's transformer baseline looks weak.** With the same data, tokenizer and size, our attention reaches
  200 validation perplexity after 5 epochs, against the paper's 248.4 after 30 (Norman's: 248.7). Our Grassmann
  reaches 269.9 against the paper's 275.7. Our Grassmann roughly matches theirs; our transformer is ~20% better.
  Likely causes: pre-LN, weight decay 0.1, warm-up and cosine schedule. Unconfirmed until the 30-epoch runs.

### Caveats

- 5 epochs, and every model is still improving, so the ranking could change at 30 epochs.
- `concat` keeps 248 numbers per token and is 2.6% larger, so its result mixes "no geometry" with "less
  compression". `concat32` (r = 32) has been added to separate the two (PROTOCOL.md change log).

### What worked and what didn't

- **Worked:** the pilot design (cheap, validation only, protocol intact), seed consistency, and the tests (all
  passed on Colab: 92 at that commit).
- **Didn't:**
  - The L4 is ~5× slower than the A100, with a shared bottleneck that hasn't been profiled.
  - `grassmann` and `sym` are ~2× slower than `outer` because of the upper-triangle gather.
  - Our block is pre-LN with two residual paths, not the paper's post-LN block. That is fine for Q1 but open
    for Q2.

---

## 2026-11-03 (later): Speed fix, edge-choosing layer, hybrids; predictions before running

**Status:** code and four notebooks written; nothing has run yet. 187 tests pass on CPU.

### What was built

- **Fast PairMixer** (identical maths):
  - average the per-offset pair statistics, then project once (6× fewer multiply-adds in the projection);
  - build the average from r×r products, so the upper-triangle gather happens once per layer;
  - normaliser in closed form (Lagrange's identity for antisym).
  - `reference_forward` keeps the paper's order. Tests compare outputs and all gradients in float64 for every
    kind, offset set and length. Planted bugs (sign, missing mask, wrong norm) each fail 10–49 tests.
- **Edge selection** (`select=True`): a softmax over the offsets, scored GATv2-style from [z_t; z_{t−Δ}] plus a
  per-offset bias, optionally plus sin(angle). It starts as the equal average (tested). This is graph attention
  on a causal graph whose edges are the offsets, from the user's GNN idea.
  - Variants: `g-select` (paper offsets), `g-dilated` (offsets 1, 2, 4, 16, 64, 127, equal weights), `g-both`
    (dilated plus selection), `g-both-angle`.
- **Hybrids:** `--attn-layers k` makes layer k full attention.
- **Readouts:** `--save` keeps the best-validation weights. `readout.py` computes edge usage, edge ablation and
  sin(angle) at sentence boundaries.

### Predictions, written before any run (guesses; to be checked)

- **speed:** the fast path matches the paper's order to ~1e-5 in float32. `grassmann` and `sym` get ~1.5–3× faster
  on the A100. In the profile, the shared parts (output layer and loss over 30,522 words, embedding gradient,
  optimizer) take much of the step on the L4.
- **choose vs reach:**
  - Choosing alone and reach alone each help a little (a few %). Both together help most.
  - `g-both` still stays well behind attention, closing maybe 20–40% of the gap.
  - Attention's edge at this scale is likely content-based lookup over the whole block (repeated names in an
    article), which needs both choosing *and* reach.
- **hybrid:**
  - One attention layer recovers most (≥ 60%) of each local mixer's gap.
  - Inside a hybrid, the local mixer type matters little (≤ 3%).
  - Attention at layer 3 vs layer 6: a small difference.
- **readouts:**
  - Early layers put most weight on t−1, with a top weight well above 1/6.
  - Patterns are broadly similar across seeds. Removing t−1 in layer 0 costs the most.
  - sin(angle) is slightly higher after sentence ends. The angle weights stay small.

---

## 2026-11-03 (later): Speed notebook results (A100, commit ca4b164)

### Same numbers

Fast path vs the paper's order, float32, all four kinds, with and without normalisation: max output difference
≤ 1.2e-6, max gradient difference ≤ 1.9e-9. That is rounding. **Pass.**

### Speed (training tok/s, random tokens, batch 32 × 128, bf16)

| model | fast | paper order | speed-up |
|---|---|---|---|
| grassmann | 67,363 | 48,647 | 1.4× |
| sym | 59,119 | 50,557 | 1.2× |
| outer | 76,643 | 77,021 | 1.0× |
| concat | 81,095 | 85,016 | 1.0× |
| concat32 | 82,926 | 84,515 | 1.0× |
| g-dilated | 69,239 | 59,200 | 1.2× |
| g-select / g-both / g-both-angle | 43,808 / 43,786 / 30,849 | — | — |
| hyena / swa / attention | 231,291 / 268,110 / 287,824 | — | — |

**The prediction was wrong.** 1.4× for `grassmann`, against the predicted 1.5–3×. The fast `grassmann` is still 4.3×
slower than attention.

### Profile (10 steps)

- **attention:** 10.3 ms/step. matmul 24%, dtype copies 17%, AdamW 16%, log-softmax backward 12%, cross-entropy
  ~10%. The shared parts (output layer, loss, optimizer, casts) dominate. This was measured on the A100, so the L4
  question is still unprofiled.
- **grassmann:** 37.6 ms/step. **The upper-triangle gather's backward (index_put) is 26% of GPU time**, the largest
  single item, even though it now runs once per layer. Element-wise multiplies are 19%, copies 10%, sums 10%:
  building six (B, T, r, r) products per layer.

### Finding: the multi-offset layer collapses to one wedge

Every pair shares z_t as its first vector, and the Plücker map is linear in its second argument. So the averaged
feature is

  Σ_Δ w_Δ · (z_t ∧ z_{t−Δ}) / ‖z_t ∧ z_{t−Δ}‖  =  z_t ∧ b̄_t,   with  b̄_t = Σ_Δ (w_Δ / ‖z_t ∧ z_{t−Δ}‖) · z_{t−Δ}

- **The paper's "multi-scale Grassmann flow" is the wedge of the current token with a weighted average of its
  past window.** Without normalisation, the weights are uniform and b̄ is a plain causal moving average of z.
- The same holds for every pair kind: concat becomes [z_t; b̄_t], and sym and outer use z_t ⊗ b̄_t.
- This may explain why the geometry adds nothing over concat. The layer only ever sees the current token against
  an average of its past, never a choice among past tokens.
- **Speed:** b̄_t costs six cheap r-dimensional shifted sums, followed by **one** product and one gather per layer.
  Folding the gather into the projection weights removes the slow index backward entirely. Exact; not yet
  implemented.
