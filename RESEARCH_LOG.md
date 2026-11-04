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

---

## 2026-11-03 (later): L4 speed results, and the one-pair fast path implemented

### L4 numbers (the same speed notebook, commit ca4b164; the user ran it on both GPUs)

| model | fast | paper order | speed-up |
|---|---|---|---|
| grassmann | 15,655 | 8,842 | 1.8× |
| sym | 14,635 | 8,271 | 1.8× |
| outer | 17,955 | 14,040 | 1.3× |
| concat / concat32 | 16,397 / 21,002 | 14,437 / 19,669 | 1.1× |
| g-dilated | 13,922 | 9,189 | 1.5× |
| g-select / g-both / g-both-angle | 12,877 / 13,460 / 13,143 | — | — |
| hyena / swa / attention | 20,777 / 20,739 / 20,564 | — | — |

- **The L4 slowdown is the matrix multiplies.** In the attention profile, `aten::mm` is 65.8% of GPU time and every
  linear layer (25 per step, 3 kernels each) runs `magma_sgemmEx_kernel<float, __nv_bfloat16, ...>`, at about 2 ms
  per call whatever the shape. That is a fallback kernel; the tensor cores are not being used.
- Attention is 14× slower than on the A100 (20.6k vs 288k tok/s), where spec sheets suggest ~2.6×. Every non-pair
  model hits the same ceiling (~20.6k).
- The cause (library choice, vocabulary alignment of 30,522, precision, or a throttled GPU) is not known yet.
  `speed.ipynb` section 4 now times plain matmuls by precision (bf16/fp16/fp32/TF32), BLAS library (cuBLAS vs
  cuBLASLt), and padded vs unpadded vocabulary, and prints clocks and power.

### One-pair fast path (implemented)

- PairMixer now builds b_t = Σ_Δ s_Δ z_{t−Δ} (s_Δ = edge weight / ‖pair‖) and **one** pair (z_t, b_t) per token.
  For concat, the first half is z_t · Σ_Δ s_Δ.
- The upper-triangle selection is folded into the projection weights (`_folded_weight`, an index_add on the
  d × 496 weight). The per-token gather is gone; the folded projection runs in fp32 to keep the a_i b_j − a_j b_i
  cancellation exact.
- It equals the paper's order in float64 for all kinds, offset sets and lengths (outputs and gradients).
  Two planted bugs were each caught: the folded sign failed 14 tests, the concat scale 7. 187 tests pass.
- Speed is not measured yet. Re-run `speed.ipynb`.

---

## 2026-11-03 (later): Correction — the "L4" speed runs were on a Tesla T4; A100 one-pair results

### The GPU was a T4

The speed notebook's own record (`speed.json` in the user's results zip, 08:08) says `"gpu": "Tesla T4"`. The
earlier "L4" speed numbers (attention ~20.6k tok/s, all models stuck at the same ceiling, matmuls on
`magma_sgemmEx_kernel<float, __nv_bfloat16, ...>`) match this run, so they were very likely T4 runs too. The first
200-step smoke test cannot be checked, because its device was not pasted.

- **Why everything was slow:** the T4 (Turing, compute capability 7.5) has tensor cores for fp16 but **not
  bf16**. `torch.cuda.is_bf16_supported()` still returns True, so our code chose bf16 autocast, and every matmul
  fell back to a slow non-tensor-core kernel.
- **Fix (not implemented yet):** use fp16 autocast with a GradScaler on GPUs below compute capability 8.0, and
  bf16 on A100, L4 and H100.
- The earlier entries that blame "the L4" (the ~43 h estimate and the "shared bottleneck") should be read as T4.

### T4, one-pair version (tok/s)

grassmann 12,306 (paper order 7,810, 1.6×); sym 13,921 (7,887, 1.8×); outer 17,446 (14,431); concat 16,788
(14,438); concat32 20,251 (19,423); g-dilated 13,963 (9,138, 1.5×); g-select 12,695; g-both 13,258;
g-both-angle 12,803; hyena 20,760; swa 20,916; attention 20,849.

### A100, one-pair version (commit 905a014)

- **Equivalence:** ≤ 1.2e-6 (output), ≤ 1.9e-9 (gradient).
- **tok/s:** grassmann 63,641 (paper order 48,571), sym 55,616, outer 73,902, concat 70,640, concat32 69,965,
  g-select 41,484, g-dilated 66,601, g-both 42,427, g-both-angle 29,600, hyena 214,908, swa 248,355,
  attention 273,174.
- **Relative to attention, grassmann is unchanged from the first fix** (0.233 vs 0.234). The whole run was ~5%
  slower.
- **Profile:** the index backward is gone, and grassmann GPU time per 10 steps fell 376 → 240 ms. But CPU time is
  985 ms: the model is **CPU / kernel-launch bound** (~800 small ops per step from the per-offset loop). Attention
  is partly launch-bound too (259 ms CPU vs 102 ms GPU).
- **Matmul check:**
  - bf16 tensor cores work (83–193 TFLOP/s).
  - Padding the vocabulary 30,522 → 30,528 makes the output-layer matmul 0.77 → 0.40 ms (cuBLAS) and 0.64 →
    0.33 ms (cuBLASLt).
  - fp32 15–19 TFLOP/s, TF32 55–93.

### Next speed changes (proposed)

1. fp16 + GradScaler below compute capability 8.0 (T4, V100).
2. Pad the output-layer vocabulary to a multiple of 8, exact by slicing the logits.
3. Batch the offset loop into one tensor op and enable `torch.compile` for all models.

---

## 2026-11-03 (later): Three more speed changes (not yet measured on a GPU)

1. **No Python loop over offsets.**
   - `PairMixer._past` takes all offsets from one strided window (`unfold`) and one small `index_select`, giving
     (B, T, E, r). Norms, edge scores and the weighted average are then single batched operations (`einsum`).
   - `pair_norm` now clamps the *square* at eps²: same value as clamping the norm, but a finite gradient at
     zero-length pairs. Zero pairs now occur for every position before an offset exists, and the earlier version
     gave NaN gradients there; caught by the fp64 equivalence test.
2. **Padded output layer.** The matmul uses the vocabulary padded to a multiple of 64 (30,522 → 30,528), then
   slices back. Exactly `head(x)` (tested in fp64).
3. **Precision by GPU.** bf16 on compute capability ≥ 8; fp16 with a GradScaler below that (T4). `--amp`
   overrides, and the run record stores `amp` and `compile`.
4. **`torch.compile`.** `speed.ipynb` times every model eager and compiled. The experiment notebooks have
   `COMPILE = False` until that measurement says otherwise. Every model traces with `fullgraph=True` (tested on
   CPU with the eager backend).

**Tests:** 210 pass. New: finite gradients with zero pairs (all kinds, with and without selection), the padded
head is exact, the precision rule (T4 → fp16, A100 → bf16), throughput on CPU, and no graph breaks for any
model.

**Expectations (qualitative only, after the 1.5–3× miss):**
- On the T4, fp16 should lift every model above the ~21k tok/s ceiling.
- On the A100, compiling should help the pair models most, since they were launch-bound.
- To be measured with `speed.ipynb` on both GPUs.

**Caveat:** fp16 (T4) and bf16 (A100) runs are not directly comparable. Each notebook's models must run on one
GPU type.

---

## 2026-11-03 (later): A100 speed after the three changes (results (8).zip, speed.json)

Training tok/s, bf16, batch 32 × 128, random tokens. "paper" = the paper's order, eager.

| model | eager | compiled (compile s) | paper | eager / paper | compiled / paper |
|---|---|---|---|---|---|
| grassmann | 131,333 | 273,504 (36) | 48,426 | 2.71× | **5.65×** |
| sym | 118,083 | 261,242 (30) | 50,293 | 2.35× | 5.19× |
| outer | 135,971 | 283,433 (24) | 73,221 | 1.86× | 3.87× |
| concat | 139,436 | 271,798 (26) | 77,193 | 1.81× | 3.52× |
| concat32 | 135,883 | 263,189 (24) | 76,711 | 1.77× | 3.43× |
| g-select | 100,555 | 197,661 (34) | — | | |
| g-dilated | 129,039 | 256,049 (24) | 56,757 | 2.27× | 4.51× |
| g-both | 95,812 | 212,810 (31) | — | | |
| g-both-angle | 81,612 | 83,653 (1) | — | | |
| hyena | 212,839 | 209,012 (0) | — | | |
| swa | 253,400 | 242,873 (0) | — | | |
| attention | 263,281 | 251,139 (0) | — | | |

- **Eager `grassmann` doubled** from the one-pair version (63.6k → 131k): batching the offsets removed the
  launch bottleneck.
- **Compiled `grassmann` (273k) now matches eager attention (263k).**
- **Measurement bug:** the last four "compiled" numbers (g-both-angle, hyena, swa, attention) show 0–1 s compile
  time and eager-level speed. Dynamo's recompile limit (8 graphs for `LM.forward`) was reached after the first
  eight models, so these ran eager. Their compiled speed is **not measured**.
  - Fixed: `throughput` now calls `torch._dynamo.reset()` before compiling.
  - Training runs are separate processes and are not affected.
- The equivalence (section 1), profile and matmul outputs are not in the zip; to check from the notebook.

---

## 2026-11-03 (later): T4 speed after the three changes (speed (2).ipynb, commit a66bcde)

- **Equivalence:** ≤ 9.5e-7 (output), ≤ 1.9e-9 (gradient). Pass. 210 tests pass on the T4.
- **Precision chosen:** fp16, as intended.
- **Matmul check (the cause of the earlier slowness, now confirmed):**
  - bf16 runs at **1.6 TFLOP/s** (fallback kernel); fp16 at **13.5 TFLOP/s** (8.4× faster); fp32 3.3.
  - Padding the vocabulary does not help on the T4 (fp16 4.75 → 5.14 ms).
  - The GPU read 585 MHz of 1590 max at 71 °C during the check, so it may be throttled, and these are lower
    bounds.

| model | eager | compiled (s) | paper order | eager / paper | compiled / paper |
|---|---|---|---|---|---|
| grassmann | 49,393 | 99,113 (56) | 12,007 | 4.1× | **8.3×** |
| sym | 46,634 | 86,628 (47) | 11,762 | 4.0× | 7.4× |
| outer | 50,583 | 94,026 (39) | 32,609 | 1.6× | 2.9× |
| concat | 29,433 | 71,286 (41) | 33,470 | **0.9×** | 2.1× |
| concat32 | 54,100 | 100,933 (32) | 48,158 | 1.1× | 2.1× |
| g-select / g-dilated / g-both | 43,188 / 34,278 / 33,258 | 80,912 / 73,292 / 79,420 | — / 13,541 / — | | |
| g-both-angle / hyena / swa / attention | 32,618 / 57,301 / 61,816 / 59,486 | not measured | | | |

- fp16 lifted attention from ~20.8k (bf16 fallback) to 59.5k tok/s (2.9×).
- `concat` (r = 248) is slower eager than the paper's order on the T4: its (B, T, E, 248) past tensor is large.
- The last four "compiled" numbers hit dynamo's recompile limit (warning in the log); this run predates the
  `torch._dynamo.reset()` fix (43a1058). Compiled attention is still unmeasured on both GPUs.

### What the speed work does and does not show

- **Shows:** with an exact rewrite (one pair per token, batched offsets, compile), the paper's layer costs about
  the same as attention at length 128 on the A100 (274k vs 263k eager attention). Its slowness in the paper was
  the implementation, not the idea. That partly supports the paper's efficiency remark ("with further engineering
  ... can in principle be more scalable").
- **Does not show:**
  - anything about quality (pilot: 26% behind attention; Plücker no better than concat);
  - the linear-scaling advantage, which only matters at long sequences (not tested);
  - a comparison with *compiled* attention (not yet measured).

---

## 2026-11-03 (later): Choose vs reach — results (A100, commit 53dd11a, compiled, bf16, 2 seeds × 5 epochs)

| model | seed 0 | seed 1 | mean | vs grassmann | vs attention |
|---|---|---|---|---|---|
| grassmann (equal weights, offsets 1–16) | 269.9 | 270.4 | 270.2 | — | +35.0% |
| g-dilated (equal weights, offsets to 127) | 258.2 | 258.3 | 258.3 | −4.4% | +29.0% |
| g-select (learned weights, offsets 1–16) | 215.6 | 216.2 | **215.9** | **−20.1%** | **+7.8%** |
| g-both (learned weights, offsets to 127) | 217.4 | 218.8 | 218.1 | −19.3% | +8.9% |
| swa (attention over the last 16) | 202.0 | 203.9 | 202.9 | −24.9% | +1.3% |
| attention (all 128) | 200.8 | 199.6 | 200.2 | −25.9% | — |

Best epoch 5 for every run, and every model still improving. Seed spread 0.1–1.9.

### What it shows

- **The gap to attention is about choosing, not reach.**
  - Learned edge weights over the same 6 offsets close **78%** of grassmann's gap to attention
    ((270.2 − 215.9) / (270.2 − 200.2)), at linear cost.
  - Extending reach to 127 with equal weights gives only 4.4%.
  - With selection already present, the long edges add nothing: g-both is 1.0% *worse* than g-select. The dilated
    set swaps offsets 8 and 12 for 64 and 127.
- **Reach beyond 16 tokens barely matters at this scale.** swa (16 tokens back) is within 1.3% of full attention.
- **The remaining g-select vs swa gap (6.4%)** differs in candidate count (6 offsets vs 17 positions), multiple
  heads, scoring (GATv2 on r = 32 vs dot products), and the message (a wedge with the weighted past vs values in
  full d). Untested which matters.
- **Sanity check:** grassmann 270.2 and attention 200.2 reproduce the pilot (269.9 and 200.1) after the speed
  rewrite, compile and padded head, so the speed changes did not change results.

### Prediction check (written before running)

**Wrong.** I predicted that choosing alone and reach alone would each help "a little", that both together would
help most, and that g-both would close 20–40% of the gap. In fact choosing alone closed ~78%, reach helped little,
and both together was no better than choosing.

### Notes

- tok/s for seed 0 includes compile time; seed 1 reused the compile cache, so it is faster (e.g. grassmann 98k vs
  214k).
- Per run: 0.7–2.4 min on the A100.

### Open questions this raises

1. Once the layer chooses, does the Plücker message still matter? Compare g-select with concat32 + selection.
2. Does the remaining 6% come from candidate count? Compare g-select over all 16 offsets 1–16.

---

## 2026-11-03 (later): Readouts — results (g-both-angle, NVIDIA L4, bf16, compiled, 2 seeds × 5 epochs)

**Perplexity:** 217.37 / 218.83, against g-both without the angle 217.40 / 218.85 (A100). Same seed, same initial
weights, and the angle weight starts at 0. **The angle changes perplexity by ~0.01%.** The saved weights reproduce
the logged validation perplexity.

### 1. Readable — edge share per layer (positions where all edges exist)

| layer | seed | t−1 | t−2 | t−4 | t−16 | t−64 | t−127 | top weight |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 / 1 | 0.59 / 0.62 | 0.15 / 0.14 | 0.07 / 0.06 | 0.06 / 0.06 | 0.06 / 0.06 | 0.08 / 0.06 | 0.59 / 0.62 |
| 1 | 0 / 1 | 0.54 / 0.55 | 0.15 / 0.19 | 0.11 / 0.08 | 0.08 / 0.05 | 0.06 / 0.05 | 0.05 / 0.08 | 0.56 / 0.56 |
| 2 | 0 / 1 | 0.54 / 0.44 | 0.21 / 0.19 | 0.09 / 0.17 | 0.05 / 0.07 | 0.04 / 0.05 | 0.07 / 0.07 | 0.54 / 0.47 |
| 3 | 0 / 1 | 0.38 / 0.29 | 0.27 / 0.14 | 0.21 / 0.21 | 0.05 / 0.19 | 0.04 / 0.10 | 0.05 / 0.08 | 0.45 / 0.33 |
| 4 | 0 / 1 | 0.27 / 0.44 | 0.17 / 0.26 | 0.22 / 0.15 | 0.17 / 0.05 | 0.09 / 0.04 | 0.08 / 0.06 | 0.33 / 0.49 |
| 5 | 0 / 1 | 0.45 / 0.41 | 0.22 / 0.27 | 0.17 / 0.19 | 0.05 / 0.04 | 0.04 / 0.04 | 0.06 / 0.06 | 0.48 / 0.44 |

- **Seed agreement** (correlation of shares) per layer: 1.00, 0.99, 0.97, 0.75, 0.79, 0.98.
- **Layers 3 and 4 swap roles between seeds.** Seed 0's layer 4 (0.27/0.17/0.22/0.17/0.09/0.08) matches seed 1's
  layer 3 (0.29/0.14/0.21/0.19/0.10/0.08), and seed 0's layer 3 matches seed 1's layer 4. The same two jobs appear
  in both seeds, in a different layer order: a "local" job (t−1, t−2, t−4) and a "wider" job (which also uses
  t−16).
- Top weights are 0.33–0.62, against 0.17 for no preference.
- Long edges (t−64, t−127) get 4–10%.

### 2. Used — removing one edge (Δ mean NLL in nats, first 300 validation blocks)

- **t−1:** layer 0 +0.194 / +0.165 (≈ +18–21% perplexity), layer 1 +0.097 / +0.115, layer 2 +0.076 / +0.038,
  later layers +0.007 to +0.029.
- **t−16** matters only in the "wider" layer: seed 0 layer 4 +0.006, seed 1 layer 3 +0.010.
- **t−64 and t−127:** ≤ 0.001 everywhere. Removing them changes nothing.
- The shares predict the ablation effects: high-share edges cost the most when removed.

### 3. Meaningful — sin(angle) between z_t and z_{t−1}, after a sentence end (`.` `!` `?`, n = 8,966) vs elsewhere

- **Higher after a sentence end in all 12 layer × seed cases.** Differences +0.017 to +0.092; for example seed 0
  layer 1: 0.860 vs 0.772, layer 2: 0.893 vs 0.801.
- **Learned angle weights have a consistent sign pattern across seeds:** layer 0 positive (+0.41 / +0.36), layers
  1–5 negative (−0.03 to −0.64); layer 5 strongly negative in both (−0.61 / −0.64).
- No significance test was run. The baseline sin is already high (0.6–0.92).

### Prediction check

- **Right:** early layers favour t−1 with top weight well above 1/6; patterns broadly similar across seeds;
  removing t−1 in layer 0 costs most; sin(angle) slightly higher after sentence ends.
- **Wrong:** "angle weights stay small". Layer 5 is about −0.6 in both seeds. The angle still does not help
  perplexity.

### What it shows, and what it does not

- **Shows:** the edge-choosing layer can be read from 6 numbers per layer, and the reading is reproducible across
  seeds, up to layer order. It agrees with causal ablations. Its angle carries a weak but consistent boundary
  signal. Long edges are unused, matching choose-vs-reach (reach did not help).
- **Does not show:** that it is *more* readable than attention. The same summaries (distance histogram per head,
  ablation by distance) have not been computed for swa or attention. Also, 5 epochs only.

---

## 2026-11-03 (later): Hybrid — results (A100-SXM4-40GB, commit adbcdb0, bf16, compiled, 2 seeds × 5 epochs)

| model | seed 0 | seed 1 | mean | vs attention |
|---|---|---|---|---|
| **hyena+attn@5** | 183.5 | 184.7 | **184.1** | **−8.0%** |
| **hyena** (local only, 16 back) | 192.5 | 191.8 | **192.1** | **−4.0%** |
| g-both+attn@5 | 197.4 | 199.4 | 198.4 | −0.9% |
| attention (all 6 layers) | 200.8 | 199.6 | 200.2 | — |
| g-both | 217.4 | 218.8 | 218.1 | +8.9% |
| concat32+attn@2 (attention at layer 3) | 226.2 | 228.3 | 227.3 | +13.5% |
| grassmann+attn@5 | 244.4 | 244.4 | 244.4 | +22.1% |
| concat32+attn@5 | 247.3 | 248.2 | 247.7 | +23.7% |
| grassmann | 269.9 | 270.4 | 270.1 | +34.9% |
| concat32 | 280.2 | 280.8 | 280.5 | +40.1% |

Best epoch 5 for every run; every model still improving. Seed spread 0.1–2.1. grassmann (270.1), g-both (218.1) and
attention (200.2) reproduce the earlier notebooks.

### What it shows

- **Short Hyena beats full attention at this budget,** by 4% alone and by 8% with one attention layer, in both
  seeds. It sees only 16 tokens back. **This is the strongest model so far.** It is also the first result that
  bears on Hyena-audio (a separate project and log): a 16-step explicit Hyena filter with gating is a strong local
  mixer here. Caveat: 5 epochs, with every model still improving. Attention may catch up or overfit differently by
  30 epochs; the full protocol run includes hyena, swa and attention on the test set.
- **One attention layer at the end recovers only ~40%** of the gap for weak local mixers (grassmann 37%, concat32
  41%). **Placing it in the middle recovers more:** concat32 with attention at layer 3 is 227.3, against 247.7 at
  layer 6 (8% better).
- **g-both + one attention layer ≈ full attention** (198.4 vs 200.2).
- **At equal token compression (r = 32), Plücker beats concat by 3.8%** (grassmann 270.1 vs concat32 280.5). But
  concat32 hands the projection 64 numbers per pair against 496, and concat at r = 248 (496 numbers) matched
  grassmann in the pilot. So the product mainly gets many features out of a small r, not special geometry. Untested
  directly.

### Prediction check

**Mostly wrong.**
- "One attention layer recovers ≥ 60%": 37–41% at layer 6, 66% at layer 3, and 110% for g-both.
- "The local mixer type matters little inside a hybrid (≤ 3%)": hybrids range from 184 to 248.
- "Layer 3 vs 6 is a small difference": 8%.
- Hyena beating attention was not predicted.

---

## 2026-11-04: Full protocol — results, two independent runs (commit 2702c74)

The confirmatory experiment from `PROTOCOL.md`, run twice with the same code:

| run | hardware | precision | wall time | source |
|---|---|---|---|---|
| **A** | 1 × A100-SXM4-40GB (Colab) | bf16, compiled | not recorded (est. ~4 h from throughput) | `grassman_colab (2).ipynb` outputs only |
| **B** | 3 × NVIDIA B200 | bf16, compiled, torch 2.8 | 52 min (05:56–06:48), 2.4 GPU-hours | `grassman_results.zip` (`runs.jsonl`, logs) |

Both: 210 tests pass; LR sweep (3 LRs × 10 epochs, seed 0, validation only); then 3 seeds × 30 epochs per
model, test set at the best-validation epoch.

The LR sweep chose the same LRs in both runs except `outer` and `concat`: 3e-3 in A, 1e-3 in B. The sweep is one
seed, so near-ties between LRs can flip between machines.

### Main table (test perplexity, mean ± 95% interval over 3 seeds)

| model | A: test ppl | B: test ppl | vs attention (A / B) |
|---|---|---|---|
| grassmann (paper) | 267.1 ± 4.1 | 265.2 ± 2.3 | +47.8% / +46.8% |
| grassmann-raw (no L2 norm) | 270.3 ± 7.8 | 268.6 ± 10.0 | +49.6% / +48.7% |
| sym | 273.6 ± 3.8 | 273.4 ± 3.9 | +51.4% / +51.3% |
| outer | 270.4 ± 12.1 | 269.3 ± 2.8 | +49.7% / +49.1% |
| concat | **259.0 ± 4.4** | 263.5 ± 2.1 | +43.4% / +45.8% |
| concat32 | 265.0 ± 7.0 | 265.6 ± 8.0 | +46.7% / +47.0% |
| g-select | 204.1 ± 3.7 | 204.5 ± 0.9 | +13.0% / +13.2% |
| hyena (16-step) | 192.1 ± 1.5 | 192.9 ± 2.3 | +6.3% / +6.8% |
| **hyena+attn@5** | **174.8 ± 2.6** | **175.3 ± 1.8** | **−3.3% / −3.0%** |
| swa (window 16) | 192.5 ± 3.5 | 192.5 ± 2.2 | +6.6% / +6.5% |
| attention | 180.7 ± 3.7 | 180.7 ± 2.2 | — |

Apart from concat, every model agrees between the runs to within 1.6 perplexity.

Best epoch (run B, seed 0): the pair models peak at epochs 13–15 (concat32 at 20, g-select at 18), hyena at 13,
swa at 20, attention at 23.

Throughput on B200: grassmann 418k tok/s, hyena 511k, attention 585k. On A100: 252k, 296k, 338k. At 128 tokens,
attention is the fastest mixer.

### Confirmatory family (Holm-corrected; difference = other − grassmann, + means the paper's layer is better)

| other | A: diff [95% cluster CI], Holm p, wins | B: diff [95% cluster CI], Holm p, wins |
|---|---|---|
| sym | +6.55 [+4.77, +8.49], 0.0008, 3/3 | +8.18 [+6.22, +10.23], 0.0008, 3/3 |
| outer | +3.32 [+0.55, +5.50], 0.0208, 2/3 | +4.10 [+2.01, +6.08], 0.0008, 3/3 |
| concat | **−8.02 [−10.37, −6.01], 0.0008, 0/3** | **−1.72 [−3.31, +0.30], 0.0858, 0/3** |
| attention | −86.39 [−99.38, −73.61], 0.0008, 0/3 | −84.53 [−96.83, −72.09], 0.0008, 0/3 |

### Q1 (does the geometry matter?): **No.** The same verdict in both runs.

The pre-registered rule needed `sym`, `outer` **and** `concat` all to be significantly worse, with grassmann winning
every seed. Concatenation is never worse:
- In run A it is significantly better (−8.0).
- In run B it is better in all 3 seeds but not significantly (−1.7, p = 0.086).

The size of the concat advantage depends on the LR the sweep picked, so we report it as **0 to 8 perplexity in
concat's favour**.

What does hold, in both runs:
- **Order matters.** `sym` is worse by 6.6–8.2, in 3/3 seeds.
- **The antisymmetric form beats the full outer product slightly**, by 3.3–4.1.

But the plain ordered pair [z_t, z_{t−Δ}] does at least as well. **The Plücker coordinates add nothing over
concatenation.**

Exploratory:
- `concat32` vs grassmann: −2.1 (p = 0.07) in A, +0.3 (p = 0.69) in B, so no difference at equal r. This **does not
  confirm** the 5-epoch hybrid pilot, where Plücker beat concat32 by 3.8%.
- L2 normalisation of pair features helps slightly: grassmann-raw is worse by 3.3–3.4 in both runs.

### Q2 (gap to a transformer): **47–48%, against the paper's 10–15%** (and 22.6% in the earlier reproduction)

- Our grassmann (265–267) is close to the paper's 275.7.
- Our size-matched attention (180.7) is far better than the paper's transformer (241.0–253.6).

This supports the pilot reading that **the paper's transformer baseline was weak**. The gap is in the comparison,
not in the reproduction of their layer.

### Exploratory models (chosen after the pilots, so not claims)

- **Choosing closes ~72–73% of the gap.** g-select is at 204 vs grassmann 266 and attention 181. The 5-epoch pilot
  said 78%, so the result holds at 30 epochs.
- **Short Hyena does not beat attention at 30 epochs.** It is worse by 12.2 [9.6, 14.9], p = 0.0002, 3/3 seeds
  (run B, paired against attention). **This reverses the 5-epoch pilot** (192 vs 200). The pilot measured early
  learning speed. Hyena and swa (both 16 tokens back) end up tied (192.1–192.9 vs 192.5).
- **Hyena + one attention layer beats full attention.** By 5.4 perplexity [2.1, 9.6], p = 0.0004, 3/3 seeds
  (−6.2, −3.7, −6.3; run B, paired cluster bootstrap against attention). Run A gives the same mean gap (−5.9).
  This is the best model in both runs. Caveats:
  - it was added after seeing the pilots;
  - one dataset, at 128 tokens;
  - it is slower than attention at this length.
- **The gap is not reach.** At positions 0–15, every mixer sees the same context. Yet the pair models are still at
  301–310 there, against 264 for attention and 250–254 for hyena. They use the context they have worse. This
  matches the choose-vs-reach result.

### Prediction check

- **Right:**
  - Q1 not shown (the pilot had Plücker ≈ concat);
  - order matters;
  - the paper's transformer is weak;
  - choosing holds up at 30 epochs.
- **Wrong:**
  - "short Hyena beats attention" (pilot) — reversed;
  - "Plücker beats concat at equal r" (hybrid pilot) — not confirmed;
  - the ~9 h run-time estimate: run B took 52 min on 3 B200s, and run A is estimated at ~4 h.
