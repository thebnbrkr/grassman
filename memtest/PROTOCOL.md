# Memory tests: protocol (fixed before any run, 2026-11-04)

Two questions, one notebook each. Nothing here has been run yet.

## Track A: is the associative matrix a useful store? (`notebooks/store_test.ipynb`)

Store N items (key vector → item id), look them up by a query, and rank ids. Every compressed method gets
**the same byte budget** as the matrix (dk × dv × 4 bytes). Product quantisation (PQ) codebooks count towards
the budget.

| test | data | what varies |
|---|---|---|
| A1 capacity | random unit keys; queries at cosine 1.0 / 0.9 / 0.7 to their key | d = 64, 256, 1024; N/d from 1/8 to 8; 5 seeds |
| A2 real | Quora duplicate questions (key = question, query = its paraphrase), MiniLM-L6 (384-d) | N = 96 … 6144; budgets 98 KB / 590 KB / 2.4 MB; 5 seeds |
| A3 updates | A2 test split, 590 KB | overwrite or delete 25% / 100% of items; float drift over 20,000 deletes and re-writes |
| A5 cost | random keys, d = 384 | N = 10³ … 3×10⁵; CPU timings |

Methods:
- **matrix**: `add`, `delta` (1 or 3 passes) and `lstsq` (a non-streaming upper bound); `+c` means keys centred
  on a calibration mean;
- **pq**: at the same budget;
- **fifo**: the most recent items that fit, in fp16;
- **flat** and **hnsw**: no budget, as ceilings.

**Selection:** the best streaming matrix variant is chosen on the **dev** split (mean recall@1 at 590 KB over all N).
The test split is used once.

**Confirmatory family** (Holm, α = 0.05): chosen variant − PQ at 590 KB, recall@1 on test, at N = 192, 384, 768, 1536.
Paired bootstrap over queries pooled across the 5 seeds (10,000 resamples).

**Decision rules:**
- **Dropped** if the matrix is worse than PQ at N = 384 and N = 768 (Holm p < 0.05). Then it is not a store, and
  the time-series test (A4) is not built.
- **"Useful memory tier"** only if its recall@1 is within 2 points of PQ, or better, at every N ≥ 384.

**Predictions:**
- A1: `add` falls below 90% recall@1 well before N = d with noisy queries. `delta` and `lstsq` hold up to about
  N ≈ d, then all matrix variants collapse.
- A2: the matrix loses to PQ at N ≥ 384, by more than 10 points. Correlated real embeddings hurt it more than
  random keys do. Centring helps the matrix.
- A2: at the smallest budget (98 KB), 8-bit PQ codebooks barely fit, so the matrix may win at small N.
- A3: `add` deletes exactly; `delta` overwrites without knowing the old value, with some stale answers.

## Track B: does a memory layer give Hyena long recall? (`notebooks/recall_test.ipynb`)

4-layer models, d = 128, all the same block (pre-LN, mixer + MLP), trained on fresh synthetic data. The memory
layer sits at layer 3.

| model | layers | memory kept per sequence (inference) |
|---|---|---|
| hyena | HHHH | input history of the long convolution |
| hyena+attn | HHAH | + attention cache (2dT) in one layer |
| hyena+linear | HHLH | + one fixed matrix per head |
| hyena+delta | HHDH | + one fixed matrix per head (delta rule) |
| hyena+moment | HHMH | + 4 summary slots per 64-token chunk |
| attention | AAAA | 2dT per layer |
| swa | WWWW | window 128 |
| hyena+attn@0, @3 | AHHH, HHHA | exploratory: placement |

Tasks:
- **far** (recall at distance): 8 pairs scattered in noise, queried at the end, T = 1024 and 4096.
- **mqar** (Zoology): 64 pairs, T = 512.

Learning rate: 3e-4 / 1e-3 / 3e-3, chosen per model on **dev** accuracy (`far`, T = 1024, seed 0). The same LR is
used for every task. Then 3 seeds per model and task. Fixed dev and test sets are shared by all models.

**Primary metric:** test accuracy on `far`, T = 4096, for queries whose key is ≥ 1024 tokens back.

**Confirmatory family** (Holm, α = 0.05), difference in accuracy points, paired cluster bootstrap over test sequences:
1. hyena+moment − hyena (does the moment memory add long recall?)
2. hyena+moment − hyena+attn (does it match the attention layer?)
3. hyena+delta − hyena+attn (does a fixed-size delta memory match it?)

**Decision rules:**
- "Moment memory adds recall": comparison 1 is positive with Holm p < 0.05 and positive in 3/3 seeds.
- "Matches attention": the 95% interval of comparison 2 lies within ±5 points.

**Predictions:**
- attention and hyena+attn: ≥ 95% on `far` at all distances.
- swa: near chance (1/3584) beyond its reach.
- hyena: below 60% far accuracy at T = 4096.
- hyena+moment: within 10 points of hyena+attn on `far`, but **far below it on `mqar`**, by more than 30 points
  (64 dense pairs do not fit 4 slots per chunk).
- hyena+delta beats hyena+linear on both tasks.

**Known limits:** synthetic tasks only; one model size; the Hyena state counts the full input history
(no distillation into a recurrence); step counts are a guess, checked by the pilot cell before the main runs.
