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
