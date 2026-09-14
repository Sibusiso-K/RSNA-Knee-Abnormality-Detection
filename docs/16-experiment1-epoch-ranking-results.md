# 16 — Experiment 1 results: does CUDA agree with TPU's checkpoint selection?

Date: 2026-09-14. Result of docs/15-claude-score-improvement-strategy.md,
experiment 1. Raw output: `kaggle/evaluate-epoch-ranking/results/`.

## Run provenance

- `knee-train-6slice-24ep-snap`: small/xattn/6-slice/336px, 24 epochs, 5
  folds, SEED=0. Identical backend/loss/schedule to the existing
  `train-6slice-24ep` control — only the checkpoint-saving is new (snapshots
  at epochs 16/20/24 alongside the existing best-on-TPU file).
- `knee-evaluate-epoch-ranking`: scores all 20 checkpoints (5 folds × 4
  variants) on CUDA. Asserts a complete roster and matching `seed` before
  scoring anything.

## Result: TPU and CUDA agree on 3 of 5 folds

| fold | TPU-best epoch | best CUDA score | e16 CUDA | e20 CUDA | e24 CUDA | agrees | gain if switched |
|---|---|---|---|---|---|---|---|
| 0 | 21 | 0.8546 | 0.8450 | 0.8533 | 0.8544 | yes | 0 |
| 1 | 17 | 0.8241 | 0.8201 | 0.8211 | 0.8215 | yes | 0 |
| 2 | 15 | 0.8507 | 0.8507 | 0.8610 | **0.8631** | **no** | **+0.0124** |
| 3 | 19 | 0.8488 | 0.8389 | 0.8488 | 0.8475 | yes | 0 |
| 4 | 19 | 0.8600 | 0.8522 | 0.8600 | 0.8605 | no | +0.0005 |

The two disagreements are not equally meaningful:

- **Fold 2 is a genuine rank reversal, not a magnitude shift.** TPU's own
  recorded score *declines* monotonically as training continues past epoch
  15 (0.8293 → 0.8277 → 0.8274 at epochs 15/19/23), while CUDA's score on
  the SAME checkpoints *rises* monotonically (0.8507 → 0.8610 → 0.8631). The
  two backends don't just disagree by an offset here — they disagree about
  which direction the model is moving.
- **Fold 4's gain (+0.0005) is noise-level** — it technically flips the
  argmax but is not a meaningful disagreement in practice.

## What this does and does not establish

Does: confirms the TPU/CUDA backend delta (docs/12, docs/15) is not just a
constant score offset — it can change which epoch looks best, concretely in
one fold out of five.

Does not: this is one run, one seed, and the epoch selection is measured on
the same folds used to train it (selection bias, same caveat as any other
checkpoint here). It does not by itself justify switching the project's
checkpoint-selection policy from "best on the training backend" to "best on
CUDA" — that would need confirming on data not used to find this pattern,
per docs/15's acceptance protocol (separate exploratory from confirmation
folds).

## Recommendation

Not conclusive enough on its own to change checkpoint-selection policy or
promote a submission. Two ways to firm this up if it's worth the TPU/GPU
budget: (a) a second seed of the same recipe to see whether fold-2-style
reversals recur or were a one-off, or (b) simply note this as a known,
measured risk and move to experiment 2 (uncertainty-weighted loss), which
was next in priority regardless of this result.
