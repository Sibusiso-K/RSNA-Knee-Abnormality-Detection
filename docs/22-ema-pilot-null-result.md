# 22 — EMA pilot: null result (negative under CUDA/TTA), no submission

Date: 2026-09-22/23. docs/20 job 2, the last planned experiment in that
handover. New code: `src/model/ema.py` (9 tests), per-fold seeding wired
into `notebooks/kaggle_06_train_slots.py` behind `EMA_DECAY`/`PER_FOLD_SEED`
(both off by default; every other kernel unaffected). Training:
`kaggle/train-ema-pilot`, TPU v5e-8, folds 0/1, small/xattn/6-slice/336px,
24 epochs, EMA decay 0.999 fixed and predeclared, `PER_FOLD_SEED=1`. Both
ordinary and EMA weights saved at the same predefined final epoch (24).
Evaluator: `kaggle/evaluate-ema-pilot`, CUDA T4, exact production five-view
TTA. Artifacts: `kaggle/evaluate-ema-pilot/results/summary.json`,
`predictions.npz`.

## Training-time signal (TPU validation, for context only)

At save time (epoch 24, 10,416 EMA updates per fold):

| fold | ordinary | EMA | Δ |
|---|---|---|---|
| 0 | 0.8243 | 0.8249 | +0.0006 |
| 1 | 0.8020 | 0.8028 | +0.0008 |

Both folds positive on the backend that trained them - the same pattern
docs/12/docs/15/docs/16 already document as unreliable on its own (TPU and
CUDA validation disagree on this architecture).

## CUDA/TTA result: negative on both folds, both comparisons

**1. Standalone (small family alone, TTA, ordinary vs EMA):**

| fold | ordinary | EMA | Δ | 95% CI |
|---|---|---|---|---|
| 0 | 0.858707 | 0.858202 | −0.000505 | [−0.00187, +0.00015] |
| 1 | 0.825825 | 0.825547 | −0.000278 | [−0.00130, +0.00082] |

Mean Δ **−0.000392**, CI [−0.00127, +0.00031], bootstrap P(Δ>0) = 0.134,
0/2 folds non-negative.

**2. Ensemble (EMA swapped for the small member in the deployed 0.5 small +
0.5 base blend):**

| fold | ordinary+base | EMA+base | Δ | 95% CI |
|---|---|---|---|---|
| 0 | 0.867605 | 0.867187 | −0.000418 | [−0.00135, −0.00006] |
| 1 | 0.838506 | 0.838309 | −0.000197 | [−0.00085, +0.00040] |

Mean Δ **−0.000308**, CI [−0.00082, +0.00007], bootstrap P(Δ>0) = 0.05,
0/2 folds non-negative. Fold 0's interval excludes zero - on the negative
side.

EMA loses under CUDA/TTA on every fold and every comparison run. The sign
flips relative to the TPU validation numbers above, consistent with this
project's established TPU/CUDA disagreement rather than with EMA helping.

## What this does and does not establish

Does: with one predeclared decay (0.999) and a two-fold pilot, EMA does not
help this architecture/recipe under the metric and backend the submission
actually uses - it costs slightly, on every fold checked.

Does not: rule out EMA at a different decay, or over a different training
length. This pilot exists specifically so that question would not need a
five-fold confirmation run if the two-fold signal were negative or flat.

## Bug check inherited from job 1

Reused the same per-batch view-cache pattern `evaluate-snap-tta` needed
fixed after its CUDA OOM (docs/21) - built it correctly this time
(cache created fresh per batch, dropped before the next). This run completed
without incident on the first push.

## Decision

Per docs/20's own rule ("Expand to the remaining folds only if the pilot
supports an ensemble benefit"): it does not. Stop this branch. No submission.
Both docs/20 jobs are now closed with null results; the 0.893 TTA submission
(56441791) remains the anchor. No further action queued from docs/20 - next
steps need a new decision from Codex/the user.
