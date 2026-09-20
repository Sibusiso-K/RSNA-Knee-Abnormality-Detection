# 18 — Experiment 3 pilot: does a second 24-epoch small seed help the deployed ensemble?

Date: 2026-09-20. Exploratory pilot on folds 0 and 1 only. Artifacts:
`kaggle/evaluate-seed-diversity/results/` (`summary.json`, `provenance.json`,
`predictions.npz`). Evaluator: `kaggle/evaluate-seed-diversity` v2 on a T4.

## What was compared

Baseline = the ORIGINAL deployed checkpoints behind the 0.891 submission
(`knee-slot-6slice-24ep-v1` small + `knee-slot-base-24ep-v1` base, best-on-TPU
files), uniform raw-probability mean, scored on CUDA. Not the retrained
seed-0 control from experiment 2.

New member = `train-seed2-pilot`, small/xattn/6-slice/336px/24-epoch, SEED=1,
default loss (`UNDECIDED_WEIGHT=1.0`). Predeclared arms: add it at 10% and
20%, proportionally shrinking the two existing members
(`(1-w)*0.5*(small+base) + w*seed1`).

Baseline reproduced exactly before anything was compared: fold 0
0.8668725809 and fold 1 0.8370606354, both matching evaluate-fresh24 v5
(assertion tolerance 1e-4).

## Result

| | fold 0 | fold 1 | mean |
|---|---|---|---|
| baseline (small+base) | 0.86687 | 0.83706 | 0.85197 |
| + seed 1 at 10% | 0.86695 (+0.00008) | 0.83847 (+0.00141) | 0.85271 (+0.00074) |
| + seed 1 at 20% | 0.86663 (−0.00025) | 0.83907 (+0.00201) | 0.85285 (+0.00088) |

Paired scanner-fingerprint-group bootstrap (500 resamples, within fold,
same resample for baseline and candidate; fold 0 has only 29 scanner groups):

| delta vs baseline | fold 0 95% CI | fold 1 95% CI | mean-of-folds 95% CI | boot P(mean>0) |
|---|---|---|---|---|
| 10% | [−0.00082, +0.00098] | [−0.00041, +0.00216] | [−0.00026, +0.00123] | 0.924 |
| 20% | [−0.00218, +0.00119] | [−0.00114, +0.00327] | [−0.00096, +0.00172] | 0.792 |

Every interval includes zero.

Standalone CUDA AUC of the new seed: fold 0 0.8533 (small 0.8554, base 0.8631),
fold 1 0.8281 (small 0.8251, base 0.8321).

Per-target mean change (folds 0/1 averaged) at 10%: Fracture +0.0022,
Synovitis +0.0013, ACL +0.0012, MCL +0.0012, Effusion +0.0009, others
+0.0002–0.0006, Contusion −0.0003. At 20%: Fracture +0.0030, MCL +0.0026,
ACL +0.0020, Synovitis +0.0015, Contusion −0.0009. Fold 1 improves on
12/12 targets at both weights; fold 0 is mixed (6/12 targets negative at
10%, 5/12 at 20%). Full per-fold table: `summary.json`.

Correlation (supporting evidence only), mean per-target Pearson:
seed1–small 0.947 / 0.917 (folds 0/1) vs small–base 0.909 / 0.891. The new
seed is more redundant with the small model than base is, so little
diversity was expected.

## Recommendation: reject both; do not spend confirmation training

- Mean gains of +0.0007 / +0.0009 are roughly a quarter of the +0.003
  (≥4/5 non-negative folds) bar used for blend candidates in this project,
  and no interval excludes zero.
- The gain is carried by fold 1. Fold 0 is flat at 10% and negative at 20%.
- Two weights were tried; the better one on two exploratory folds is not a
  confirmed gain. If one had to be frozen it would be 10% (non-negative on
  both folds, smaller downside), but that choice is made on these same folds.
- This is a per-fold 3-member vs 2-member comparison. The deployed ensemble
  already averages 5 fold-models per family, which absorbs seed variance, so
  a second seed's marginal value in deployment is likely smaller still.
- Confirming on folds 2–4 would cost roughly 4 h of TPU (estimated from the
  earlier two-fold legs) plus a submission slot, for an expected gain under
  0.001 AUC.

Next: docs/15 step 4 (modest translation augmentation at inference on the
existing checkpoints), which needs no training.
