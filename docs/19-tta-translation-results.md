# 19 — Translation test-time augmentation on the deployed checkpoints

Date: 2026-09-20. docs/15 step 4. Artifacts: `kaggle/evaluate-tta/results/`
(`summary.json`, `runtime.json`, `provenance.json`, `predictions.npz` with
aligned ids, scanner groups, targets, identity probabilities, per-view logits
and TTA probabilities for every member). Evaluator `kaggle/evaluate-tta` v1,
T4, all five folds, each scored only by its own held-out checkpoint.

## Design (predeclared, evaluated once, nothing tuned on these folds)

- Views: identity + four axis shifts of ±0.03 (`affine_grid` units, ~5 px at
  336 px, 60% of the ±0.05 training range). No flips, rotation, zoom or
  intensity change; every image of a study shifted identically (slot
  assignment and slice order untouched). `src/model/tta.py`, 7 tests.
- Weights fixed at 0.2 each. Order: group-average logits → TTA-average
  logits → sigmoid → probability-average across members (the deployed
  uniform probability blend).
- Uncertainty mask and `macro_auc` unchanged.

## Reproduction (hard assertions; the run aborts otherwise)

All ten identity-only checkpoint scores match evaluate-fresh24 v5 within
1e-4; the deployed blend mean is 0.858423780326844, exactly the anchor; the
identity view equals `predict()` to 6e-8 (max over all ten).

## AUC

| | identity | TTA | Δ | 95% CI (paired scanner-group bootstrap, 500) | folds ≥ 0 |
|---|---|---|---|---|---|
| **blend (deployed)** | 0.858424 | 0.859991 | **+0.001567** | [+0.000924, +0.002030] | 5/5 |
| small | 0.848680 | 0.851421 | +0.002741 | [+0.001733, +0.003501] | 5/5 |
| base | 0.853260 | 0.855263 | +0.002003 | [+0.001183, +0.002711] | 5/5 |

Blend, per fold (Δ, CI): f0 +0.000842 [−0.000562, +0.001515]; f1 +0.002931
[+0.000937, +0.004200]; f2 +0.000719 [−0.000418, +0.001199]; f3 +0.001832
[+0.000445, +0.003016]; f4 +0.001512 [+0.000304, +0.002882]. The interval
excludes zero on folds 1, 3, 4 and overall; folds 0 and 2 are positive
points whose intervals include zero. (~30 scanner groups per fold.)

Both families improve on all five folds; small gains more. Single views are
each within about ±0.002 of identity (no outlier view), so the gain comes
from averaging, not from any one shift.

## Per-target (blend, mean over folds)

ACL +0.0030, MCL +0.0029, Contusion +0.0024, Fracture +0.0024, Medial
Meniscus +0.0017, Lateral Meniscus +0.0016, Medial OA +0.0013, Lateral OA
+0.0011, Baker's +0.0011, PF OA +0.0009, Synovitis +0.0003, Effusion
+0.0001. Effusion and Synovitis are flat (negative on 3/5 folds each); the
rest are negative on at most 1 fold. Lateral Meniscus gain is entirely the
small family (base 0.0000). Full tables in `summary.json`.

## Runtime (T4, measured)

| | identity | five-view | ratio |
|---|---|---|---|
| OOF eval, all 10 checkpoints × ~870 studies | 1133 s (18.9 min) | 3964 s (66.1 min) | 3.5× |
| small, s/study | 0.131 | 0.257 | 2.0× |
| base, s/study | 0.130 | 0.655 | 5.04× |

Base is compute-bound (a clean 5×); small was partly I/O-bound in this
evaluation (mmap cache reads), so its ratio understates the compute cost.
The robust quantity is the added forward cost per extra view: 0.0315 s/study
(small), 0.131 s/study (base).

**Estimated hidden-test submission** (10 members × ~1,300 studies, GPU):
added forward time ≈ 4,230 s ≈ **71 min** (small 14 min + base 57 min).
Forward-only inference goes from roughly 18 min to roughly 88 min. The
existing decode/cache-build stage is unchanged and dominant: kaggle_07's
own note estimates ~38 min for 1,300 studies at 0.57 study/s (an older
3-slice figure; the 6-slice multi-grid path may be up to ~2× that). Total
run ≈ 2.2–3 h against the 9 h cap, i.e. about a third of the limit, and
still under 6 h if these compute estimates were off by 2×. These are
extrapolations: the public notebook run has 3 studies and says nothing about
hidden-set runtime.

## Limitations

- OOF here is one model per family per fold. The deployed ensemble averages
  five fold models per family, which already removes some of the variance
  TTA averages away, so the deployed gain is plausibly smaller than +0.0016.
  Unmeasured.
- Every checkpoint was trained and epoch-selected on the fold it is scored
  on; shift set chosen a priori, not fitted.
- Reference scale: the probability-vs-rank blend gained +0.000636 OOF and
  moved the public score 0.890 → 0.891.

## Recommendation

The evidence supports building a TTA submission candidate: gains on both
families and all five folds, overall interval excluding zero, no per-target
regression beyond noise, runtime fits with a wide margin. Before submitting:

1. Rehearse runtime: run the real submission inference path (10 members,
   five views) forward-only over ≥1,300 cached studies and record wall time.
2. Implement TTA in the submission via `src.model.tta` only, then verify the
   identity-only path reproduces the existing 0.891 predictions exactly and
   that TTA predictions are finite, non-constant, with exact IDs/columns and
   the full 10-checkpoint roster.
3. Submit as a new candidate; keep 0.891 as the anchor.

The reduced-view fallback (score fewer views offline from the saved
per-view logits, no rerun) is not needed at this runtime.
