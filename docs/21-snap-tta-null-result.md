# 21 — Snapshot-epoch TTA candidates: null result, no submission

Date: 2026-09-22. docs/20 job 1. Evaluator: `kaggle/evaluate-snap-tta`, CUDA
T4, all five folds, exact production five-view TTA (`src.model.tta.tta_probs`).
Artifacts: `kaggle/evaluate-snap-tta/results/` (`summary.json`, `runtime.json`,
`predictions.npz`). No retraining; existing `knee-slot-6slice-24ep-snap-v1`
checkpoints only.

## Reproduction

Anchor (0.5 × TTA(deployed small) + 0.5 × TTA(deployed base), per fold) mean
AUC reproduced exactly: 0.8599910526581681, matching the 0.893 submission's
own evaluator to machine precision.

## Result: neither candidate clears the bar

| | mean anchor | mean candidate | mean Δ | 95% CI | boot P(Δ>0) | folds ≥ 0 |
|---|---|---|---|---|---|---|
| A (0.5·E24 + 0.5·B) | 0.859991 | 0.860320 | +0.000329 | [−0.00084, +0.00276] | 0.824 | 2/5 |
| B (0.25·E20 + 0.25·E24 + 0.5·B) | 0.859991 | 0.860198 | +0.000207 | [−0.00092, +0.00247] | 0.772 | 2/5 |

Both candidates' entire mean gain is carried by fold 2 (A: +0.00327, B:
+0.00275, the same fold experiment 1 flagged as a genuine TPU/CUDA epoch
disagreement). Folds 0, 1 and 3 are flat-to-negative for both candidates;
fold 4 is small and positive. Every per-fold and the mean bootstrap interval
includes zero. Candidate B (the extra E20 average) does not outperform the
simpler candidate A - averaging in a second late epoch adds cost without
adding signal here.

## Runtime (T4, measured, informational only since not promoted)

Small-family members (S, E20, E24) run at ~0.22 s/study/member under TTA;
base (B) at ~0.55 s/study/member - consistent with docs/19's family-level
timing. No incremental runtime concern either way.

## Bug found and fixed during this run (not a result, but recorded)

The first two pushes OOM'd 80s in: the per-fold view-translation cache was
keyed by batch start position and never cleared, so translated float32 views
from every batch in the fold (~109 of them) stayed resident at once instead
of being freed after the four members sharing that batch used them. Fixed by
building the cache fresh per batch and discarding it before the next batch
(`kaggle/evaluate-snap-tta/compare.py`). This is local to this evaluator's
multi-member-per-batch sharing pattern; the submission notebook's own
per-batch `view_caches` (cleared every batch already) was not affected.

## Decision: stop this branch, per docs/20's pre-registered rule

"If both fail, stop this branch" — no per-fold cherry-picking, no new
candidates from this snapshot family, no submission. Move to docs/20 job 2:
the controlled EMA pilot.
