# 25 — Seed-average submission (fresh15) and the fp32 precision diagnostic

Date: 2026-10-12.

## Candidate

`0.25 * TTA(S) + 0.25 * TTA(E24) + 0.5 * TTA(B)`. S is the deployed small/xattn
24-epoch family, B the deployed base family, and E24 the epoch-24 checkpoints of
the independent snapshot run (`knee-slot-6slice-24ep-snap-v1`, same recipe,
different training run). The 50/50 small/base family split is unchanged; E24 is
added alongside S rather than replacing it (docs/21 tested replacement and
rejected it).

## Evidence (offline, saved TTA predictions)

`scripts/eval_small_seed_average.py` on `kaggle/evaluate-snap-tta/results/predictions.npz`,
output `kaggle/evaluate-snap-tta/results/small_seed_average.json`. The anchor
reproduces 0.8599910527 exactly.

| | mean-fold OOF AUC |
|---|---|
| anchor 0.5·S + 0.5·B (LB 0.893) | 0.859991 |
| candidate | 0.861336 |
| Δ | +0.001345, 95% CI [+0.00066, +0.00269], 5/5 folds positive, bootstrap P(Δ>0) = 1.0 (1,000 scanner-group resamples) |

Per fold Δ: +0.0005, +0.0012, +0.0026, +0.0009, +0.0016. 11 of 12 targets
improve (MCL +0.0034, Baker's +0.0042, PF OA +0.0030); Fracture falls by 0.0018.

**Status: post hoc.** These folds and checkpoints were inspected in docs/21. The
blend was fixed before computing and was the only variant tried, but the result
is exploratory, not fresh validation.

## Submission notebook

`kaggle/submit-fresh15-seedavg-tta/build.py` chains onto the 0.893 TTA builder
and changes only the roster (15 members, strict `(tag, fold)` check, snapshot
dataset filtered to `_snap_e24.pth`) and the family weights (0.05 / 0.05 / 0.10
per member).

- Rehearsal, 192 training studies (`results/rehearsal192_run_stats.json`): all 15
  members loaded; E24 parity against the snapshot evaluator 0.0 (folds 0, 1);
  S/B parity ≤ 6e-8 (TTA and identity); evaluator-formula check 6e-8; no problems.
  Forward cost 5.06 s/study for 15 members.
- Projected hidden-test runtime: about 2.8 h for ~1,300 studies (forward 5.06
  s/study plus decode 2.57 s/study from the 1,400-study rehearsal). Cap 9 h.
- Public pass v1 (`results/public_run_stats_v1.json`): 15 members, 3/3 scored,
  IDs and columns exact, values finite in [0, 1], no problems.
- **Submitted:** `knee-submit-fresh15-seedavg-tta` v1 → submission 57116420.
  Score: pending.

The 0.893 submission (56441791) remains available as the fallback.

## fp32 precision diagnostic (docs/24 follow-up)

`knee-train-small-cuda-fp32-f0` trained fold 0 with `CUDA_AMP=0` (no autocast,
no GradScaler). On-device validation at epoch 24 was 0.8342, versus 0.8366 for
the fp16 CUDA run. Epochs took about 756 s each (the 50-step probe implied ~220 s),
so treat probe projections for fp32 as optimistic. The TTA comparison against
the fp16 CUDA and TPU controls is running in `kaggle/evaluate-cuda-fp32-pilot`.
