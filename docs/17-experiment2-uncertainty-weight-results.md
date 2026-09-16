# 17 — Experiment 2 results: uncertainty-weighted loss pilot

Date: 2026-09-16. Result of docs/15-claude-score-improvement-strategy.md,
experiment 2. Raw output: `kaggle/evaluate-undecided-weight/results/`.

## Run provenance

Two-fold pilot (folds 0,1 only), small/xattn/6-slice/336px/24-epoch,
SEED=0, `uncertainty_weighted_bce` (src/model/loss.py) in place of the
plain `BCEWithLogitsLoss()` for all three legs — the control (weight 1.0)
goes through the identical reduction code as the candidates, proven
numerically equivalent to the old loss by `tests/test_loss.py`.

- `train-uw-control` — `UNDECIDED_WEIGHT=1.0`
- `train-uw-025` — `UNDECIDED_WEIGHT=0.25`
- `train-uw-000` — `UNDECIDED_WEIGHT=0.0` (full mask)

Sanity check: the control leg's TPU-recorded scores for folds 0/1
(0.8241/0.8087, same epochs) are byte-identical to the earlier
`train-6slice-24ep-snap` run for the same two folds — confirms `SEED=0`
makes training genuinely reproducible run-to-run, beyond this experiment's
own scope.

`knee-evaluate-undecided-weight` scored every checkpoint from all three legs
(best + snapshot epochs 16/20/24) on CUDA — per experiment 1, TPU and CUDA
validation can disagree, so comparing the three weight settings needs a
shared backend rather than each leg's own TPU-recorded number.

## Result: no candidate wins consistently — a clean null

Deployment-relevant comparison (each leg's own best-on-TPU checkpoint,
scored on CUDA):

| fold | control | w025 | w025 Δ | w000 | w000 Δ |
|---|---|---|---|---|---|
| 0 | 0.8546 | 0.8534 | −0.0012 | 0.8551 | +0.0006 |
| 1 | 0.8241 | 0.8237 | −0.0004 | 0.8186 | **−0.0055** |

- **w025 (0.25 weight):** mildly negative on both folds, never positive.
  Not a win.
- **w000 (full mask):** roughly flat on fold 0, but a real loss on fold 1
  (−0.0055, well above what the noise on the other three cells looks like).
  Consistent with docs/15's own warning: missingness in these reports is
  not random, so fully masking those cells can teach the model to ignore
  an entire mode of the data rather than just remove noise.

At the end of the fixed schedule (e24, avoiding each leg's own possibly
noisy "best" pick) the picture is similar: control 0.8380 mean, w025 0.8389
(+0.0009), w000 0.8393 (+0.0014) — differences an order of magnitude below
anything this project has treated as a real signal, and not consistent in
direction per fold either.

## Decision

**Per docs/15's acceptance protocol ("tiny fitted gains alone do not
justify promotion"): no candidate promoted.** Do not resume this loss
weighting on the remaining folds — the pilot's own purpose was to filter
before spending that budget, and neither candidate cleared the bar.

Loss/label-uncertainty mismatch was a plausible bottleneck, not a proven
one, going in — this measures it and finds it is not the bottleneck at this
weight, on this architecture, at least not in the direction tested (down-
weighting toward masking). It does not rule out a different treatment of
uncertain labels (e.g. label smoothing in the other direction, or a
loss that treats uncertainty as a target rather than a weight) — those are
different experiments, not variations of this one.

Moving to experiment 3 (a second 24-epoch seed for genuine diversity) per
docs/15's priority order.
