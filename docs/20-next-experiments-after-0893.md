# Next experiments after the 0.893 TTA submission

2026-09-22. Codex handover to Claude. User objective: exceed 0.900 public
score. This objective is not a forecast: the remaining gain is unproven.
Claude implements and runs experiments; Codex reviews the evidence.

## New anchor

- Submission 56441791, `knee-submit-fresh10-24ep-prob-tta` version 1:
  COMPLETE, public score **0.893**, checked via Kaggle API September 22.
- Original ten 24-epoch checkpoints, 50/50 small/base probability mean,
  five fixed translation views. CUDA mean-fold OOF **0.8599910526581681**.
- Reference predictions: `kaggle/evaluate-tta/results/predictions.npz`.
- The previous identity-only OOF 0.8584237803 and LB 0.891 are historical
  controls, not the promotion baseline for the next experiment.
- Preserve the successful notebook version and its dependency provenance.

## First job: evaluate existing late snapshots with TTA, no training

Evidence: experiment 1 found a CUDA epoch-ranking reversal on fold 2:
the snapshot run's TPU-best checkpoint scored 0.850726, while its e24
checkpoint scored 0.863115. Other folds did not show a comparable gain.
Those are standalone scores without TTA, from a different training run
than the deployed small family. They do not establish a deployment gain.
The snapshot family must be evaluated as a whole, not cherry-picked by fold.

Use existing `knee-slot-6slice-24ep-snap-v1` checkpoints. Freeze two candidates
before running. For a study in fold f, define probabilities after the exact
production five-view TTA and group averaging:

- S = original deployed small fold-f member.
- B = original deployed base fold-f member.
- E20, E24 = snapshot-run small fold-f checkpoints at human epochs 20,24.
- Anchor = 0.5*S + 0.5*B.
- Candidate A = 0.5*E24 + 0.5*B.
- Candidate B = 0.25*E20 + 0.25*E24 + 0.5*B.

Candidate A tests a fixed final-epoch replacement. Candidate B tests
averaging late-epoch predictions, not model weights. Neither changes base
family weight or the translation set. Reuse saved baseline probabilities
after checking IDs, labels, mask, folds and inference provenance. Run one
same-input parity check before relying on reused artifacts.

Use all five folds and only each study's held-out checkpoints. Save raw
per-member probabilities, IDs, targets, scanner groups, checkpoint hashes,
source versions, per-target AUC and paired scanner-group bootstrap deltas.
Compare the two candidates against the new TTA anchor, including incremental
runtime and peak memory. Candidate B adds five small models at deployment;
load/process them in a way that stays within GPU memory and decode once.

These epochs were already inspected in experiment 1. Treat the result as
exploratory evidence, not a fresh untouched validation set. Do not fit
per-fold epochs, per-target weights, translation shifts or extra variants
after seeing the outcome. Positive evidence may justify one controlled
submission after production parity and runtime validation, but does not
guarantee a leaderboard improvement. If both fail, stop this branch.

## Second job: controlled EMA pilot if snapshots do not help

This is the remaining untested experiment from docs/15. Use a two-fold
small/xattn/6-slice/336px/24-epoch pilot with the existing labels and full
uncertainty-cell weight. Keep optimizer, augmentation and learning-rate
schedule fixed. Specify one EMA configuration before launch; do not search
decay values in this pilot.

Maintain EMA alongside ordinary weights in the same run. Initialize it from
the current model at the declared start; update after actual optimizer steps,
with explicit parameter/buffer handling. Save ordinary and EMA weights at
the same predefined final epoch, with decay, update count, fold, seed,
source/label/cache identity and configuration recorded. Verify that EMA
does not mutate ordinary model weights and remains compatible with XLA.

Evaluate both on CUDA with the exact successful TTA. First compare paired
ordinary versus EMA weights; then compare replacing the small family in
the original TTA ensemble. The ordinary trajectory is the experimental
control; the deployed TTA ensemble remains the score-improvement anchor.
Expand to the remaining folds only if the pilot supports an ensemble benefit.

Use a recorded per-fold seed for new training so an isolated fold retry
does not depend on whether earlier folds ran. This changes the seed scheme
from older runs: do not claim byte-identical reproduction of their training.
Check GPU/TPU availability before launch; do not overlap another worker's run.

## Decision and submission discipline

- No fixed +0.003 gate. Earlier +0.000636 OOF led to +0.001 public score;
  the TTA +0.001567 OOF led to +0.002 public score. These two observations
  do not define a reliable conversion from OOF gain to LB gain.
- Prefer consistent ensemble gains, paired uncertainty estimates, acceptable
  per-target effects and practical runtime. Intervals are conditional on the
  current noisy labels and reused validation folds, not private-LB guarantees.
- Do not resume the weak seed blends or uncertainty-weight sweeps without
  new evidence. Do not blindly increase TTA views or sweep public scores.
- Preserve the shared TTA cache-copy and autocast fixes. Mutating cached
  float images inside `SlotNet.forward` previously invalidated predictions.
- Validate actual submission inference against saved held-out predictions.
  Runtime estimates must include decode and setup; measure the corrected
  production path, not only the three-study public smoke test.
- If a candidate qualifies, publish a separate notebook, validate its output,
  submit the exact successful version, and report ID/status/score. Retain
  the 0.893 submission. Record null results as well as wins.

Immediate deliverable from Claude: the two snapshot candidate comparisons
against 0.859991 OOF, with artifacts, runtime and a promote/stop decision.
No new training is required to answer that first question.
