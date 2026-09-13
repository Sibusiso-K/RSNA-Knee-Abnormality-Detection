# Claude handover: next score-improvement experiments

Date: 2026-09-13. Prepared by Codex at the user's request. This is a strategy
handover, not a report of implemented changes or newly launched experiments.
Claude handles implementation; the user relays coordination with Codex.

## Verified anchor and corrections

- Kaggle submission 56165500, `knee-submit-fresh10-24ep-prob` version 1,
  completed with public score **0.891**, checked via Kaggle API on September 13.
- It uses five small and five base 24-epoch checkpoints, uniform raw-probability
  averaging. Its predecessor, uniform rank averaging, scored **0.890**.
- Deployment-aligned CUDA mean-fold OOF: probability **0.8584237803**, rank
  **0.8577873588**. Probability wins on all five folds.
- The 56.25% base candidate's apparent mean-fold gain is only about 0.000109.
  Leave-one-fold-out global-weight selection gives 0.8584298952: only about
  **0.000006** above fixed 50/50. Codex previously overstated this evidence.
  Treat the candidate as low priority, not a demonstrated improvement.
- Per-label nested weights previously underperformed fixed 50/50 probability
  averaging (0.858029 versus 0.858424). Do not resume unrestricted weight sweeps.
- `docs/00-state.md` contains old historical scores. Its 0.864 ceiling claim
  is superseded by the verified 0.891 anchor above.

## First: make checkpoint selection deployment-aligned

`notebooks/kaggle_06_train_slots.py::run_fold` saves only improvements in the
training backend's validation metric and overwrites the prior best checkpoint.
TPU validation and CUDA validation differ for the current architecture. It is
not yet known whether their epoch rankings differ. The observed score gap
does not establish that CUDA-selected checkpoints will improve the ensemble.

For the next controlled run, retain snapshots at a small, predefined set of
late epochs (for example 16, 20 and 24), plus the existing TPU best. Record
epoch, fold, seed, labels/cache identity and backend. Evaluate these on CUDA
and compare epoch rankings. Keep one unchanged 24-epoch control; do not change
the training backend, loss and schedule simultaneously. Do not try to recover
overwritten epochs from the existing best-only checkpoint files.

Deliver: CUDA/TPU epoch-ranking table, aligned OOF predictions, and a comparison
of checkpoint-selection policies. Epoch selection on a validation fold makes
its reported score selection-biased: verify the chosen policy on folds not
used to design it before expanding training or promoting a submission.

## Second: test the mismatch between loss and label uncertainty

The training loss is ordinary `BCEWithLogitsLoss()`. Validation drops cells
with `abs(target - 0.5) < 0.05`, but these cells still receive full training
weight. They pull logits toward zero even when the report did not commit to
presence or absence. This is a plausible bottleneck, not a proven bug.

Run a controlled two-fold pilot using the successful small/xattn/6-slice/
336px/24-epoch configuration and the exact existing labels and scanner folds.
Compare the full-weight control with undecided-cell weights 0.25 and 0.0.
Keep committed soft targets unchanged. Compute unreduced BCE, apply weights,
normalize each target by its valid weight sum, then average supported targets;
guard zero-support batches and nonfinite labels. Apply the same reduction to
both controls and candidates so normalization does not confound the result.

Do not reinterpret report silence as a negative label. Missingness is not
random, so masking can also harm generalization. Keep the validation mask
fixed and report per-target support and AUC, especially sparse targets.
Lock the winning policy before testing the remaining folds. Gold is a small
sanity check, not a hyperparameter-selection set.

## Third: add diversity from another strong 24-epoch seed

Train a second seed of a successful xattn family with identical labels,
preprocessing, schedule and scanner folds. Older 10-epoch seed ensembles do
not answer whether another equally strong 24-epoch family helps.

Assess incremental ensemble AUC alongside standalone AUC and prediction
correlation. For each OOF study, use only members that excluded its scanner
group during training. Never blend predictions from other folds' models that
trained on that study. Use a small prespecified family-weight comparison,
validated outside the folds used to choose it. Expand to five folds only
after the pilot supports additional value.

## Fourth: test modest inference augmentation

Using existing checkpoints, compare identity inference with identity plus
small translations within the training augmentation range. Keep laterality,
slot assignment and target semantics intact; avoid indiscriminate horizontal
flips. Specify averaging across slice groups, augmentations and model members
explicitly, then use exactly the same operations in OOF and submission.

Measure per-fold and ensemble gains plus runtime before promotion. The
visible three-study notebook run is only a smoke test, not evidence that
hidden-test inference will meet Kaggle's runtime limit.

## Fifth: evaluate EMA in a separate controlled run

Compare ordinary weights with exponential moving average weights under a
fixed schedule, recording the averaging start and decay. Keep a normal-weight
control. EMA may stabilize noisy epoch selection at unchanged inference cost;
benefit here is unproven. Average only compatible weights from the same
training trajectory, never different encoder widths or fold checkpoints.

PyTorch reference:
https://docs.pytorch.org/docs/main/generated/torch.optim.swa_utils.AveragedModel.html

## Acceptance protocol for every experiment

1. Reproduce the 0.8584237803 CUDA probability-blend baseline before comparison.
   The evaluator uses its local uncertainty-dropping `macro_auc`, not the
   differently defined helper in `src/model/validation.py`. A previous local
   analysis discrepancy was caused by the wrong label mask, not NPZ layout.
2. Freeze scanner-fold membership, study IDs, labels and uncertainty mask.
   Preserve raw OOF probabilities and provenance for every candidate.
3. Report standalone and ensemble changes, all five fold deltas when available,
   per-target AUC/support, decode failures, and runtime. Where scanner IDs are
   available, use paired scanner-group bootstrap intervals, not an independent
   row bootstrap. Tiny fitted gains alone do not justify promotion.
4. Separate exploratory folds from confirmation folds. Existing checkpoints
   were selected on validation folds; OOF is not an untouched test set.
   Avoid repeatedly using the gold set or public LB to tune parameters.
5. Keep 0.891 as the anchor. Submit only a validated artifact with finite,
   nonconstant predictions, exact IDs/columns and a complete checkpoint roster.
   A successful notebook execution is not itself a competition submission.

## Execution and coordination notes

- Explicitly request `--accelerator NvidiaTeslaT4` for CUDA Kaggle kernels.
  Prior implicit GPU assignments produced incompatible P100 runs.
- `SlotNet` exposes its backbone as `net.vit`, not `net.encoder`. The weighted
  blend builder was corrected in commit 378adb7 after a failed version 1.
  Do not submit that failed version's all-0.5 fallback CSV.
- Coordinate TPU/GPU use with the user/Codex before overlapping expensive runs.
- Do not refresh label datasets underneath active experiments; record versions.
- Prioritize checkpoint selection and uncertainty-weighted loss. Avoid adding
  weaker historical families or another unvalidated pseudo-label blend.
- Return a concise implementation summary, exact run/version IDs, measured
  results, limitations and the next decision for the user to relay to Codex.

The backend diagnostic establishes a reproducible backend-associated score
difference. The exact divergent operation remains unidentified; equal scalar
AUC on TPU is not proof of bitwise-identical full prediction arrays. Do not
describe the gap as a universal constant across architectures or epochs.
