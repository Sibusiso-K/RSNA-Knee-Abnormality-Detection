# 14 — TPU/XLA inference does not reproduce on GPU/CUDA (root cause, confirmed)

## The finding

`knee-diagnose-tpu` reran the exact fold-0-small check from `evaluate-fresh24`
on TPU instead of GPU, through the SAME unmodified `predict()` function used
during training's own per-epoch validation. Result:

```
score:          0.8205603644292246
recorded_score: 0.8205603644292246
delta:          0.0
```

**Bit-for-bit exact**, to 16 significant figures. On GPU, the same checkpoint
scores ~0.8554-0.8556 regardless of precision (fp32/fp16/bf16 all agree with
each other — see `kaggle/diagnose-precision`). So the variable is not
precision. It is the **backend** — TPU/XLA vs GPU/CUDA execution of this
specific DINOv2 + attention-head architecture produce genuinely different
numerical outputs, large enough to move macro AUC by 0.007-0.035 depending on
fold (`kaggle/diagnose-fresh24` measured this across all 10 24-epoch
checkpoints, all mismatching in the same direction).

Ruled out before landing on this (see `kaggle/diagnose-fresh24`,
`kaggle/diagnose-precision` and the 2026-09-07 commit history for the full
trail): fold membership (GroupKFold is provably row-order-independent, and
`n_groups`/fold sizes match exactly), data drift (labels and cache files are
untouched since well before these checkpoints trained), and
architecture/loading (`strict=True` state_dict loads succeed; no
non-persistent random buffers in `slotnet.py`).

## Why this matters beyond "the evaluator has a bug"

This project's own operational rule has always been **train on TPU, submit
on GPU or CPU** (TPU can't decode DICOM). That means every 24-epoch
checkpoint's "recorded CV score" — used for best-epoch selection during
training, and for every CV-vs-LB comparison in this project's history — was
computed on TPU, while the actual submitted, scored predictions have always
been computed on GPU/CPU.

This plausibly explains a pattern that looked like a mystery before now:
every 24-epoch submission scored dramatically better on the leaderboard than
its TPU-measured CV gain predicted —

| Family | CV gain (TPU) | LB gain |
|---|---|---|
| small, 10ep -> 24ep | +0.0076 (below the project's own +0.02 "real gain" bar) | +0.023 |
| base, 10ep -> 24ep | +0.0188 (also below the bar) | +0.023 |

The working theory in this project's history was "the label-noise ceiling
makes CV undersell real gains near 0.82-0.83 CV." That may still be part of
the story, but this finding suggests a second, likely LARGER effect was
silently mixed in the whole time: **the CV number was measuring the wrong
backend.** The GPU-deployed model was always meaningfully better than its
own TPU-side validation showed.

## What this changes going forward

**Do not compare a GPU-computed OOF/ensemble score against a TPU-recorded
checkpoint score and call the difference a bug.** They are expected to
differ, structurally, for this architecture on this Kaggle image stack.

For any evaluator or blend-optimization work (nested-fold label-specific
rank-blend weighting, etc.): **run it on GPU and treat the GPU-computed OOF
number as the correct baseline**, not the TPU-recorded one — GPU is what
actually gets deployed and scored on the leaderboard, so it is the ground
truth that matters for blend decisions. The `evaluate-fresh24`-style
assertion (`abs(recomputed - recorded) < 0.003`) should be re-scoped: it is
a valid check for "did I load the right checkpoint against the right data"
(fold membership, labels, architecture), but it must not compare against the
TPU-recorded score on a GPU run. A safer version: assert internal
consistency (e.g., re-running the SAME GPU-side OOF computation twice gives
the same number), and treat the TPU-vs-GPU gap as a known, accepted
constant rather than a threshold to close.

## What is NOT yet known

- Which specific operation diverges between XLA and CUDA execution here
  (attention kernel selection, some HF `transformers` device-conditional
  code path, a `torch_xla`-specific lowering of an op DINOv2 uses) is still
  unidentified. Not necessary to know in order to proceed — the practical
  fix (evaluate on the deployment backend) does not require it — but worth
  flagging as an open question if anyone wants to chase it later.
- Whether this same TPU/GPU gap exists for the older 10-epoch checkpoints
  and every past submission, or is specific to something about the 24-epoch
  configuration, has not been tested. Given the mechanism (backend, not
  epoch count) there's no reason to expect it's 24-epoch-specific, but it
  hasn't been directly measured on a 10-epoch checkpoint.
