# 24 — CUDA-trained vs TPU-trained small member: negative result, no submission

Date: 2026-10-12 (evaluation finished 2026-09-23 run). docs/23 pilot.

## What was compared

- **Control (TPU-trained):** `knee-ema-pilot-v1` `ordinary` leg, small/xattn/6-slice/336px,
  24 epochs, per-fold seed (SEED+fold), epoch-24 checkpoint. EMA never wrote into this
  model, so it is the unmodified TPU control.
- **Pilot (CUDA-trained):** `knee-train-small-cuda-pilot` v1, same recipe, same verified
  initial weights (docs/23 SHA256 match on folds 0 and 1), EMA disabled, epoch-24 snapshot.
- **Evaluator:** `kaggle/evaluate-cuda-vs-tpu-pilot`, CUDA T4, exact production five-view
  TTA (`src/model/tta.py`), folds 0 and 1 only, each checkpoint scored on its held-out fold.

Gate fixed in advance (docs/23, user instruction): a win against a weaker control alone is
insufficient; the ensemble swap must support it.

## Results

Mean AUC over folds 0 and 1, paired scanner-group bootstrap (500 resamples).

| Comparison | TPU-trained | CUDA-trained | Δ (CUDA − TPU) | 95% CI | Bootstrap P(Δ>0) | Folds ≥ 0 |
|---|---|---|---|---|---|---|
| Standalone small member | 0.842266 | 0.822726 | **−0.01954** | [−0.04867, −0.00868] | 0.000 | 0/2 |
| Swapped into 0.5·small + 0.5·base | 0.853056 | 0.846479 | **−0.00658** | [−0.01718, −0.00254] | 0.000 | 0/2 |

Per fold (standalone): fold 0 Δ −0.01993, fold 1 Δ −0.01915. Per fold (ensemble): fold 0
Δ −0.00707, fold 1 Δ −0.00608.

Runtime (T4, TTA, 870 studies per fold): ~0.25 s/study per small member, ~0.62 s/study base.
No runtime concern.

## Decision

**No submission.** The gate is failed on both comparisons: the CUDA-trained small member is
worse standalone and worse inside the deployed 0.893 blend, on both folds, with every
interval excluding zero on the negative side.

## Interpretation and caveats

- Initial weights were verified identical across backends, so this difference arose
  during training or evaluation, not from initialization. The docs/23 pilot is a
  deployment-backend experiment, not an isolated hardware comparison; this result says
  the CUDA training path, as currently configured, yields a weaker small member under the
  same recipe.
- A 0.02 standalone AUC gap from backend alone is large. Candidate causes not yet
  isolated: fp16 autocast with GradScaler on CUDA versus bf16 autocast on XLA; differing
  kernel numerics over 24 epochs. A CUDA run with autocast disabled would test precision
  directly, but costs roughly another 8 GPU hours.
- Only two folds were tested, and the pilot's scope was pre-declared. Folds 2–4 are not
  justified for expansion by this result.
- Epoch-24 selection was predeclared. On-device validation scores are not comparable across
  backends (docs/12, docs/16: TPU and CUDA scoring disagree), so the TTA comparison above is
  the one that matters.

## Artifacts

- `kaggle/evaluate-cuda-vs-tpu-pilot/results/summary.json`
- `kaggle/evaluate-cuda-vs-tpu-pilot/results/predictions.npz`
- Training: `kaggle/train-small-cuda-pilot/`; initialization check: `kaggle/verify-init-cuda/`,
  `kaggle/verify-init-tpu/`; throughput probe: `kaggle/probe-small-cuda/`.
- The 0.893 TTA submission (56441791) remains the anchor and only live candidate.
