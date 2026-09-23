# CPU initialization parity and bounded CUDA throughput

## Status on 2026-09-22

The 0.893 submission remains the anchor. No full training or new submission is authorized by this experiment's launch gate.

Four duplicate polling process trees were stopped after checking their commands. The fifth exited after cancellation of the obsolete TPU run. There are no remaining shell polling loops for these checks. The single replacement is the Codex thread heartbeat `knee-initialization-and-throughput`, every ten minutes, quiet on unchanged status. Do not create another monitor.

The old CUDA v1 log was retrieved before replacement. It records data setup through 2.1 seconds, CPU encoder/head construction and device transfer, and completion at 23.3 seconds. Those timestamps do not explain the longer externally observed RUNNING state. No provisioning explanation has been established. TPU v1 remained queued with no execution log; it was stopped through the Kaggle Active Events menu and cancellation was confirmed by the CLI.

Both v2 replacements were successfully pushed with a 900-second execution timeout:

- `sibusisokhumalo11/knee-verify-init-cuda`: v2, explicit NvidiaTeslaT4; latest check COMPLETE.
- `sibusisokhumalo11/knee-verify-init-tpu`: v2, TPU v5e-8 metadata; latest check QUEUED.

## Update on 2026-09-23

Both v2 checks completed. Their original `init_fingerprints.json` artifacts are preserved under `kaggle/verify-init-cuda/results/` and `kaggle/verify-init-tpu/results/`. The exact SHA256 values match across backends for both folds:

| Fold | Seed | Full CPU state SHA256 |
| --- | ---: | --- |
| 0 | 0 | `39f4532df521c4f87cbee41466be45633908ec9a7807ca8a8eb7fa8aacd77c29` |
| 1 | 1 | `a88fb67a88846f88c65ce1e2412c6d31aac749d520510119f769708a5e107a87` |

The head (`xattn`), pool (`cls_mean_focal`) and DINOv2 encoder path match. Kaggle used PyTorch `2.10.0+cu128` on CUDA and `2.8.0+cpu` in the TPU environment. Despite that runtime difference, the recorded CPU starting states match byte for byte. This verifies the launch gate for this new pilot; it does not reconstruct the unrecorded initialization of historical training runs.

The bounded `knee-probe-small-cuda` **version 1** was pushed once after confirming that no existing kernel was accessible at that slug. Its script was rebuilt and checked for `MAX_STEPS=50`, `VERIFY_INIT_ONLY=0`, `EPOCHS=24`, and fold 0. The launch used explicit `NvidiaTeslaT4` with a 3600-second limit. Await its timing artifact before deciding on full training.

The probe completed. Its original timing artifact is preserved at `kaggle/probe-small-cuda/results/throughput_fold0.json`. The 47 steps after the three warmup steps averaged 1.0383 s/step (median 1.0100, standard deviation 0.0891). The first and second halves averaged 1.0113 and 1.0642 s/step respectively; there is a small upward drift and one 1.44 s step, so the mean should not be treated as a guarantee. The first step took 4.32 s. Full fold validation, including the gold holdout and scoring, took 127.7 s; the disposable weights save took 0.13 s. The entire 50-step probe took 226.6 s within the script.

At that measured rate, a 24-epoch fold projects to 13,887 s (3.86 h), including one validation and one save per epoch plus the measured warmup excess. Two folds project to **7.71 h**. This leaves **4.29 h** against a 12-hour session for startup, extra snapshots, variation in step/validation time and platform overhead. The observed in-script startup before the first training step was about 43 s. The paired two-fold run appears feasible in one GPU session, but its actual runtime and weekly quota should be monitored during full training. This experiment alone does not establish an AUC gain or authorize a submission.

## What changed

`VERIFY_INIT_ONLY` now exits before backend initialization, cache, labels or fold-data loading. Both verification configs attach only knee-src and DINOv2. Helpers are baked into the training script, so no knee-src republish is needed.

Verification and training call the same `construct_fold_model()` with the same per-fold seed. The verifier hashes the full CPU state before any device transfer, including tensor names, shapes, original dtypes and contiguous raw bytes with length framing. Parameters and persistent buffers are included; scalar and bfloat16 tensors are supported without converting their values to float64. Results are saved in `init_fingerprints.json`.

Validation: 236 tests passed, four expected failures. All five pre-existing builders and the new bounded probe builder passed; generated scripts parse. Seven new tests cover raw hashing and shared construction/early exit. This validates the local implementation, not remote fingerprint parity.

## Launch gate and next action

1. Retrieve **v2** artifacts into separate CUDA/TPU directories. Confirm artifact version/source against the launched script. Do not reuse the obsolete v1 hashes.
2. Compare exactly two fold records, folds 0 and 1, including seed and SHA256; head and pool must also match. Record runtime library versions and preserve both original JSON files. On any mismatch or failed run, stop and investigate; do not launch training.
3. Only when both checks complete and match, build `kaggle/probe-small-cuda/build.sh` with repo root and output directory arguments. It inherits the fold-0 CUDA pilot config, sets MAX_STEPS=50, and keeps VERIFY_INIT_ONLY off. Use the full 24-epoch LR schedule, not a shortened schedule.
4. Push `sibusisokhumalo11/knee-probe-small-cuda` once with `--accelerator NvidiaTeslaT4 -t 3600`. Record the resulting version before retrying anything. Check remote status before any repeated push.
5. Retrieve `throughput_fold0.json` and the log. It separates the first three warmup steps, the remaining steady steps, full fold/gold validation and scoring, and a disposable checkpoint save. The projection includes warmup excess but excludes startup and additional snapshot writes. Include those and a safety margin when estimating session/quota feasibility. The saved step samples allow checking whether three warmup steps were sufficient; do not assume stability without looking.
6. Report feasibility before any full training. Remove the heartbeat when the probe finishes or a failure requires a new decision.

CLI from the repo: use the bundled Python with `-X utf8 ..\kaggle_cli.py`. The supported push accelerator name is `NvidiaTeslaT4`. The installed CLI has no kernel cancellation command; use Kaggle's visible Stop Session action if needed.
