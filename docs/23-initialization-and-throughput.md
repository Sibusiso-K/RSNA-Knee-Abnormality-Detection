# CPU initialization parity and bounded CUDA throughput

## Status on 2026-09-22

The 0.893 submission remains the anchor. No full training or new submission is authorized by this experiment's launch gate.

Four duplicate polling process trees were stopped after checking their commands. The fifth exited after cancellation of the obsolete TPU run. There are no remaining shell polling loops for these checks. The single replacement is the Codex thread heartbeat `knee-initialization-and-throughput`, every ten minutes, quiet on unchanged status. Do not create another monitor.

The old CUDA v1 log was retrieved before replacement. It records data setup through 2.1 seconds, CPU encoder/head construction and device transfer, and completion at 23.3 seconds. Those timestamps do not explain the longer externally observed RUNNING state. No provisioning explanation has been established. TPU v1 remained queued with no execution log; it was stopped through the Kaggle Active Events menu and cancellation was confirmed by the CLI.

Both v2 replacements were successfully pushed with a 900-second execution timeout:

- `sibusisokhumalo11/knee-verify-init-cuda`: v2, explicit NvidiaTeslaT4; latest check COMPLETE.
- `sibusisokhumalo11/knee-verify-init-tpu`: v2, TPU v5e-8 metadata; latest check QUEUED.

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
