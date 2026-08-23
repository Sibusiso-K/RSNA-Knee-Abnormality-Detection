#!/usr/bin/env bash
# Run 1 on the TPU: fold 0, DINOv2-small, 10 epochs, site-grouped CV.
#
# The TPU's 20 h/week is a SEPARATE budget from the GPU's 30 h, and it is
# untouched, so this run costs nothing we were otherwise going to spend.
# Measured on a v5e-8 (knee-tpu-probe, 2026-08-12): grid_sample and topk both
# lower, one core sustains 63.6 study/s warm after a ~10 s compile, and a
# 10-epoch fold projects to ~0.2 h of COMPUTE. Real throughput will be lower
# because that projection used synthetic tensors and did not pay for gathering
# rows out of a 9 GB cache.
#
# This is a measurement, not an attempt to win: one honest grouped-CV number
# on the new representation, to compare against 0.7746 from the old pipeline.
set -euo pipefail
cp "$1/notebooks/kaggle_06_train_slots.py" "$2/script.py"

grep -q 'xm.optimizer_step' "$2/script.py" || { echo "XLA needs xm.optimizer_step, not optimizer.step" >&2; exit 1; }
grep -q 'LR_BACKBONE = 8e-6' "$2/script.py" || { echo "backbone LR must stay tiny - a single 1e-3 retrains the ViT and destroys the pretrained features" >&2; exit 1; }
grep -q 'max_lr=\[LR_BACKBONE, LR_HEAD\]' "$2/script.py" || { echo "OneCycleLR max_lr must be a per-group LIST; a scalar overwrites both groups and reinstates the bug above" >&2; exit 1; }
grep -q 'grouped_folds' "$2/script.py" || { echo "must use grouped folds - random KFold overstates macro AUC by ~0.053" >&2; exit 1; }
grep -q 'gold_uids' "$2/script.py" || { echo "the 58 gold studies must be held out" >&2; exit 1; }
