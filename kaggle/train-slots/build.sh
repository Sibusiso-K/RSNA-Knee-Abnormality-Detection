#!/usr/bin/env bash
# Train SlotNet on the cached slot representation. GPU, internet OFF.
#
# The cache arrives via `kernel_sources`, not as a Dataset: the knee-cache
# kernel's output is mounted directly, so there is no 9 GB publish-and-reattach
# step between building it and training on it.
#
# Internet is OFF even though this is not a submission kernel. DINOv2 comes
# from the mounted model directory, and a run that quietly downloads weights
# is a run whose exact encoder is unrecorded - and therefore not reproducible
# in the submission notebook, which cannot download anything at all.
#
# Fold 0 only by default. At 30 h/week the cost of a fold has to be MEASURED
# before five of them are committed; docs/00-state.md records a run that ate
# 8.3 h to learn nothing, which is a quarter of a week's budget.
set -euo pipefail
cp "$1/notebooks/kaggle_06_train_slots.py" "$2/script.py"

grep -q 'LR_BACKBONE = 8e-6' "$2/script.py" || { echo "backbone LR must stay tiny - a single 1e-3 retrains the ViT and destroys the pretrained features" >&2; exit 1; }
grep -q 'max_lr=\[LR_BACKBONE, LR_HEAD\]' "$2/script.py" || { echo "OneCycleLR max_lr must be a per-group LIST; a scalar overwrites both groups and silently reinstates the bug above" >&2; exit 1; }
grep -q 'grouped_folds' "$2/script.py" || { echo "must use grouped folds - random KFold overstates macro AUC by ~0.053 through site memorisation" >&2; exit 1; }
grep -q 'gold_uids' "$2/script.py" || { echo "the 58 gold studies must be held out" >&2; exit 1; }
