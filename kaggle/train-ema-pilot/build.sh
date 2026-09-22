#!/usr/bin/env bash
# docs/20 job 2: controlled EMA pilot, folds 0,1 only - the last untested
# candidate from docs/15. Identical recipe to the deployed small/xattn/
# 6-slice/336px/24-epoch family (labels, augmentation, LR schedule,
# UNDECIDED_WEIGHT=1.0 control) except:
#   - EMA_DECAY=0.999 fixed here, not searched. At ~435 steps/epoch x 24
#     epochs (~10,400 updates total for this fold's train split), decay
#     0.999 gives an effective averaging window of ~1000 steps (~2.3
#     epochs) - late-training smoothing without lagging so far it never
#     converges toward the end of a 24-epoch schedule.
#   - PER_FOLD_SEED=1: each fold seeds independently (SEED + fold) so an
#     isolated single-fold retry reproduces that fold's exact draws
#     regardless of whether the other fold ran first - see docs/20 and
#     src/model/ema.py's own tests for what the EMA class itself asserts.
#
# Ordinary and EMA weights are both saved at epoch 24 (see kaggle_06's
# `is not None and human_epoch == EPOCHS` block): *_ordinary_final.pth and
# *_ema_final.pth per fold, plus the existing best/snapshot files unchanged.
set -euo pipefail
sed -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "xattn")|' \
    -e 's|os.environ.get("FOLDS", "0")|os.environ.get("FOLDS", "0,1")|' \
    -e 's|os.environ.get("EPOCHS", "10")|os.environ.get("EPOCHS", "24")|' \
    -e 's|os.environ.get("EMA_DECAY", "")|os.environ.get("EMA_DECAY", "0.999")|' \
    -e 's|os.environ.get("PER_FOLD_SEED", "0")|os.environ.get("PER_FOLD_SEED", "1")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("HEAD", "xattn")'         "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "0,1")'           "$2/script.py" || { echo "FOLDS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("EPOCHS", "24")'           "$2/script.py" || { echo "EPOCHS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("EMA_DECAY", "0.999")'     "$2/script.py" || { echo "EMA_DECAY patch missed" >&2; exit 1; }
grep -q 'os.environ.get("PER_FOLD_SEED", "1")'     "$2/script.py" || { echo "PER_FOLD_SEED patch missed" >&2; exit 1; }
grep -q 'from src.model.ema import EMA'            "$2/script.py" || { echo "EMA import missed" >&2; exit 1; }
grep -q 'take_group'                               "$2/script.py" || { echo "multi-group cache needs group sampling" >&2; exit 1; }
grep -q 'xm.mark_step()'                           "$2/script.py" || { echo "XLA needs mark_step" >&2; exit 1; }
grep -q 'TrainingWatchdog'                         "$2/script.py" || { echo "stall watchdog missed" >&2; exit 1; }
