#!/usr/bin/env bash
# docs/23: CUDA-vs-TPU deployment-backend pilot, small/xattn family, folds
# 0 and 1, BOTH folds in one kernel. Matches the TPU ordinary-weight control
# (kaggle/train-ema-pilot's "ordinary" leg) exactly except the backend:
#
#   - same per-fold seeds (PER_FOLD_SEED=1, SEED default 0 -> fold_seed 0,1)
#   - same labels, six-slice cache, BATCH, optimizer, augmentation, 24-epoch
#     OneCycleLR schedule (all unchanged shared-script defaults)
#   - EMA_DECAY left UNSET -> EMA disabled (see kaggle_06's default)
#   - USE_XLA forced to "0": this kernel has a GPU attached and no TPU, but
#     forcing it explicitly documents the intent rather than relying on
#     torch_xla being absent.
#
# Use kaggle/train-small-cuda-pilot-f0 / -f1 instead if the throughput probe
# (kaggle/verify-init-cuda with MAX_STEPS set, or this kernel's own first
# fold) shows the combined two-fold run would not fit a 12h GPU session or
# would eat too much of the 30h/week GPU quota in one job.
set -euo pipefail
sed -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "xattn")|' \
    -e 's|os.environ.get("FOLDS", "0")|os.environ.get("FOLDS", "0,1")|' \
    -e 's|os.environ.get("EPOCHS", "10")|os.environ.get("EPOCHS", "24")|' \
    -e 's|os.environ.get("PER_FOLD_SEED", "0")|os.environ.get("PER_FOLD_SEED", "1")|' \
    -e 's|os.environ.get("USE_XLA", "auto")|"0"|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("HEAD", "xattn")'      "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "0,1")'       "$2/script.py" || { echo "FOLDS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("EPOCHS", "24")'       "$2/script.py" || { echo "EPOCHS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("PER_FOLD_SEED", "1")' "$2/script.py" || { echo "PER_FOLD_SEED patch missed" >&2; exit 1; }
grep -q 'if "0" != "0":'                       "$2/script.py" || { echo "USE_XLA patch missed" >&2; exit 1; }
grep -q 'EMA_DECAY = os.environ.get("EMA_DECAY", "")' "$2/script.py" || { echo "EMA_DECAY not left unset" >&2; exit 1; }
grep -q 'take_group'                           "$2/script.py" || { echo "multi-group cache needs group sampling" >&2; exit 1; }
grep -q 'TrainingWatchdog'                     "$2/script.py" || { echo "stall watchdog missed" >&2; exit 1; }
