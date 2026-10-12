#!/usr/bin/env bash
# docs/25: precision diagnostic. Fold 0 of the CUDA pilot, identical recipe
# (labels, cache, seeds, schedule, EMA off), with CUDA autocast and GradScaler
# disabled so the whole CUDA run is fp32. Compare against kaggle/train-small-cuda-pilot
# (fp16 autocast) and the TPU control (bf16 autocast).
set -euo pipefail
sed -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "xattn")|' \
    -e 's|os.environ.get("FOLDS", "0")|os.environ.get("FOLDS", "0")|' \
    -e 's|os.environ.get("EPOCHS", "10")|os.environ.get("EPOCHS", "24")|' \
    -e 's|os.environ.get("PER_FOLD_SEED", "0")|os.environ.get("PER_FOLD_SEED", "1")|' \
    -e 's|os.environ.get("USE_XLA", "auto")|"0"|' \
    -e 's|os.environ.get("CUDA_AMP", "1")|os.environ.get("CUDA_AMP", "0")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("HEAD", "xattn")'            "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "0")'               "$2/script.py" || { echo "FOLDS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("EPOCHS", "24")'             "$2/script.py" || { echo "EPOCHS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("PER_FOLD_SEED", "1")'       "$2/script.py" || { echo "PER_FOLD_SEED patch missed" >&2; exit 1; }
grep -q 'if "0" != "0":'                             "$2/script.py" || { echo "USE_XLA patch missed" >&2; exit 1; }
grep -q 'os.environ.get("CUDA_AMP", "0")'            "$2/script.py" || { echo "CUDA_AMP patch missed" >&2; exit 1; }
grep -q 'EMA_DECAY = os.environ.get("EMA_DECAY", "")' "$2/script.py" || { echo "EMA_DECAY not left unset" >&2; exit 1; }
grep -q 'take_group'                                 "$2/script.py" || { echo "multi-group cache needs group sampling" >&2; exit 1; }
grep -q 'TrainingWatchdog'                           "$2/script.py" || { echo "stall watchdog missed" >&2; exit 1; }
