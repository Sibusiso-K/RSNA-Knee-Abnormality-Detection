#!/usr/bin/env bash
# docs/23: single-fold split of kaggle/train-small-cuda-pilot (fold 0 only),
# used if the combined two-fold kernel's projected runtime does not fit one
# GPU session/quota budget comfortably. Same recipe, same seeding, otherwise
# identical - see that directory's build.sh for the full rationale.
set -euo pipefail
sed -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "xattn")|'     -e 's|os.environ.get("FOLDS", "0")|os.environ.get("FOLDS", "0")|'     -e 's|os.environ.get("EPOCHS", "10")|os.environ.get("EPOCHS", "24")|'     -e 's|os.environ.get("PER_FOLD_SEED", "0")|os.environ.get("PER_FOLD_SEED", "1")|'     -e 's|os.environ.get("USE_XLA", "auto")|"0"|'     "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("HEAD", "xattn")'      "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "0")'        "$2/script.py" || { echo "FOLDS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("EPOCHS", "24")'       "$2/script.py" || { echo "EPOCHS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("PER_FOLD_SEED", "1")' "$2/script.py" || { echo "PER_FOLD_SEED patch missed" >&2; exit 1; }
grep -q 'if "0" != "0":'                       "$2/script.py" || { echo "USE_XLA patch missed" >&2; exit 1; }
grep -q 'EMA_DECAY = os.environ.get("EMA_DECAY", "")' "$2/script.py" || { echo "EMA_DECAY not left unset" >&2; exit 1; }
