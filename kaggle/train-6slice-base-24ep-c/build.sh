#!/usr/bin/env bash
# DINOv2-base family, folds 3+4, EPOCHS 10 -> 24. Second half of the split
# described in train-6slice-base-24ep-b/build.sh.
set -euo pipefail
sed -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "xattn")|' \
    -e 's|os.environ.get("VARIANT", "small")|os.environ.get("VARIANT", "base")|' \
    -e 's|os.environ.get("FOLDS", "0")|os.environ.get("FOLDS", "3,4")|' \
    -e 's|os.environ.get("EPOCHS", "10")|os.environ.get("EPOCHS", "24")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("HEAD", "xattn")'   "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("VARIANT", "base")' "$2/script.py" || { echo "variant patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "3,4")'    "$2/script.py" || { echo "FOLDS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("EPOCHS", "24")'    "$2/script.py" || { echo "EPOCHS patch missed" >&2; exit 1; }
grep -q 'take_group'                        "$2/script.py" || { echo "multi-group cache needs group sampling" >&2; exit 1; }
grep -q 'xm.mark_step()'                    "$2/script.py" || { echo "XLA needs mark_step" >&2; exit 1; }
grep -q '_watchdog'                         "$2/script.py" || { echo "stall watchdog missed" >&2; exit 1; }
