#!/usr/bin/env bash
# DINOv2-base family, folds 1+2, EPOCHS 10 -> 24.
#
# The fold-0 gate (train-6slice-base-24ep-a) finished cleanly in ~301s/epoch
# x 24 = ~2h, scoring 0.8317 - well under the 9h TPU session cap and a real
# gain over the 10-epoch base baseline (fold0 0.8121). At ~2h/fold, all 5
# folds would be ~10h - still over the cap - so folds 1-4 split into two
# 2-fold kernels (~4h each) rather than risk a repeat of the run that burned
# a full 9h for zero saved output.
set -euo pipefail
sed -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "xattn")|' \
    -e 's|os.environ.get("VARIANT", "small")|os.environ.get("VARIANT", "base")|' \
    -e 's|os.environ.get("FOLDS", "0")|os.environ.get("FOLDS", "1,2")|' \
    -e 's|os.environ.get("EPOCHS", "10")|os.environ.get("EPOCHS", "24")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("HEAD", "xattn")'   "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("VARIANT", "base")' "$2/script.py" || { echo "variant patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "1,2")'    "$2/script.py" || { echo "FOLDS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("EPOCHS", "24")'    "$2/script.py" || { echo "EPOCHS patch missed" >&2; exit 1; }
grep -q 'take_group'                        "$2/script.py" || { echo "multi-group cache needs group sampling" >&2; exit 1; }
grep -q 'xm.mark_step()'                    "$2/script.py" || { echo "XLA needs mark_step" >&2; exit 1; }
grep -q '_watchdog'                         "$2/script.py" || { echo "stall watchdog missed" >&2; exit 1; }
