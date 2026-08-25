#!/usr/bin/env bash
# The exact 0.864 recipe (xattn head, 6 slices/slot, 5 folds), EPOCHS 10 -> 24.
# Nothing else changed.
#
# Why: 8 of 10 checkpoints across the two most recent 5-fold runs peaked at
# epoch 9 of 10 - the LAST epoch - across both label sets and both encoder
# sizes. Every model in the project is undertrained. The "epochs are null"
# result on record was measured on the OLD 3-slice config and does not cover
# this head/coverage combination.
set -euo pipefail
sed -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "xattn")|' \
    -e 's|os.environ.get("FOLDS", "0")|os.environ.get("FOLDS", "0,1,2,3,4")|' \
    -e 's|os.environ.get("EPOCHS", "10")|os.environ.get("EPOCHS", "24")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("HEAD", "xattn")'      "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "0,1,2,3,4")' "$2/script.py" || { echo "FOLDS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("EPOCHS", "24")'       "$2/script.py" || { echo "EPOCHS patch missed" >&2; exit 1; }
grep -q 'take_group'                           "$2/script.py" || { echo "multi-group cache needs group sampling" >&2; exit 1; }
grep -q 'xm.mark_step()'                       "$2/script.py" || { echo "XLA needs mark_step" >&2; exit 1; }
