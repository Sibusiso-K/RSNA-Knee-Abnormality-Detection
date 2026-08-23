#!/usr/bin/env bash
# EPOCHS=4 to stay directly comparable with the 0.7746 fold-0 run. OneCycleLR
# sets total_steps from EPOCHS, so a different epoch count is a different LR
# schedule and the two runs stop being comparable at all.
set -euo pipefail
sed 's/^BATCH, ACCUM, EPOCHS, LR = 2, 4, 8, 3e-4/BATCH, ACCUM, EPOCHS, LR = 2, 4, 4, 3e-4/' \
  "$1/notebooks/kaggle_02_train.py" > "$2/script.py"
grep -q "EPOCHS, LR = 2, 4, 4" "$2/script.py" || { echo "epoch patch missed" >&2; exit 1; }
