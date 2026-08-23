#!/usr/bin/env bash
# 24 slices @ 256 px, up from 16 @ 224.
#
# Budget: ~1.96x the compute of the 16/224 run, so ~8.1 h against the 12 h
# ceiling — the 4 h margin matters because a hard timeout can take the
# checkpoint with it.
#
# BATCH drops 2 -> 1 and ACCUM rises 4 -> 8, which keeps the effective batch at
# 8 (so the optimisation is unchanged) while holding peak activation memory at
# 0.98x of the run we know fits a 15 GB T4. At BATCH=2 this config would push
# 144 images of 256px through a forward pass instead of 96 of 224px, and OOM is
# a real risk there.
#
# EPOCHS stays 4 so the end-of-schedule number is comparable with 0.7746
# (16/224) and 0.7767 (16/224 + crop). OneCycleLR derives its schedule from
# EPOCHS, so changing it would break that comparison.
#
# Why more slices specifically: the weakest labels are the menisci (0.702 /
# 0.695) and the domain primer predicts why — a tear can span ~3 slices out of
# 16, so coverage is the axis most likely to be starving that label.
set -euo pipefail
sed -e 's/^N_SLICES, SIZE = 16, 224/N_SLICES, SIZE = 24, 256/' \
    -e 's/^BATCH, ACCUM, EPOCHS, LR = 2, 4, 8, 3e-4/BATCH, ACCUM, EPOCHS, LR = 1, 8, 4, 3e-4/' \
    "$1/notebooks/kaggle_02_train.py" > "$2/script.py"
grep -q "N_SLICES, SIZE = 24, 256"          "$2/script.py" || { echo "slice/size patch missed" >&2; exit 1; }
grep -q "EPOCHS, LR = 1, 8, 4"              "$2/script.py" || { echo "batch/epoch patch missed" >&2; exit 1; }
