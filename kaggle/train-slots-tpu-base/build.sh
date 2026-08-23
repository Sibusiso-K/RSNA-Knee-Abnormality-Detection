#!/usr/bin/env bash
# Fold 0 with DINOv2-BASE instead of small. One variable.
#
# Why this and not more slices: the public 0.899 notebook runs N_GROUP_MAX = 1,
# so it gets no inference-time group averaging either - more slices per slot is
# NOT what separates us from it, and a 4 h cache rebuild would have bought a
# difference that recipe does not contain.
#
# Encoder capacity is untested and is the largest single lever left. base is
# 768-wide against small's 384, ~4x the parameters, and ~4x the activation
# memory - which is the risk: batch 8 = 48 slot images at 336px fitted small in
# 15.75 GB of HBM with room to spare, and base may not. An OOM shows up at the
# first compile, ~2 min in, so this is cheap to find out.
#
# Compare against fold 0 small = 0.7990. Fold noise measured across the 5-fold
# run is +/-0.011, so base must give >= ~0.02 to be believed on one fold.
set -euo pipefail
cp "$1/notebooks/kaggle_06_train_slots.py" "$2/script.py"
grep -q 'os.environ.get("FOLDS", "0")' "$2/script.py" || { echo "keep FOLDS=0: one fold, one variable" >&2; exit 1; }
grep -q 'xm.mark_step()' "$2/script.py" || { echo "XLA needs an explicit mark_step" >&2; exit 1; }
grep -q 'hidden size' "$2/script.py" || { echo "must log the encoder width - it is the variable under test" >&2; exit 1; }
