#!/usr/bin/env bash
# The DINOv2-base family, EPOCHS 10 -> 24. Same fix as train-6slice-24ep,
# applied to the second encoder size.
#
# Why: the undertraining diagnosis (8 of 10 checkpoints across the two most
# recent 5-fold runs peaking at epoch 9 of 10) held across BOTH label sets
# AND both encoder sizes - so if the small-encoder 24ep run measured
# anything real, the base family should show the same pattern rather than
# being a one-off. Only model_sources differs from train-6slice-24ep
# (base/1 instead of small/1); find_dinov2() picks up whichever DINOv2
# checkpoint is mounted, so no code path needs to branch on variant, but
# VARIANT is still baked to "base" purely for correct bookkeeping in the
# saved checkpoint metadata.
set -euo pipefail
sed -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "xattn")|' \
    -e 's|os.environ.get("VARIANT", "small")|os.environ.get("VARIANT", "base")|' \
    -e 's|os.environ.get("FOLDS", "0")|os.environ.get("FOLDS", "0,1,2,3,4")|' \
    -e 's|os.environ.get("EPOCHS", "10")|os.environ.get("EPOCHS", "24")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("HEAD", "xattn")'      "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("VARIANT", "base")'    "$2/script.py" || { echo "variant patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "0,1,2,3,4")' "$2/script.py" || { echo "FOLDS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("EPOCHS", "24")'       "$2/script.py" || { echo "EPOCHS patch missed" >&2; exit 1; }
grep -q 'take_group'                           "$2/script.py" || { echo "multi-group cache needs group sampling" >&2; exit 1; }
grep -q 'xm.mark_step()'                       "$2/script.py" || { echo "XLA needs mark_step" >&2; exit 1; }
