#!/usr/bin/env bash
# Fold 0 with the cross-attention head. One variable: the head.
#
# Everything else is held at the 0.7990 baseline - same 336 px cache, same
# DINOv2-small, same labels, same 10 epochs, same folds - so the comparison is
# clean.
#
# Why this and not another encoder or another resolution: four experiments
# returned nothing (epochs twice, base -0.0005, 448 -0.0014) and the published
# top pipelines differ from ours mainly in the HEAD. SlotHead pools each slot to
# one vector, so twelve target queries attend over six numbers and every
# spatial detail is gone before any finding-specific reasoning happens. That
# single choice explains all four nulls at once, and it explains why the focal
# findings (PF OA .711, MCL .747, Lateral Meniscus .754) trail the diffuse ones
# (Baker's .845, Medial OA .832) - focal evidence is exactly what a mean
# destroys.
#
# Costs 0.31M -> 1.08M head parameters, 11.0M -> 11.7M trainable. It is access,
# not capacity, which matters because capacity is what already failed.
#
# Bar: >= ~0.02 over 0.7990 to clear the +/-0.011 single-fold noise floor.
set -euo pipefail
sed -e 's|SlotNet(dinov2, unfreeze_last=UNFREEZE_LAST, pool=POOL)|SlotNet(dinov2, unfreeze_last=UNFREEZE_LAST, pool=POOL, head="xattn")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'head="xattn"' "$2/script.py" || { echo "head patch missed - this would silently re-run the 0.7990 baseline and look like the head does not help" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "0")' "$2/script.py" || { echo "keep FOLDS=0: one fold, one variable" >&2; exit 1; }
grep -q 'os.environ.get("SIZE", str(IMG))' "$2/script.py" || { echo "SIZE must stay at the 336 default; this run changes the head only" >&2; exit 1; }
grep -q 'xm.mark_step()' "$2/script.py" || { echo "XLA needs an explicit mark_step" >&2; exit 1; }
