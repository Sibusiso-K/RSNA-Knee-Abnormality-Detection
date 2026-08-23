#!/usr/bin/env bash
# Cross-attention head + 20 epochs. Two patches, both asserted.
#
# 20 epochs was already tested ONCE and returned +0.0016 - noise - so re-running
# it needs a reason. The reason: that test ran on the OLD head, which plateaued
# at epoch ~5 and then oscillated while loss kept falling. The new head has not
# plateaued at all - 0.8008, 0.8021, 0.8035 across epochs 7-9, still climbing at
# the end of the schedule with loss still falling.
#
# So the earlier null may have measured the bottleneck rather than the schedule:
# a head with nowhere to put extra training cannot benefit from more of it.
# Worth exactly one run to find out, and no more.
#
# Compare END-OF-SCHEDULE only: OneCycleLR stretches to whatever EPOCHS is, so
# epoch 9 here is mid-anneal and epoch 9 of the 10-epoch run was finished.
# Baseline to beat: xattn @10ep = 0.8035.
set -euo pipefail
sed -e 's|SlotNet(dinov2, unfreeze_last=UNFREEZE_LAST, pool=POOL)|SlotNet(dinov2, unfreeze_last=UNFREEZE_LAST, pool=POOL, head="xattn")|' \
    -e 's|os.environ.get("EPOCHS", "10")|os.environ.get("EPOCHS", "20")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'head="xattn"'                  "$2/script.py" || { echo "head patch missed - would silently re-run the slot-head baseline" >&2; exit 1; }
grep -q 'os.environ.get("EPOCHS", "20")' "$2/script.py" || { echo "EPOCHS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "0")'   "$2/script.py" || { echo "keep FOLDS=0" >&2; exit 1; }
grep -q 'xm.mark_step()'                 "$2/script.py" || { echo "XLA needs mark_step" >&2; exit 1; }
