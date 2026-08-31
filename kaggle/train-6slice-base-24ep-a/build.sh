#!/usr/bin/env bash
# DINOv2-base family, fold 0 ONLY, EPOCHS 10 -> 24. A gate, not a commitment.
#
# The unsplit train-6slice-base-24ep burned the FULL 9h TPU session cap
# without saving even fold 0's checkpoint (0 bytes) - the same failure
# signature as knee-train-pseudo-sel's separate 12h GPU timeout with only an
# epoch-2 checkpoint to show for it. Both got an empty log because Kaggle's
# log shipping does not survive the hard SIGKILL at the timeout. It is not
# yet known whether that was a true stall (now caught fast by the watchdog
# added to kaggle_06_train_slots.py) or the base encoder genuinely needing
# more than 9h for even one fold at this size/epoch count.
#
# So: one fold, find out which, THEN decide how to split the remaining four
# (or whether 24 epochs is even viable for base without a longer session).
set -euo pipefail
sed -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "xattn")|' \
    -e 's|os.environ.get("VARIANT", "small")|os.environ.get("VARIANT", "base")|' \
    -e 's|os.environ.get("FOLDS", "0")|os.environ.get("FOLDS", "0")|' \
    -e 's|os.environ.get("EPOCHS", "10")|os.environ.get("EPOCHS", "24")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("HEAD", "xattn")'   "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("VARIANT", "base")' "$2/script.py" || { echo "variant patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "0")'      "$2/script.py" || { echo "FOLDS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("EPOCHS", "24")'    "$2/script.py" || { echo "EPOCHS patch missed" >&2; exit 1; }
grep -q 'take_group'                        "$2/script.py" || { echo "multi-group cache needs group sampling" >&2; exit 1; }
grep -q 'xm.mark_step()'                    "$2/script.py" || { echo "XLA needs mark_step" >&2; exit 1; }
grep -q '_watchdog'                         "$2/script.py" || { echo "stall watchdog missed" >&2; exit 1; }
