#!/usr/bin/env bash
# Experiment 2 (docs/15), control leg: UNDECIDED_WEIGHT=1.0, folds 0,1 only.
#
# Same reduction code path as the two candidates (uncertainty_weighted_bce
# with weight=1.0 collapses to the ordinary BCE mean - see its own test),
# not a separately-coded baseline, so the comparison isolates the WEIGHT,
# not the reduction implementation. Same small/xattn/6-slice/336px/24-epoch
# recipe, same labels, same scanner folds as every other 24ep run here.
#
# Two folds only, per docs/15: "Lock the winning policy before testing the
# remaining folds." Folds 2-4 are held out of this pilot on purpose.
set -euo pipefail
sed -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "xattn")|' \
    -e 's|os.environ.get("FOLDS", "0")|os.environ.get("FOLDS", "0,1")|' \
    -e 's|os.environ.get("EPOCHS", "10")|os.environ.get("EPOCHS", "24")|' \
    -e 's|os.environ.get("UNDECIDED_WEIGHT", "1.0")|os.environ.get("UNDECIDED_WEIGHT", "1.0")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("HEAD", "xattn")'            "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "0,1")'             "$2/script.py" || { echo "FOLDS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("EPOCHS", "24")'             "$2/script.py" || { echo "EPOCHS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("UNDECIDED_WEIGHT", "1.0")'  "$2/script.py" || { echo "UNDECIDED_WEIGHT patch missed" >&2; exit 1; }
grep -q 'uncertainty_weighted_bce'                   "$2/script.py" || { echo "weighted-loss wiring missed" >&2; exit 1; }
grep -q 'take_group'                                 "$2/script.py" || { echo "multi-group cache needs group sampling" >&2; exit 1; }
grep -q 'xm.mark_step()'                             "$2/script.py" || { echo "XLA needs mark_step" >&2; exit 1; }
grep -q 'TrainingWatchdog'                           "$2/script.py" || { echo "stall watchdog missed" >&2; exit 1; }
