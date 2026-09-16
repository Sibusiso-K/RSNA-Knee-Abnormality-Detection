#!/usr/bin/env bash
# Experiment 3 (docs/15): a second seed of the same successful family,
# folds 0,1 only - a pilot, per docs/15: "Expand to five folds only after
# the pilot supports additional value."
#
# Identical labels, preprocessing, schedule and scanner folds to the
# original 24-epoch small/xattn/6-slice/336px family (and to experiment 2's
# control leg): only SEED changes, 0 -> 1. UNDECIDED_WEIGHT left at its
# default 1.0 - experiment 2 found no case for changing it.
#
# The question this pilot answers: does a second seed trained the SAME way
# add real ensemble value (prediction diversity, not just noise), or is it
# another 10-epoch-era null in a different costume? Older seed ensembles in
# this project were all at 10 epochs; a second 24-epoch seed is untested.
set -euo pipefail
sed -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "xattn")|' \
    -e 's|os.environ.get("FOLDS", "0")|os.environ.get("FOLDS", "0,1")|' \
    -e 's|os.environ.get("EPOCHS", "10")|os.environ.get("EPOCHS", "24")|' \
    -e 's|os.environ.get("SEED", "0")|os.environ.get("SEED", "1")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("HEAD", "xattn")' "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "0,1")' "$2/script.py" || { echo "FOLDS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("EPOCHS", "24")' "$2/script.py" || { echo "EPOCHS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("SEED", "1")'    "$2/script.py" || { echo "SEED patch missed" >&2; exit 1; }
grep -q 'take_group'                     "$2/script.py" || { echo "multi-group cache needs group sampling" >&2; exit 1; }
grep -q 'xm.mark_step()'                 "$2/script.py" || { echo "XLA needs mark_step" >&2; exit 1; }
grep -q 'TrainingWatchdog'               "$2/script.py" || { echo "stall watchdog missed" >&2; exit 1; }
