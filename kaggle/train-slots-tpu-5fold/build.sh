#!/usr/bin/env bash
# All five folds in ONE kernel, 10 epochs each.
#
# Why an ensemble is not an optimisation here: the public 0.899 notebook is a
# rank-mean of several PRE-TRAINED members restored from a published weights
# dataset (find_weights -> manifest.json -> members -> load_state_dict), not a
# single model. Comparing our single fold-0 against it was never like for like.
#
# Why one kernel rather than five: TPU time is billed as wall clock and
# mounting the 9 GB cache costs minutes every launch. Five separate kernels pay
# that five times.
#
# Why 10 epochs and not 20: measured. 20 gave 0.7964 against 0.7948, inside
# noise, while loss kept falling - the curve plateaus by epoch ~5 and the rest
# is overfitting.
#
# Cost: ~30 min/fold, so ~2.5 h of a 20 h/week budget.
set -euo pipefail
sed -e 's|os.environ.get("FOLDS", "0")|os.environ.get("FOLDS", "0,1,2,3,4")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"
grep -q 'os.environ.get("FOLDS", "0,1,2,3,4")' "$2/script.py" || { echo "FOLDS patch missed" >&2; exit 1; }
grep -q 'xm.mark_step()' "$2/script.py" || { echo "XLA needs an explicit mark_step or the graph grows until it OOMs" >&2; exit 1; }
grep -q 'grouped_folds' "$2/script.py" || { echo "must use grouped folds" >&2; exit 1; }
grep -q 'gold_uids' "$2/script.py" || { echo "the 58 gold studies must be held out" >&2; exit 1; }
