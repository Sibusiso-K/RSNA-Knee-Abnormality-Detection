#!/usr/bin/env bash
# DINOv2-base on folds 1-4, to fill out a second ensemble family.
#
# NOT because base is better - it is not, it measured -0.0005 against small on
# fold 0. Because base is DIFFERENT. Across five experiments the only positive
# signal was ensemble diversity: small+base rank-mean on one fold gave +0.0086
# with a per-label Spearman of 0.853, i.e. the two genuinely disagree.
#
# That was under the 0.010 bar for buying base x 5 as a deliberate spend. This
# is not that spend: the TPU is idle, the 6-slice cache is still building on
# CPU, GPU is ~35 h away, and these checkpoints drop straight into the
# submission that runs when it arrives. Idle quota does not accumulate.
#
# fold 0 already exists (knee-slot-base-v1), so only 1-4 are trained.
# ~42 min/fold at 448-free 336 px, so ~2.8 h of a 13 h remaining budget.
set -euo pipefail
sed -e 's|os.environ.get("FOLDS", "0")|os.environ.get("FOLDS", "1,2,3,4")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"
grep -q 'os.environ.get("FOLDS", "1,2,3,4")' "$2/script.py" || { echo "FOLDS patch missed" >&2; exit 1; }
grep -q 'xm.mark_step()' "$2/script.py" || { echo "XLA needs an explicit mark_step" >&2; exit 1; }
grep -q 'grouped_folds' "$2/script.py" || { echo "must use grouped folds" >&2; exit 1; }
grep -q 'gold_uids'     "$2/script.py" || { echo "the 58 gold studies must be held out" >&2; exit 1; }
