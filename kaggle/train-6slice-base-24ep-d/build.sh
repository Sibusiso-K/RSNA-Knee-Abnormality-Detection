#!/usr/bin/env bash
# DINOv2-base family, fold 2 ONLY - a retry.
#
# train-6slice-base-24ep-b (folds 1,2) finished fold 1 cleanly (best CV
# 0.8136) then stalled ~150 steps into fold 2's first epoch; the watchdog
# caught it within 141s (limit 120s) instead of burning the rest of the
# session, exactly as designed. This retries fold 2 alone - if it stalls
# again in the same place, that is a real, reproducible bug in fold 2's
# data/scheduling rather than a one-off; if it completes, it was transient.
set -euo pipefail
sed -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "xattn")|' \
    -e 's|os.environ.get("VARIANT", "small")|os.environ.get("VARIANT", "base")|' \
    -e 's|os.environ.get("FOLDS", "0")|os.environ.get("FOLDS", "2")|' \
    -e 's|os.environ.get("EPOCHS", "10")|os.environ.get("EPOCHS", "24")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("HEAD", "xattn")'   "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("VARIANT", "base")' "$2/script.py" || { echo "variant patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "2")'      "$2/script.py" || { echo "FOLDS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("EPOCHS", "24")'    "$2/script.py" || { echo "EPOCHS patch missed" >&2; exit 1; }
grep -q 'take_group'                        "$2/script.py" || { echo "multi-group cache needs group sampling" >&2; exit 1; }
grep -q 'xm.mark_step()'                    "$2/script.py" || { echo "XLA needs mark_step" >&2; exit 1; }
grep -q '_watchdog'                         "$2/script.py" || { echo "stall watchdog missed" >&2; exit 1; }
