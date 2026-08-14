#!/usr/bin/env bash
# 12 slices per slot, fold 0. Same axis that just produced the only real gain.
#
#   3 slices  0.8035   (xattn head)
#   6 slices  0.8179   +0.0144 on fold 0, +0.0236 across five folds
#  12 slices  ?
#
# Both shards are mounted: 12 slices is 33.4 GB, over the 20 GB per-kernel
# output limit, so the cache was built as two study-halves. The training script
# already globs cache_train_*.npy and concatenates, and load_shards only skips
# the copy when there is exactly one - with two it concatenates, which is the
# correct behaviour here and costs the extra 33 GB of RAM the TPU VM has.
#
# Fold 0 only: TPU is down to ~4.7 h and the five-fold version belongs on the
# GPU budget that refreshes in ~19 h.
set -euo pipefail
sed -e 's|SlotNet(dinov2, unfreeze_last=UNFREEZE_LAST, pool=POOL)|SlotNet(dinov2, unfreeze_last=UNFREEZE_LAST, pool=POOL, head="xattn")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"
grep -q 'head="xattn"'                 "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "0")' "$2/script.py" || { echo "fold 0 only" >&2; exit 1; }
grep -q 'take_group'                   "$2/script.py" || { echo "multi-group cache needs group sampling" >&2; exit 1; }
