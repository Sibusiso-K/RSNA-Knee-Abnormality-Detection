#!/usr/bin/env bash
# Six slices per slot instead of three, with the xattn head. Tests COVERAGE.
#
# The last untested representation axis. Three samples across the 20-80% band
# sit 6-14 slices apart while a meniscal tear spans 2-3 consecutive slices, so
# focal evidence can fall entirely between them - and the per-label ordering
# runs almost exactly diffuse (Baker's .845) to focal (PF OA .711).
#
# Paired with the xattn head deliberately: extra slices are only useful if the
# head can attend to them, and the pooled head averages them away by
# construction. Baseline is therefore xattn @ 3 slices = 0.8035, not 0.7990.
#
# Only the 6-slice cache is mounted; both caches contain index_train_0.csv and
# find_dir resolves by content.
set -euo pipefail
sed -e 's|SlotNet(dinov2, unfreeze_last=UNFREEZE_LAST, pool=POOL)|SlotNet(dinov2, unfreeze_last=UNFREEZE_LAST, pool=POOL, head="xattn")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"
grep -q 'head="xattn"'                "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "0")' "$2/script.py" || { echo "keep FOLDS=0" >&2; exit 1; }
grep -q 'xm.mark_step()'               "$2/script.py" || { echo "XLA needs mark_step" >&2; exit 1; }
