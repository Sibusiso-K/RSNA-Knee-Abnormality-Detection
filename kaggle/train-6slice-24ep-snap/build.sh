#!/usr/bin/env bash
# The unchanged small/xattn/6-slice/336px/24-epoch control from
# train-6slice-24ep - EXACT same backend, loss and schedule - but the shared
# training script (kaggle_06_train_slots.py) now snapshots epochs 16/20/24
# unconditionally alongside the existing best-on-TPU checkpoint.
#
# This is experiment 1 of docs/15-claude-score-improvement-strategy.md:
# TPU (training) validation and CUDA (deployment) validation measurably
# disagree on this architecture - a confirmed, reproducible backend delta,
# not yet a proven bug - and it is unknown whether they even RANK epochs the
# same way. Best-only checkpointing overwrites every earlier epoch's
# weights, so answering that requires re-running with snapshots kept, not
# recovering anything from the existing best-only files. A fresh run rather
# than reusing train-6slice-24ep's own checkpoints so the seed is recorded
# (SEED defaults to 0, newly added) and every snapshot's provenance is
# explicit.
set -euo pipefail
sed -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "xattn")|' \
    -e 's|os.environ.get("FOLDS", "0")|os.environ.get("FOLDS", "0,1,2,3,4")|' \
    -e 's|os.environ.get("EPOCHS", "10")|os.environ.get("EPOCHS", "24")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("HEAD", "xattn")'      "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "0,1,2,3,4")' "$2/script.py" || { echo "FOLDS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("EPOCHS", "24")'       "$2/script.py" || { echo "EPOCHS patch missed" >&2; exit 1; }
grep -q 'take_group'                           "$2/script.py" || { echo "multi-group cache needs group sampling" >&2; exit 1; }
grep -q 'xm.mark_step()'                       "$2/script.py" || { echo "XLA needs mark_step" >&2; exit 1; }
grep -q 'SNAPSHOT_EPOCHS'                      "$2/script.py" || { echo "epoch-snapshot instrumentation missed" >&2; exit 1; }
grep -q '"seed": SEED'                         "$2/script.py" || { echo "seed recording missed" >&2; exit 1; }
