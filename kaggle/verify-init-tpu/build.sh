#!/usr/bin/env bash
# docs/23: TPU-side half of the init-weight-fingerprint check - see
# kaggle/verify-init-cuda for what this establishes. Identical patches
# except the accelerator (metadata enable_tpu:true, USE_XLA left at its
# "auto" default so it takes the TPU path as normal training kernels do).
set -euo pipefail
sed -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "xattn")|' \
    -e 's|os.environ.get("FOLDS", "0")|os.environ.get("FOLDS", "0,1")|' \
    -e 's|os.environ.get("PER_FOLD_SEED", "0")|os.environ.get("PER_FOLD_SEED", "1")|' \
    -e 's|os.environ.get("VERIFY_INIT_ONLY", "0")|os.environ.get("VERIFY_INIT_ONLY", "1")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("HEAD", "xattn")'            "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "0,1")'              "$2/script.py" || { echo "FOLDS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("PER_FOLD_SEED", "1")'        "$2/script.py" || { echo "PER_FOLD_SEED patch missed" >&2; exit 1; }
grep -q 'os.environ.get("VERIFY_INIT_ONLY", "1")'     "$2/script.py" || { echo "VERIFY_INIT_ONLY patch missed" >&2; exit 1; }
grep -q 'weight_fingerprint'                          "$2/script.py" || { echo "fingerprint helper missed" >&2; exit 1; }
grep -q 'xm.mark_step()'                              "$2/script.py" || { echo "XLA needs mark_step" >&2; exit 1; }
