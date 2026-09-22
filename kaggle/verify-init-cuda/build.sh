#!/usr/bin/env bash
# docs/23: CUDA-side half of the init-weight-fingerprint check. Constructs
# the small/xattn model for folds 0 and 1 with the SAME per-fold seeding as
# the CUDA training pilot, logs a SHA-256 fingerprint of the freshly
# constructed (untrained) weights, then exits - no optimizer, no data
# beyond what construction needs. Compare the logged fingerprints against
# kaggle/verify-init-tpu's for the same folds; the pair kaggle/train-ema-pilot
# used for the docs/20/22 job is REPLACED here so this pilot's own weights
# are checked before spending GPU time training them.
set -euo pipefail
sed -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "xattn")|' \
    -e 's|os.environ.get("FOLDS", "0")|os.environ.get("FOLDS", "0,1")|' \
    -e 's|os.environ.get("PER_FOLD_SEED", "0")|os.environ.get("PER_FOLD_SEED", "1")|' \
    -e 's|os.environ.get("VERIFY_INIT_ONLY", "0")|os.environ.get("VERIFY_INIT_ONLY", "1")|' \
    -e 's|os.environ.get("USE_XLA", "auto")|"0"|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("HEAD", "xattn")'            "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "0,1")'              "$2/script.py" || { echo "FOLDS patch missed" >&2; exit 1; }
grep -q 'os.environ.get("PER_FOLD_SEED", "1")'        "$2/script.py" || { echo "PER_FOLD_SEED patch missed" >&2; exit 1; }
grep -q 'os.environ.get("VERIFY_INIT_ONLY", "1")'     "$2/script.py" || { echo "VERIFY_INIT_ONLY patch missed" >&2; exit 1; }
grep -q 'if "0" != "0":'                              "$2/script.py" || { echo "USE_XLA patch missed" >&2; exit 1; }
grep -q 'weight_fingerprint'                          "$2/script.py" || { echo "fingerprint helper missed" >&2; exit 1; }
