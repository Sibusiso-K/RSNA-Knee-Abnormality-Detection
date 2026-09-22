#!/usr/bin/env bash
# One fold, 50 steps, full 24-epoch LR schedule; launch only after init parity.
set -euo pipefail
bash "$1/kaggle/train-small-cuda-pilot-f0/build.sh" "$1" "$2"
sed 's|os.environ.get("MAX_STEPS", "0")|os.environ.get("MAX_STEPS", "50")|' "$2/script.py" > "$2/probe.tmp"
mv "$2/probe.tmp" "$2/script.py"
grep -q 'os.environ.get("MAX_STEPS", "50")' "$2/script.py"
grep -q 'os.environ.get("VERIFY_INIT_ONLY", "0")' "$2/script.py"
