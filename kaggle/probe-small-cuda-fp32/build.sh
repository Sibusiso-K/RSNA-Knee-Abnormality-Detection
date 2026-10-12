#!/usr/bin/env bash
# docs/25: 50-step timing probe for the fp32 fold-0 diagnostic, full 24-epoch
# schedule. Launch before the full run to project fp32 wall time against the
# 12 h GPU session cap; fp32 on a T4 is expected to be several times slower
# than fp16.
set -euo pipefail
bash "$1/kaggle/train-small-cuda-fp32-f0/build.sh" "$1" "$2"
sed 's|os.environ.get("MAX_STEPS", "0")|os.environ.get("MAX_STEPS", "50")|' "$2/script.py" > "$2/probe.tmp"
mv "$2/probe.tmp" "$2/script.py"
grep -q 'os.environ.get("MAX_STEPS", "50")' "$2/script.py"
grep -q 'os.environ.get("VERIFY_INIT_ONLY", "0")' "$2/script.py"
grep -q 'os.environ.get("CUDA_AMP", "0")' "$2/script.py"
