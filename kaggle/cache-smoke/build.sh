#!/usr/bin/env bash
# Smoke test for the cache build: 300 studies, CPU, no GPU quota.
#
# What this is checking, and why it is worth a run of its own:
#
#   1. Laterality actually resolves. If `unknown` dominates, the geometry read
#      is wrong and the mirror is a no-op -- which would look like a successful
#      run while silently leaving the defect in place.
#   2. The 130 mm crop fits. If it mostly falls back, one output pixel is not a
#      constant number of millimetres and the resolution fix is fiction.
#   3. Slot fill rates are plausible. Six slots is a claim about this corpus;
#      if COR_T1 is empty everywhere then the weighting classifier is wrong.
#   4. Throughput. The full run is 4,407 studies against a 12 h cap, so the
#      per-study rate measured here is what decides whether it needs sharding.
#
# None of the four can be checked from a laptop -- the DICOMs only exist on
# Kaggle. Spending 10 minutes of free CPU to avoid a wasted 6 h run is the
# trade this project has already got wrong twice.
set -euo pipefail
sed -e 's|os.environ.get("LIMIT", "0")|os.environ.get("LIMIT", "300")|' \
    "$1/notebooks/kaggle_05_cache.py" > "$2/script.py"
grep -q 'os.environ.get("LIMIT", "300")' "$2/script.py" \
  || { echo "LIMIT patch missed" >&2; exit 1; }
