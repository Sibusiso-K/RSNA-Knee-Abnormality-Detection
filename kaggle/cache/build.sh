#!/usr/bin/env bash
# Full cache build: all 4,407 training studies, CPU, no GPU quota.
#
# Sizing, measured on the 40-study smoke rather than assumed:
#   0.53 study/s  -> 4,407 studies in ~2.3 h, inside the 12 h CPU cap
#   2.03 MB/study -> ~8.95 GB, inside the 20 GB /kaggle/working limit
#
# So this runs as ONE shard. `SHARD`/`N_SHARD` stay wired up in the script
# because the margin on the 12 h cap is a measurement, not a guarantee, and a
# queue-slow run should cost a shard rather than the whole build.
#
# The script defaults are already the full-run values, so this build only has
# to assert them. That is on purpose: a build.sh that patches nothing cannot
# silently patch the wrong thing, and the greps below still fail loudly if the
# defaults drift.
set -euo pipefail
cp "$1/notebooks/kaggle_05_cache.py" "$2/script.py"

grep -q 'os.environ.get("LIMIT", "0")'   "$2/script.py" || { echo "LIMIT must default to 0 (no cap) for the full run" >&2; exit 1; }
grep -q 'os.environ.get("SPLIT", "train")' "$2/script.py" || { echo "SPLIT must default to train" >&2; exit 1; }
grep -q 'os.environ.get("N_SHARD", "1")' "$2/script.py" || { echo "N_SHARD must default to 1" >&2; exit 1; }
# The three constants the cache's whole value depends on. A cache written at
# the wrong resolution or with the wrong slot count is not a cache, it is a
# 9 GB dataset that has to be rebuilt from scratch.
grep -q '^CROP_MM = 130.0'               "$1/src/data/slots.py" || { echo "CROP_MM must be 130" >&2; exit 1; }
grep -q '^IMG = 336'                     "$1/src/data/slots.py" || { echo "IMG must be 336 (14x24, patch-14 safe)" >&2; exit 1; }
grep -q '^GROUP = 3'                     "$1/src/data/slots.py" || { echo "GROUP must be 3" >&2; exit 1; }
