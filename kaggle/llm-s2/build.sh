#!/usr/bin/env bash
# Shard 2 of 3.  Batch 2 avoids the one observed CUDA OOM from shard 0.
set -euo pipefail
{ echo 'import os'; echo 'os.environ["N_SHARDS"] = "3"'; echo 'os.environ["SHARD"] = "2"'; \
  sed 's/^BATCH = 4$/BATCH = 2/' "$1/notebooks/kaggle_04_llm_labels.py"; } > "$2/script.py"
grep -q '^BATCH = 2$' "$2/script.py" || { echo 'batch safety patch missed' >&2; exit 1; }
