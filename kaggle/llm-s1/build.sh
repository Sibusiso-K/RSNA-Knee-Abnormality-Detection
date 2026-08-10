#!/usr/bin/env bash
# Shard 1 of 3. Env vars cannot be set on a Kaggle kernel, so they are baked
# into the script here rather than configured at run time.
set -euo pipefail
{ echo 'import os'; echo 'os.environ["N_SHARDS"] = "3"'; echo 'os.environ["SHARD"] = "1"';   cat "$1/notebooks/kaggle_04_llm_labels.py"; } > "$2/script.py"
