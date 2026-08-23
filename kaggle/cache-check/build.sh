#!/usr/bin/env bash
# Audit the built cache on free CPU before spending GPU quota on it.
# Checks the four things that would make a training run meaningless while
# still printing a plausible AUC. See the notebook docstring.
set -euo pipefail
cp "$1/notebooks/kaggle_08_cache_check.py" "$2/script.py"
grep -q 'check_grouping' "$2/script.py" || { echo "must check grouping" >&2; exit 1; }
