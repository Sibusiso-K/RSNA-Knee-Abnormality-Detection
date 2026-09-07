#!/usr/bin/env bash
set -euo pipefail
python "$1/kaggle/submit-fresh10-24ep/build.py" "$2"
python -m pytest -q "$1/tests/test_members.py"
