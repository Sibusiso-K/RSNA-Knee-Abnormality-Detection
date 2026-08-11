#!/usr/bin/env bash
# Inference + submission for the slot pipeline. GPU, internet OFF.
#
# Internet OFF is not optional here - competition submissions run offline, so
# anything the notebook needs must be mounted. That is why DINOv2 arrives via
# model_sources and the checkpoints via a Dataset.
#
# The test cache is built inside the notebook through src/data/cache.py, the
# same module that built the training cache. The asserts below are the guard
# against the failure that would be invisible otherwise: a submission computed
# from correctly-shaped but differently-preprocessed pixels scores badly and
# looks exactly like a model that did not work.
set -euo pipefail
cp "$1/notebooks/kaggle_07_submit_slots.py" "$2/script.py"

grep -q 'from src.data.cache import' "$2/script.py" || { echo "must build the test cache through the SHARED decode module, never a local copy" >&2; exit 1; }
grep -q 'submission.to_csv("submission.csv", index=False)' "$2/script.py" || { echo "must write submission.csv" >&2; exit 1; }
grep -q 'write_and_exit' "$2/script.py" || { echo "degraded-mode fallback missing: a raising notebook burns a submission slot for nothing" >&2; exit 1; }
grep -q 'still the 0.5 default' "$2/script.py" || { echo "missing the constant-submission tripwire" >&2; exit 1; }
