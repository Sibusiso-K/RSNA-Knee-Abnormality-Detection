#!/usr/bin/env bash
# Probe the TPU before committing any of the 20 h/week to a real run.
# Answers: does grid_sample lower, does topk lower, what is the step time,
# and does one core stand any chance inside the weekly budget.
set -euo pipefail
cp "$1/notebooks/kaggle_09_tpu_probe.py" "$2/script.py"
grep -q 'xm.optimizer_step' "$2/script.py" || { echo "must use xm.optimizer_step, not optimizer.step" >&2; exit 1; }
