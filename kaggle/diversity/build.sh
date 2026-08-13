#!/usr/bin/env bash
# Does a second architecture add anything, measured before paying for it.
#
# Three capacity levers returned nothing (8-vs-4 epochs, 20-vs-10, base-vs-
# small at -0.0005). The remaining hypothesis is that the public 0.899 wins on
# DIVERSITY - it rank-means several members while ours is five folds of one
# config. Training base on all five folds to test that would cost ~3.5 h of a
# 20 h budget on a hunch; small-fold0 and base-fold0 share a validation split,
# so combining their predictions answers it in ~20 min instead.
#
# BOTH encoder widths are mounted here on purpose: the probe loads each
# checkpoint against the DINOv2 whose hidden_size matches it.
set -euo pipefail
cp "$1/notebooks/kaggle_10_diversity.py" "$2/script.py"
grep -q 'rank(pct=True)' "$2/script.py" || { echo "must combine by rank, matching the submission path" >&2; exit 1; }
grep -q 'grouped_folds' "$2/script.py" || { echo "fold 0 must be reconstructed the same way training defined it" >&2; exit 1; }
