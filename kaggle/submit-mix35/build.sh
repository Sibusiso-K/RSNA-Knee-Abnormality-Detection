#!/usr/bin/env bash
# 35-member mixed-grid submission: our 15 six-slice members + the 20 public
# twelve-slice ones, each scored on the grid it was trained on.
#
# No source patching here, unlike submit-6slice/ and submit-public25/. Those
# rebind N_SLICE globally because their notebook builds one cache for everyone;
# this notebook takes the grid from each checkpoint instead, so there is
# nothing to override and no chance of an override silently missing.
set -euo pipefail
cp "$1/notebooks/kaggle_08_submit_mixed.py" "$2/script.py"

# The guards that matter, all of them earned by a bug that shipped:
grep -q 'build_study_multi'          "$2/script.py" || { echo "must build one cache per grid" >&2; exit 1; }
grep -q 'xk = x\[slices\]'           "$2/script.py" || { echo "each member must read its OWN grid" >&2; exit 1; }
grep -q 'shared-decode check FAILED' "$2/script.py" || { echo "union decode must be proven against build_study" >&2; exit 1; }
grep -q 'for g in range(n_groups)'   "$2/script.py" || { echo "inference must average over groups" >&2; exit 1; }
grep -q 'rank(pct=True)'             "$2/script.py" || { echo "members must combine by rank" >&2; exit 1; }
grep -q 'still the 0.5 default'      "$2/script.py" || { echo "constant-submission tripwire missing" >&2; exit 1; }
grep -q 'head.cross_attn'            "$2/script.py" || { echo "head must be inferred from the weights" >&2; exit 1; }
grep -q 'trained at {img}px'         "$2/script.py" || { echo "fingerprint gate missing - champ members would be fed 336px" >&2; exit 1; }
