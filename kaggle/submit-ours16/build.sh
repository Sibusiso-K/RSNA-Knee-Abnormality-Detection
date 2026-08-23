#!/usr/bin/env bash
# Sixteen members, ALL OURS. No public weights.
#
# Our fifteen have never been submitted on their own. The 0.864 that still
# stands is FIVE of them; mix35 ran all fifteen but diluted them to 60% with
# twenty public members that measure 0.839 through this pipeline, and landed
# 0.861. So the obvious ensemble is the one never actually tried.
#
#   family A  5x DINOv2-small seed 1   CV 0.8185
#   family B  5x DINOv2-base           CV 0.8108
#   family C  5x DINOv2-small seed 2   CV 0.8160
#   gattn     1x group-attention head  CV 0.8171 (fold 0)
#
# gattn is in despite returning a null against xattn (0.8171 vs 0.8207, inside
# the +/-0.011 band). A null on ACCURACY is not a null on DIVERSITY: it reads
# the slice axis through one attention rather than through averaged logits, so
# it is wrong in different places, and rank-mean pays for exactly that. One
# member in sixteen is a cheap way to find out, and its own score says it will
# not drag the blend.
#
# Uniform weights: every member is six-slice, so GRID_WEIGHT is constant here
# and the 2:1 tilt that mattered for mix35 does nothing.
set -euo pipefail
cp "$1/notebooks/kaggle_08_submit_mixed.py" "$2/script.py"

grep -q 'build_study_multi'          "$2/script.py" || { echo "must build one cache per grid" >&2; exit 1; }
grep -q 'xk = x\[slices\]'           "$2/script.py" || { echo "each member must read its OWN grid" >&2; exit 1; }
grep -q 'shared-decode check FAILED' "$2/script.py" || { echo "union decode must be proven against build_study" >&2; exit 1; }
grep -q 'rank(pct=True)'             "$2/script.py" || { echo "members must combine by rank" >&2; exit 1; }
grep -q 'still the 0.5 default'      "$2/script.py" || { echo "constant-submission tripwire missing" >&2; exit 1; }
grep -q 'refuse_reason'              "$2/script.py" || { echo "fingerprint gate missing" >&2; exit 1; }
grep -q 'os.walk(INPUT)'             "$2/script.py" || { echo "checkpoint search must prune test_series" >&2; exit 1; }

# gattn takes the whole slice axis in ONE pass. Slicing a triplet out for it
# would run, load, and score a different model than the one that was trained -
# the same class of silent-wrong-input bug as feeding champ 336px.
grep -q 'if head == "gattn"'         "$2/script.py" || { echo "gattn must not be fed one group at a time" >&2; exit 1; }
grep -q 'for g in range(n_groups)'   "$2/script.py" || { echo "non-gattn members must still average over groups" >&2; exit 1; }

python -m pytest -q "$1/tests/test_members.py" "$1/tests/test_groupattn.py" >/dev/null \
  || { echo "member/head tests fail" >&2; exit 1; }
