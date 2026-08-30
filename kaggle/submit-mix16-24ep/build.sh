#!/usr/bin/env bash
# Sixteen members, ALL OURS, family A upgraded to the 24-epoch retrain.
#
# knee-submit-ours16 (unmodified family A, CV 0.8185) scored 0.864 standalone
# for the FIVE-member subset, matching the standing best. knee-submit-6slice-24ep
# (family A alone, retrained EPOCHS 10 -> 24, CV 0.8261) scored 0.887 standalone
# on the LB - a real gain that the CV number's own +0.0076 undersold. This
# config asks whether blending the OLD, weaker-but-differently-wrong families
# B/C/gattn on top of the STRONGER family A still helps, the way it helped at
# 0.8185.
#
#   family A  5x DINOv2-small seed 1, 24 EPOCHS   CV 0.8261 (LB 0.887 alone)
#   family B  5x DINOv2-base, 10 epochs           CV 0.8108
#   family C  5x DINOv2-small seed 2, 10 epochs   CV 0.8160
#   gattn     1x group-attention head, 10 epochs  CV 0.8171 (fold 0)
#
# B/C/gattn are unretrained - only the dataset_sources entry for family A
# changed (knee-slot-6slice-v1 -> knee-slot-6slice-24ep-v1). If B/C/gattn's
# now-larger accuracy gap to family A hurts more than their diversity helps,
# this will score BELOW 0.887, not just below some larger blend - that
# comparison is the point of testing it standalone first.
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
