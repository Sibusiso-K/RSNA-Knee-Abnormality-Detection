#!/usr/bin/env bash
# Fold 0 with the group-attention head: the slice axis moved inside the model.
#
# THE PRE-COMMITTED BAR, stated before the run and honoured after it:
#
#   fold 0, xattn head, 6 slices  ->  0.8207   (family A, the 0.864 submission)
#   fold noise is +/- 0.011
#   gattn counts as a real gain at  >= 0.841   (+0.02, two standard errors)
#   0.821 - 0.840 is NOISE and will be reported as "no difference"
#
# Fold 0 only. Five folds of a head that turns out to be flat would burn ~3.3 h
# of a 20 h/week TPU budget to learn the same thing one fold learns in 40 min.
#
# Why this head, and why now. Coverage was the only lever that ever paid:
# 3 -> 6 slices gave +0.0236, concentrated on FOCAL findings (Fracture +0.068,
# PF OA +0.049) exactly as the slice-spacing hypothesis predicted. But 6 -> 12
# gave -0.005. That is not "twelve slices are too many"; it is the harness.
# Training samples ONE group of three per step and inference averages the
# logits, so more groups means more independent predictions to average, and a
# finding present in one group is diluted by the groups without it as fast as
# the extra coverage adds it. gattn attends over every group at once, so the
# query selects the slice group with the evidence instead of averaging it away.
#
# BATCH drops 8 -> 4 because the encoder now runs S*G images per study instead
# of S: at G=2 a batch of 8 would be 96 encoder images per step against the 48
# this TPU is known to hold. Halving the batch keeps the step identical in
# memory. Steps double, so expect ~2x the wall clock of a 6-slice xattn fold.
set -euo pipefail
sed -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "gattn")|' \
    -e 's|os.environ.get("BATCH", "8")|os.environ.get("BATCH", "4")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("HEAD", "gattn")'  "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("BATCH", "4")'     "$2/script.py" || { echo "batch patch missed - 96 encoder images/step will OOM" >&2; exit 1; }
grep -q 'head=HEAD'                        "$2/script.py" || { echo "SlotNet must be built from HEAD, not a hardcoded default" >&2; exit 1; }

# The pairing that makes this measure anything. Patching the model to gattn
# while leaving np.random.randint(N_GROUPS) sampling one triplet per step would
# run to completion, save a checkpoint, print a plausible AUC, and test nothing.
grep -q 'ALL_GROUPS = HEAD == "gattn"'     "$2/script.py" || { echo "gattn must switch the data path too" >&2; exit 1; }
grep -q 'take_input'                       "$2/script.py" || { echo "training must feed all groups for gattn" >&2; exit 1; }
grep -q 'xm.mark_step()'                   "$2/script.py" || { echo "XLA needs mark_step" >&2; exit 1; }

python -m pytest -q "$1/tests/test_groupattn.py" >/dev/null || { echo "group-attention head tests fail" >&2; exit 1; }
