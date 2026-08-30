#!/usr/bin/env bash
# gattn head, fold 0 only, EPOCHS 10 -> 32.
#
# Revised from a 24-epoch schedule after train-6slice-24ep's result came in:
# LB 0.887 (+0.023 over the 0.864 standing best) from a run whose OWN CV gain
# (+0.0076) sat below the project's +0.02 null bar. The CV yardstick is
# graded against labels_blend_v1, whose own measured ceiling against real
# gold labels is 0.8930 - it cannot see gains that fall in the gap between
# "matches the noisy label" and "matches the real finding". So the CV bar is
# a good filter against illusory gains, but not a reliable stop signal for
# genuine ones once a lever is this close to the label-noise floor.
#
# Per-fold epoch usage from that run also argued for MORE headroom, not less:
# folds 1/2/4 peaked early (15, 17, 17 of 24) but folds 0 and 3 were still
# using epoch 22 and 21 of 24 - close enough to the schedule boundary that
# they may not have finished improving. 32 gives those late folds real room
# without more than doubling the original 10-epoch cost.
#
# THE PRE-COMMITTED BAR, stated before the run and honoured after it:
#
#   fold 0, gattn head, 6 slices, 10 epochs  ->  0.8171  (measured null vs xattn's 0.8207)
#   fold noise is +/- 0.011
#   32ep counts as a real CV gain over the 10ep gattn number at  >= 0.8371  (+0.02)
#   BUT per the reasoning above, a CV number inside that band is no longer
#   grounds alone to skip testing on the LB - submit it and read the real
#   number, the way train-6slice-24ep should have been read from the start.
#
# Why retry a null: the undertraining diagnosis (checkpoints peaking at the
# last epoch of 10) was measured on xattn and slot heads, never on gattn.
# gattn attends over every slice group in one pass instead of averaging
# per-group logits, which means more computation per sample and potentially
# a slower-converging head - exactly the profile that would be starved by a
# 10-epoch schedule even if xattn was not. Fold 0 only, same reasoning as the
# original gattn run: five folds of a head that turns out flat would spend
# TPU budget re-learning what one fold already shows.
set -euo pipefail
sed -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "gattn")|' \
    -e 's|os.environ.get("BATCH", "8")|os.environ.get("BATCH", "4")|' \
    -e 's|os.environ.get("EPOCHS", "10")|os.environ.get("EPOCHS", "32")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("HEAD", "gattn")'  "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("BATCH", "4")'     "$2/script.py" || { echo "batch patch missed - 96 encoder images/step will OOM" >&2; exit 1; }
grep -q 'os.environ.get("EPOCHS", "32")'   "$2/script.py" || { echo "EPOCHS patch missed" >&2; exit 1; }
grep -q 'head=HEAD'                        "$2/script.py" || { echo "SlotNet must be built from HEAD, not a hardcoded default" >&2; exit 1; }
grep -q 'ALL_GROUPS = HEAD == "gattn"'     "$2/script.py" || { echo "gattn must switch the data path too" >&2; exit 1; }
grep -q 'take_input'                       "$2/script.py" || { echo "training must feed all groups for gattn" >&2; exit 1; }
grep -q 'xm.mark_step()'                   "$2/script.py" || { echo "XLA needs mark_step" >&2; exit 1; }

python -m pytest -q "$1/tests/test_groupattn.py" >/dev/null || { echo "group-attention head tests fail" >&2; exit 1; }
