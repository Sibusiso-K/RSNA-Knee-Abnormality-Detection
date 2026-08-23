#!/usr/bin/env bash
# Are we label-limited? A SENSITIVITY probe, deliberately using WORSE labels.
#
# Six levers have now returned nothing: epochs (x3), encoder size, pixel
# density, slices past six, the aggregation head, and ensemble breadth
# (5 -> 16 members moved the LB by 0.000). A score insensitive to all of those
# suggests the targets are the ceiling. But that is an inference, and this
# project has been wrong about inferences twice this week.
#
# It cannot be tested the obvious way. There is no better label set: v2 scores
# 0.8935 against the 58 gold studies and v1 scores 0.8930, and 0.0005 on 58
# studies is nothing. So run it in the other direction and measure the SLOPE.
#
#   labels_blend_v1.csv   0.8930 vs gold   -> fold 0 CV 0.8207  (known)
#   labels_v1.csv         0.7565 vs gold   -> fold 0 CV ???     (this run)
#
# A 0.137 drop in label quality is enormous. What the model does with it is the
# whole question, and either answer is worth the 40 minutes:
#
#   CV falls a LOT (say below ~0.78)  -> the model tracks its targets closely,
#       labels ARE the binding constraint, and better labels are the only thing
#       worth building next.
#   CV barely moves (above ~0.81)     -> the model is NOT reading the labels
#       that finely. Label quality is not the ceiling, I should stop saying it
#       is, and the constraint is somewhere neither the architecture nor the
#       targets have accounted for.
#
# Note the direction of the bar: this run SUCCEEDS at explaining things whether
# the number is high or low. It is the first experiment here in a while that
# cannot come back null.
set -euo pipefail
sed -e 's|LABEL_CANDIDATES = ("labels_blend_v1.csv", "llm_labels_v4_blend.csv", "labels_v1.csv")|LABEL_CANDIDATES = ("labels_v1.csv",)|' \
    -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "xattn")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'LABEL_CANDIDATES = ("labels_v1.csv",)' "$2/script.py" || { echo "label patch missed - would silently train on blend_v1 and measure nothing" >&2; exit 1; }
grep -q 'os.environ.get("HEAD", "xattn")'       "$2/script.py" || { echo "must match the 0.8207 baseline head, or two variables move at once" >&2; exit 1; }
grep -q 'take_group'                            "$2/script.py" || { echo "xattn samples one group per step" >&2; exit 1; }
grep -q 'xm.mark_step()'                        "$2/script.py" || { echo "XLA needs mark_step" >&2; exit 1; }
