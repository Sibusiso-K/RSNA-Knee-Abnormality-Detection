#!/usr/bin/env bash
# Same run, 20 epochs instead of 10. One variable.
#
# Run 1 finished at CV 0.7948 / gold58 0.8105 and was STILL RISING at the last
# epoch (CV 0.7854 -> 0.7948, gold58 0.8028 -> 0.8105) with loss still falling.
# Under OneCycleLR that is a converged 10-epoch schedule, not an interrupted
# one, so the question "would a longer schedule reach higher" is genuinely
# open and cannot be answered by reading run 1's curve.
#
# It is worth asking here specifically because the OLD pipeline answered it the
# other way: 8 epochs scored no better than 4, with loss falling 61% while AUC
# declined - overfitting to noisy labels. The labels are now 0.8930 against
# gold rather than 0.7565, so the regime may simply be different.
#
# Compare END-OF-SCHEDULE numbers only. OneCycleLR stretches to whatever EPOCHS
# is, so epoch 9 of this run is mid-anneal and epoch 9 of run 1 was finished.
set -euo pipefail
sed -e 's|os.environ.get("EPOCHS", "10")|os.environ.get("EPOCHS", "20")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"
grep -q 'os.environ.get("EPOCHS", "20")' "$2/script.py" || { echo "EPOCHS patch missed" >&2; exit 1; }
grep -q 'xm.mark_step()' "$2/script.py" || { echo "XLA needs an explicit mark_step or the graph grows until it OOMs" >&2; exit 1; }
grep -q 'max_lr=\[LR_BACKBONE, LR_HEAD\]' "$2/script.py" || { echo "max_lr must stay a per-group list" >&2; exit 1; }
