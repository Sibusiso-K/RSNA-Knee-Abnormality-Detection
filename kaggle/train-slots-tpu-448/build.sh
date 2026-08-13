#!/usr/bin/env bash
# Fold 0 at 448 px instead of 336. One variable: pixel density.
#
# 130 mm / 448 = 0.290 mm/px against 0.387. Resolution is the representation
# axis already shown to matter here - the original pipeline sat at 0.714 mm/px,
# which is why its 160 mm crop experiment returned 0.002.
#
# SIZE MUST BE PATCHED, and this is the trap the build exists to close.
# kaggle_06_train_slots.py defaults SIZE to IMG, which src/data/slots.py still
# defines as 336, and SlotNet.forward INTERPOLATES whenever img_size differs
# from the tensor. Left alone this run would faithfully downsample every 448 px
# frame back to 336 and measure precisely nothing - while completing, printing
# a plausible AUC, and looking like evidence that resolution does not help.
#
# Attention is quadratic in tokens: 448 gives 1025 tokens against 577, so ~3.2x
# the attention memory. base at 336 fitted 15.75 GB of HBM; small at 448 may not
# fit at batch 8. An OOM surfaces at the first compile, ~2 min in.
#
# Compare against small fold 0 @336 = 0.7990. Fold noise is +/-0.011, so this
# needs >= ~0.02 to be believed.
set -euo pipefail
sed -e 's|os.environ.get("SIZE", str(IMG))|os.environ.get("SIZE", "448")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("SIZE", "448")' "$2/script.py" || { echo "SIZE patch missed - the run would silently downsample 448 back to 336" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "0")' "$2/script.py" || { echo "keep FOLDS=0: one fold, one variable" >&2; exit 1; }
grep -q 'xm.mark_step()' "$2/script.py" || { echo "XLA needs an explicit mark_step" >&2; exit 1; }
python - <<'PY'
assert 448 % 14 == 0, "DINOv2 is patch-14"
print(f"[check] 448 = 14 x {448 // 14} tokens/side, {(448 // 14) ** 2 + 1} tokens total")
PY
