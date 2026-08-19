#!/usr/bin/env bash
# 448 px x 6 slices, xattn head, fold 0. Re-testing a null on purpose.
#
# 448 px was measured at fold-0 CV 0.7976 against a 0.7990 baseline and written
# down as "nothing". That is not being ignored — it is being re-run because the
# configuration it was measured under no longer exists.
#
# That run used the SLOT head, which mean-pools every patch token into ONE
# vector per slot before the classifier sees anything. Going 336 -> 448 raises
# the token count 577 -> 1025, so under a mean-pool all 448 extra tokens are
# averaged into the same single vector. The pixels were genuinely computed —
# step time rose 0.35s -> 0.56s, matching 1025 tokens against 577 — they simply
# had nowhere to go. XAttnHead cross-attends those tokens directly.
#
# The prior points the same way. The only lever that ever paid here was slice
# coverage, and it paid on the FOCAL findings (Fracture +0.068, PF OA +0.049)
# while the diffuse ones barely moved (Effusion +0.007). That is the signature
# of a model short of SPATIAL PRECISION, not of capacity — and 448 px is
# 0.290 mm/px against 0.387.
#
# THE BAR, stated before the run:
#   fold 0, xattn, 6 slices, 336 px  ->  0.8207
#   fold noise +/- 0.011
#   448 counts as a real gain at     ->  >= 0.841
#   0.821 - 0.840 is NOISE and gets reported as no difference.
#
# If this is null too it is the seventh, and the honest conclusion is that the
# representation is not what is holding the score at 0.864.
#
# BATCH 8 -> 4: 1025 tokens per image against 577 is 1.78x the activation, and
# a v5e-8 core already OOMed once on this model at 16.36G of 15.75G.
set -euo pipefail
sed -e 's|os.environ.get("HEAD", "slot")|os.environ.get("HEAD", "xattn")|' \
    -e 's|os.environ.get("BATCH", "8")|os.environ.get("BATCH", "4")|' \
    -e 's|os.environ.get("SIZE", str(IMG))|os.environ.get("SIZE", "448")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'os.environ.get("HEAD", "xattn")' "$2/script.py" || { echo "head patch missed - a slot head here would just repeat the 2026-08-12 null" >&2; exit 1; }
grep -q 'os.environ.get("BATCH", "4")'    "$2/script.py" || { echo "batch patch missed - 1025 tokens at batch 8 will OOM" >&2; exit 1; }
grep -q 'os.environ.get("SIZE", "448")'   "$2/script.py" || { echo "SIZE patch missed - the model would downscale 448 back to 336 and measure nothing" >&2; exit 1; }
grep -q 'take_group'                      "$2/script.py" || { echo "xattn samples one group per step" >&2; exit 1; }
grep -q 'xm.mark_step()'                  "$2/script.py" || { echo "XLA needs mark_step" >&2; exit 1; }
