#!/usr/bin/env bash
# All five folds, 6-slice cache, xattn head - a SECOND ensemble family.
#
# Coverage is the one lever that worked. Fold 0, everything else held:
#   slot head, 3 slices   0.7990
#   xattn head, 3 slices  0.8035   (+0.0045)
#   xattn head, 6 slices  0.8179   (+0.0144)
# and the per-label pattern confirms the mechanism rather than just the number:
# Fracture +0.068, PF OA +0.049, Medial Meniscus +0.049 - the FOCAL findings -
# against Effusion +0.007 and Synovitis +0.011, the diffuse ones. Evidence was
# falling between samples spaced 6-14 slices apart; halving that spacing to 2-5
# recovers it.
#
# Five folds at ~40 min each is ~3.3 h of the remaining TPU budget, and yields
# five members that each beat every previous member.
set -euo pipefail
sed -e 's|SlotNet(dinov2, unfreeze_last=UNFREEZE_LAST, pool=POOL)|SlotNet(dinov2, unfreeze_last=UNFREEZE_LAST, pool=POOL, head="xattn")|' \
    -e 's|os.environ.get("FOLDS", "0")|os.environ.get("FOLDS", "0,1,2,3,4")|' \
    "$1/notebooks/kaggle_06_train_slots.py" > "$2/script.py"

grep -q 'head="xattn"'                          "$2/script.py" || { echo "head patch missed" >&2; exit 1; }
grep -q 'os.environ.get("FOLDS", "0,1,2,3,4")'  "$2/script.py" || { echo "FOLDS patch missed" >&2; exit 1; }
grep -q 'take_group'                            "$2/script.py" || { echo "multi-group cache needs group sampling" >&2; exit 1; }
grep -q 'xm.mark_step()'                        "$2/script.py" || { echo "XLA needs mark_step" >&2; exit 1; }
