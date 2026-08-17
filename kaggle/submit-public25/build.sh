#!/usr/bin/env bash
# Submission for the 6-slice models. The TEST cache must match TRAINING.
#
# This is the train/test skew guard, and it is the whole reason this config
# exists separately. The models were trained on triplets drawn from a SIX-slice
# sampling grid; building a three-slice test cache would produce tensors of the
# right shape containing differently-sampled anatomy, score badly, and look
# exactly like a model that did not work.
#
# Same two-name rebind as the cache build: src/data/cache.py computes
# N_SLICE = GROUP * N_GROUP at import time and this script imports N_SLICE from
# it, so setting slots.N_GROUP alone rebinds neither.
set -euo pipefail
cp "$1/notebooks/kaggle_07_submit_slots.py" "$2/script.py"

python - "$2/script.py" <<'PY'
import sys
path = sys.argv[1]
src = open(path).read()
marker = "from src.data.slots import GROUP, IMG, N_SLOT        # noqa: E402"
assert marker in src, "import line moved"
src = src.replace(marker, marker + """

# --- 12-slices-per-slot override, matching the public members manifest --------------
import src.data.cache as _cache                              # noqa: E402
import src.data.slots as _slots                              # noqa: E402

_slots.N_GROUP = 4
N_SLICE = _slots.GROUP * _slots.N_GROUP
_cache.N_SLICE = N_SLICE
assert _cache.N_SLICE == N_SLICE == 12, "N_SLICE override failed"
print(f"[cfg] test cache N_SLICE = {N_SLICE}", flush=True)
""", 1)
open(path, "w").write(src)
PY

grep -q '_cache.N_SLICE = N_SLICE' "$2/script.py" || { echo "N_SLICE override missed - test cache would not match training" >&2; exit 1; }
grep -q 'for g in range(N_GROUPS)' "$2/script.py" || { echo "inference must average over groups" >&2; exit 1; }
grep -q 'rank(pct=True)'           "$2/script.py" || { echo "members must combine by rank" >&2; exit 1; }
grep -q 'still the 0.5 default'    "$2/script.py" || { echo "constant-submission tripwire missing" >&2; exit 1; }

grep -q "head.cross_attn" "$2/script.py" || { echo "submission must infer the head from the weights; a slot-head default silently fails to load xattn checkpoints" >&2; exit 1; }

grep -q "champ_fold" "$2/script.py" || { echo "champ members must be in the glob" >&2; exit 1; }
grep -q "encoder.module" "$2/script.py" || { echo "must handle the DataParallel encoder prefix" >&2; exit 1; }
