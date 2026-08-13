#!/usr/bin/env bash
# 12 slices per slot, shard 0 of 2. Free CPU, no accelerator quota.
#
# Coverage is the lever that worked: 3 -> 6 slices gave +0.0144 on fold 0, and
# the gain landed on the FOCAL findings (Fracture +0.068, PF OA +0.049, Medial
# Meniscus +0.049) exactly as the slice-spacing argument predicted. 12 slices
# halves the spacing again.
#
# Sharded by STUDY because 12 slices is 33.4 GB total, over the 20 GB per-kernel
# output limit; each shard holds all 12 slices for half the studies at 16.7 GB.
# Training concatenates both.
#
# THREE names must be rebound, all bound at import time:
#   N_GROUP (slots) -> N_SLICE (cache) -> N_SLICE (script)
set -euo pipefail
sed -e 's|os.environ.get("SHARD", "0")|os.environ.get("SHARD", "0")|'     -e 's|os.environ.get("N_SHARD", "1")|os.environ.get("N_SHARD", "2")|'     "$1/notebooks/kaggle_05_cache.py" > "$2/script.py"

python - "$2/script.py" <<'PY'
import sys
path = sys.argv[1]
src = open(path).read()
marker = "from src.data.slots import IMG, N_SLOT, SLOTS        # noqa: E402"
assert marker in src, "import line moved"
src = src.replace(marker, marker + """

# --- 12-slices-per-slot override ----------------------------------------
import src.data.cache as _cache                              # noqa: E402
import src.data.slots as _slots                              # noqa: E402

_slots.N_GROUP = 4                     # GROUP(3) x N_GROUP(4) = 12 slices
N_SLICE = _slots.GROUP * _slots.N_GROUP
_cache.N_SLICE = N_SLICE
assert _cache.N_SLICE == N_SLICE == 12, "N_SLICE override failed"
print(f"[cfg] N_SLICE overridden to {N_SLICE}", flush=True)
""")
open(path, "w").write(src)
PY

grep -q 'os.environ.get("SHARD", "0")'   "$2/script.py" || { echo "SHARD patch missed" >&2; exit 1; }
grep -q 'os.environ.get("N_SHARD", "2")' "$2/script.py" || { echo "N_SHARD patch missed" >&2; exit 1; }
grep -q '_cache.N_SLICE = N_SLICE'       "$2/script.py" || { echo "N_SLICE override missed" >&2; exit 1; }
grep -q 'open_memmap'                    "$2/script.py" || { echo "memmap required at this size" >&2; exit 1; }
