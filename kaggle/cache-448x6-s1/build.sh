#!/usr/bin/env bash
# Shard 1 of 2: the 448 px x 6 slices cache.
#
# 448 px was measured as a null once — fold 0 CV 0.7976 against a 0.7990
# baseline. That measurement is not being ignored; it is being re-run because
# the configuration it was made under no longer exists.
#
# That run used the SLOT head, which pools every patch token into one vector
# per slot before the classifier sees anything. Raising the side 336 -> 448
# raises the token count 577 -> 1025, and under a mean-pool all 448 of those
# extra tokens are averaged into the same single vector. The resolution could
# not reach the head. Step time rose 0.35s -> 0.56s, so the pixels were
# genuinely there; they just had nowhere to go.
#
# XAttnHead cross-attends the patch tokens directly, so they are individually
# addressable now. And the one lever that ever paid here — slice coverage —
# paid on the FOCAL findings (Fracture +0.068, PF OA +0.049) while the diffuse
# ones barely moved, which is the signature of a model that is short of spatial
# precision rather than short of capacity.
#
# 29.7 GB over two shards = 14.8 GB each, inside the 20 GB working limit.
# CPU, so it costs nothing from the GPU or TPU budgets.
set -euo pipefail
sed -e 's|os.environ.get("SHARD", "0")|os.environ.get("SHARD", "1")|'     -e 's|os.environ.get("N_SHARD", "1")|os.environ.get("N_SHARD", "2")|'     "$1/notebooks/kaggle_05_cache.py" > "$2/script.py"

python - "$2/script.py" <<'PY'
import sys

path = sys.argv[1]
src = open(path).read()
marker = "from src.data.slots import IMG, N_SLOT, SLOTS        # noqa: E402"
assert marker in src, "import line moved; re-point the override"

override = marker + """

# --- 448 px x 6 slices, injected by kaggle/cache-448x6-s*/build.sh ----------
# IMG must be a multiple of 14: DINOv2 is patch-14, and 448 = 14 x 32.
import src.data.cache as _cache                              # noqa: E402
import src.data.slots as _slots                              # noqa: E402

_slots.IMG = 448
_slots.N_GROUP = 2                     # GROUP(3) * N_GROUP(2) = 6 slices
_cache.IMG = 448
# And the SCRIPT's own binding: kaggle_05_cache.py does `from src.data.slots
# import IMG` and then sizes the memmap with it, so rebinding only the two
# modules would allocate a 336 tensor and try to fill it with 448 frames.
IMG = 448
N_SLICE = _slots.GROUP * _slots.N_GROUP
_cache.N_SLICE = N_SLICE
assert _cache.IMG == _slots.IMG == 448, "IMG override failed"
assert _cache.N_SLICE == N_SLICE == 6, "N_SLICE override failed"
assert _slots.IMG % 14 == 0, "DINOv2 is patch-14"
print(f"[cfg] IMG={_slots.IMG} N_SLICE={N_SLICE}", flush=True)
"""
open(path, "w").write(src.replace(marker, override))
PY

grep -q 'os.environ.get("SHARD", "1")'  "$2/script.py" || { echo "SHARD patch missed" >&2; exit 1; }
grep -q 'os.environ.get("N_SHARD", "2")'  "$2/script.py" || { echo "N_SHARD patch missed - a half cache scores like a bad model" >&2; exit 1; }
grep -q '_slots.IMG = 448'                "$2/script.py" || { echo "IMG override missed - this would rebuild the 336 cache" >&2; exit 1; }
grep -q '_cache.IMG = 448'                "$2/script.py" || { echo "cache.IMG imported at module load; both names must be rebound" >&2; exit 1; }
grep -q '^IMG = 448'                      "$2/script.py" || { echo "the script sizes its memmap from its OWN IMG; rebind that too" >&2; exit 1; }
