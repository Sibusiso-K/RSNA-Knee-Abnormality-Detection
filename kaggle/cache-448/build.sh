#!/usr/bin/env bash
# The same cache at 448 px instead of 336. Free CPU, no accelerator quota.
#
# 130 mm / 448 px = 0.290 mm/px, against 0.387 at 336. A meniscal tear is
# 1-3 mm, so this widens the lesion from ~3-8 px to ~4-10 px. Resolution is the
# one input axis already shown to matter here: the original pipeline sat at
# 0.714 mm/px and the 160 mm crop experiment returned nothing because of it.
#
# 448 = 14 x 32, so a patch-14 ViT takes it unresampled. 14.83 GB, inside the
# 20 GB working limit, and the memmap rewrite means the build no longer pays
# ~197 GB of redundant writes to get there.
#
# THE OVERRIDE HAS TO SET TWO NAMES, NOT ONE.
# kaggle_05_cache.py does `from src.data.slots import IMG`, and src/data/cache.py
# does the same, so each holds its own COPY bound at import time. Setting only
# the module attribute on src.data.slots changes neither. Getting this wrong is
# not a silent-wrong-answer bug - the script would size the memmap at 336 while
# build_study returned 448 and die on the first assignment - but it would die
# hours into a build, so it is asserted at runtime below rather than hoped for.
set -euo pipefail
cp "$1/notebooks/kaggle_05_cache.py" "$2/script.py"

python - "$2/script.py" <<'PY'
import sys

path = sys.argv[1]
src = open(path).read()
marker = "from src.data.slots import IMG, N_SLOT, SLOTS        # noqa: E402"
assert marker in src, "import line moved; re-point the override"

override = marker + """

# --- 448 px override, injected by kaggle/cache-448/build.sh ---------------
# Both the module that USES IMG (src.data.cache) and this script's own copy
# have to be rebound; each took its value at import time.
import src.data.cache as _cache                              # noqa: E402

IMG = 448
_cache.IMG = IMG
assert _cache.IMG == IMG == 448, "IMG override failed"
assert IMG % 14 == 0, "DINOv2 is patch-14: the side must be a multiple of 14"
print(f"[cfg] IMG overridden to {IMG} ({130.0 / IMG:.4f} mm/px)", flush=True)
"""
open(path, "w").write(src.replace(marker, override))
PY

grep -q '_cache.IMG = IMG' "$2/script.py" || { echo "IMG override missed" >&2; exit 1; }
grep -q 'IMG = 448'        "$2/script.py" || { echo "IMG not set to 448" >&2; exit 1; }
grep -q 'open_memmap'      "$2/script.py" || { echo "must write the cache as a memmap; np.save per flush is ~400 GB of writes at this size" >&2; exit 1; }
