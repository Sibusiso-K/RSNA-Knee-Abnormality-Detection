#!/usr/bin/env bash
# Six slices per slot instead of three, at 336 px. Free CPU, no quota.
#
# This tests COVERAGE, which is a different axis from the 448 resolution run
# and the one the per-label pattern points at. Sorted by fold-0 AUC, our
# findings run almost exactly diffuse -> focal:
#
#   Baker's .845  Medial OA .832  Effusion .831  Synovitis .825   (diffuse)
#   ...
#   Lateral Meniscus .754  MCL .747  PF OA .711                   (focal)
#
# Three samples across the 20-80% band sit 6-14 slices apart depending on
# series length, while a meniscal tear is visible on 2-3 CONSECUTIVE slices.
# So for focal findings the evidence can fall entirely between our samples -
# which would explain why the model reaches ~0.82 against the 58 gold studies
# while its own labels reach 0.8930. No amount of capacity recovers pixels
# that were never shown, and capacity is exactly what three failed experiments
# have already ruled out.
#
# Six slices halves the spacing to 2-5 slices. 16.68 GB, inside the 20 GB
# limit, and only affordable at all because flush() no longer rewrites the
# whole array.
#
# LIKE THE 448 BUILD, TWO NAMES MUST BE REBOUND. src/data/cache.py computes
# N_SLICE = GROUP * N_GROUP at import time and the script imports N_SLICE from
# it, so setting slots.N_GROUP alone changes neither.
set -euo pipefail
cp "$1/notebooks/kaggle_05_cache.py" "$2/script.py"

python - "$2/script.py" <<'PY'
import sys

path = sys.argv[1]
src = open(path).read()
marker = "from src.data.slots import IMG, N_SLOT, SLOTS        # noqa: E402"
assert marker in src, "import line moved; re-point the override"

override = marker + """

# --- 6-slices-per-slot override, injected by kaggle/cache-6slice/build.sh ---
import src.data.cache as _cache                              # noqa: E402
import src.data.slots as _slots                              # noqa: E402

_slots.N_GROUP = 2                     # GROUP(3) * N_GROUP(2) = 6 slices
N_SLICE = _slots.GROUP * _slots.N_GROUP
_cache.N_SLICE = N_SLICE
assert _cache.N_SLICE == N_SLICE == 6, "N_SLICE override failed"
print(f"[cfg] N_SLICE overridden to {N_SLICE} ({_slots.GROUP} x {_slots.N_GROUP})",
      flush=True)
"""
open(path, "w").write(src.replace(marker, override))
PY

grep -q '_cache.N_SLICE = N_SLICE' "$2/script.py" || { echo "N_SLICE override missed" >&2; exit 1; }
grep -q 'open_memmap'             "$2/script.py" || { echo "memmap required at this size" >&2; exit 1; }
grep -q '^IMG = 336'              "$1/src/data/slots.py" || { echo "this build holds IMG at 336; only slices change" >&2; exit 1; }
