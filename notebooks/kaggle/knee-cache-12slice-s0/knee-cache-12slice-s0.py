"""KAGGLE PREPROCESSING NOTEBOOK — decode once, train many times.

Settings: Accelerator **NONE (CPU)** | Internet OFF | ~12 h cap

This is the highest-leverage notebook in the project and it costs **zero GPU
quota**, which is the entire point.

The problem it solves
---------------------
`kaggle_02_train.py` calls `load_study()` inside `Dataset.__getitem__`, so every
DICOM is decoded again on every epoch. Measured: ~3,600 s/epoch, of which the
overwhelming majority is decoding, not gradient computation. An 8-epoch run
therefore performs ~1.3 M decodes on a GPU machine while the GPU idles.

Decoding is CPU work, and Kaggle CPU notebooks do not draw on the 30 h/week GPU
quota. Doing it once here and publishing the result as a Dataset turns that
quota from ~1/8 useful into nearly all useful — roughly an 8x increase in
effective training throughput on exactly the same free-tier allowance. It also
makes training portable: the cache is ~9 GB, so it runs on Lightning or a
rented box, while the 570 GB of DICOMs never leave Kaggle.

What it writes
--------------
    cache_{split}_{shard}.npy   uint8 (n_study, N_SLOT, GROUP*N_GROUP, IMG, IMG)
    mask_{split}_{shard}.npy    uint8 (n_study, N_SLOT)   1 = slot present
    index_{split}_{shard}.csv   uid, side, scanner fingerprint, per-slot series

At 6 slots x 3 slices x 336 px that is ~2.03 MB/study, so 4,407 studies is
~8.95 GB — inside the 20 GB working limit in one shard, measured rather than
estimated (the 40-study smoke wrote 0.08 GB).

The decoding itself lives in `src/data/cache.py` because the submission
notebook must build the test cache through **exactly** the same code. A fork
between the two would apply weights to pixels they were never trained on, and
nothing in the pipeline would complain.
"""

import os
import sys
import time
import shutil

import numpy as np
import pandas as pd

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --- src bootstrap -------------------------------------------------------
# Kaggle flattens uploaded folders and "knee-src" is not a legal package name,
# so rebuild a real `src` package under /kaggle/working. Resolve paths by
# CONTENT, never by hardcoded slug: this account gets a nested mount
# (/kaggle/input/competitions/...) while public snippets assume a flat one.
PKG = "/kaggle/working/pkg"
INPUT = "/kaggle/input"


def find_dir(marker, max_depth=5):
    if not os.path.isdir(INPUT):
        return None
    stack = [(INPUT, 0)]
    while stack:
        directory, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            entries = os.listdir(directory)
        except OSError:
            continue
        if marker in entries:
            return directory
        for entry in entries:
            path = os.path.join(directory, entry)
            if os.path.isdir(path):
                stack.append((path, depth + 1))
    return None


_src = find_dir("labels.py")
COMP = find_dir("train_series.csv")
log(f"src : {_src}")
log(f"comp: {COMP}")
if _src is None or COMP is None:
    raise SystemExit("Attach knee-src and the competition data.")

if not os.path.exists(PKG + "/src"):
    os.makedirs(PKG, exist_ok=True)
    shutil.copytree(_src, PKG + "/src")
sys.path.insert(0, PKG)

from src.data.cache import N_SLICE, build_study      # noqa: E402
from src.data.slots import IMG, N_SLOT, SLOTS        # noqa: E402

# --- 12-slices-per-slot override ----------------------------------------
import src.data.cache as _cache                              # noqa: E402
import src.data.slots as _slots                              # noqa: E402

_slots.N_GROUP = 4                     # GROUP(3) x N_GROUP(4) = 12 slices
N_SLICE = _slots.GROUP * _slots.N_GROUP
_cache.N_SLICE = N_SLICE
assert _cache.N_SLICE == N_SLICE == 12, "N_SLICE override failed"
print(f"[cfg] N_SLICE overridden to {N_SLICE}", flush=True)


# -------------------------------------------------------------------------

SHARD = int(os.environ.get("SHARD", "0"))
N_SHARD = int(os.environ.get("N_SHARD", "2"))
SPLIT = os.environ.get("SPLIT", "train")
LIMIT = int(os.environ.get("LIMIT", "0"))          # >0 = smoke test on N studies
FLUSH_EVERY = 200
OUT = "/kaggle/working"

SERIES_DIR = f"{COMP}/{SPLIT}_series"
meta = pd.read_csv(f"{COMP}/{SPLIT}.csv")
series_meta = pd.read_csv(f"{COMP}/{SPLIT}_series.csv")
log(f"{SPLIT}: {len(meta)} studies, {len(series_meta)} series")

studies = meta["StudyInstanceUID"].tolist()
if LIMIT:
    studies = studies[:LIMIT]
    log(f"LIMIT set: smoke test on {len(studies)} studies")
if N_SHARD > 1:
    studies = studies[SHARD::N_SHARD]
log(f"shard {SHARD}/{N_SHARD}: {len(studies)} studies")

PLANE_OF = dict(zip(series_meta["SeriesInstanceUID"],
                    series_meta["Anatomical_Plane"]))

# The cache is written straight to disk as a memmap rather than built in RAM
# and saved periodically.
#
# The old flush() called np.save on the WHOLE array, so a checkpoint every 200
# studies rewrote every byte written so far: 22 flushes x 8.96 GB is ~197 GB of
# writes to produce a 8.96 GB file. At six slices per slot that becomes ~394 GB
# and stops being viable at all, which is what blocks the obvious next
# experiment. A memmap makes each study's write land once, and it also means
# the build no longer needs the cache to fit in RAM.
CACHE_PATH = f"{OUT}/cache_{SPLIT}_{SHARD}.npy"
cache = np.lib.format.open_memmap(
    CACHE_PATH, mode="w+", dtype=np.uint8,
    shape=(len(studies), N_SLOT, N_SLICE, IMG, IMG),
)
mask = np.zeros((len(studies), N_SLOT), dtype=np.uint8)
rows = []

log(f"cache tensor: {cache.nbytes / 1024**3:.2f} GB "
    f"({len(studies)} x {N_SLOT} x {N_SLICE} x {IMG}^2) -> {CACHE_PATH}")
log(f"slots: {[s[0] for s in SLOTS]}")


def flush():
    """Only the small companions need writing; the pixels are already on disk."""
    cache.flush()
    np.save(f"{OUT}/mask_{SPLIT}_{SHARD}.npy", mask)
    pd.DataFrame(rows).to_csv(f"{OUT}/index_{SPLIT}_{SHARD}.csv", index=False)


sides = {"L": 0, "R": 0, None: 0}
crop_ok = crop_total = 0
lat_agree = lat_disagree = lat_tag_only = lat_geom_only = 0

for n, study_uid in enumerate(studies):
    try:
        volumes, study_mask, row, infos, (ok, total) = build_study(
            SERIES_DIR, study_uid, PLANE_OF
        )
        cache[n] = volumes
        mask[n] = study_mask
        rows.append(row)
        crop_ok += ok
        crop_total += total
        sides[row["side"] if row["side"] in ("L", "R") else None] += 1

        for info in infos:
            if info.tag_side and info.geom_side:
                if info.tag_side == info.geom_side:
                    lat_agree += 1
                else:
                    lat_disagree += 1
            elif info.tag_side:
                lat_tag_only += 1
            elif info.geom_side:
                lat_geom_only += 1
    except Exception as exc:                      # noqa: BLE001
        log(f"  study {study_uid} failed: {type(exc).__name__}: {exc}")
        rows.append({"StudyInstanceUID": study_uid, "side": "", "fingerprint": ""})

    if (n + 1) % 20 == 0:
        rate = (n + 1) / (time.time() - T0)
        eta = (len(studies) - n - 1) / max(rate, 1e-6) / 60.0
        both = lat_agree + lat_disagree
        # The three health numbers are printed CONTINUOUSLY, not just at the
        # end. A full build is ~2.3 h; discovering at the end that laterality
        # never resolved would cost the whole run. These are visible by study
        # 20 and stable by ~200, so a broken build can be killed in minutes.
        log(f"  {n + 1}/{len(studies)}  {rate:.2f} study/s  ETA {eta:.0f} min  "
            f"| fill {mask[: n + 1].mean():.2f} "
            f"| crop {100.0 * crop_ok / max(crop_total, 1):.0f}% "
            f"| side unknown {sides[None]} "
            f"| lat agree {100.0 * lat_agree / max(both, 1):.0f}% of {both}")
    if (n + 1) % FLUSH_EVERY == 0:
        flush()
        log(f"  flushed at {n + 1}")

flush()

# Remove the bootstrapped src copy from the OUTPUT before finishing.
#
# /kaggle/working is this kernel's output, so pkg/src ships with the cache.
# Any later kernel mounting this via kernel_sources can then resolve
# find_dir("labels.py") to THIS frozen copy instead of the live knee-src
# dataset - which is exactly what happened: two runs died on
# "SlotNet.__init__() got an unexpected keyword argument 'head'" while the
# published src demonstrably had it. Consumers now skip "pkg" too, but not
# shipping it is the fix that does not rely on every consumer remembering.
shutil.rmtree(PKG, ignore_errors=True)
log(f"removed {PKG} from the output so it cannot shadow knee-src later")

log("=== done ===")
log(f"studies: {len(studies)}")
log(f"slot fill rate: {mask.mean():.3f} overall")
for slot_i, spec in enumerate(SLOTS):
    log(f"  {spec[0]:16s} {mask[:, slot_i].mean():.3f}")
log(f"laterality: L={sides['L']} R={sides['R']} unknown={sides[None]}")

both = lat_agree + lat_disagree
log(f"laterality audit (per series): tag+geom both present {both}, "
    f"agree {lat_agree} ({100.0 * lat_agree / max(both, 1):.1f}%), "
    f"disagree {lat_disagree} | tag only {lat_tag_only} | geom only {lat_geom_only}")
if both and lat_agree / both < 0.95:
    log("!! geometry disagrees with the Laterality tag too often to trust. "
        "The mirror would be applied to the WRONG studies, which is worse than "
        "not mirroring at all. Do not train on this cache.")

log(f"physical crop applied on {crop_ok}/{crop_total} slots "
    f"({100.0 * crop_ok / max(crop_total, 1):.0f}%)")
if crop_total and crop_ok / crop_total < 0.5:
    log("!! crop mostly did NOT fit — CROP_MM is too large for this corpus, so "
        "one output pixel is not a constant number of millimetres.")
