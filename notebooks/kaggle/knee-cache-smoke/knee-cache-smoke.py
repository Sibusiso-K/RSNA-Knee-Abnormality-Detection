"""KAGGLE PREPROCESSING NOTEBOOK — decode once, train many times.

Settings: Accelerator **NONE (CPU)** | Internet OFF | ~12 h cap

This is the highest-leverage notebook in the project and it costs **zero GPU
quota**, which is the entire point.

The problem it solves
---------------------
`kaggle_02_train.py` calls `load_study()` inside `Dataset.__getitem__`, so every
DICOM is decoded again on every epoch. Measured: ~3,600 s/epoch, of which the
overwhelming majority is DICOM decoding, not gradient computation. An 8-epoch
run therefore performs ~1.3 M decodes on a GPU machine while the GPU idles.

Decoding is CPU work. Kaggle CPU notebooks do not draw on the 30 h/week GPU
quota. Doing it once here and publishing the result as a Dataset converts that
quota from ~1/8 useful to ~fully useful — roughly an 8x increase in effective
training throughput on exactly the same free-tier allowance.

What it writes
--------------
    cache_{shard}.npy   uint8 (n_study, N_SLOT, GROUP*N_GROUP, IMG, IMG)
    mask_{shard}.npy    uint8 (n_study, N_SLOT)   1 = slot present
    index_{shard}.csv   StudyInstanceUID, side, per-slot series UID, crop rate

At the defaults (6 slots x 3 slices x 336 px) that is ~2.0 MB/study, so the
full 4,407-study corpus is ~9 GB — inside Kaggle's 20 GB working limit in one
shard, and small enough to move to any other GPU provider.

Sharding exists because this project has already lost a ~7.5 h run to a timeout
with nothing saved. `--shard i/n` splits by study index and each shard is
independently publishable; partials are flushed every FLUSH_EVERY studies, so a
kill costs the tail rather than the run.
"""

import os
import sys
import time
import glob
import shutil

import numpy as np
import pandas as pd

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --- src bootstrap -------------------------------------------------------
# Kaggle flattens uploaded folders and "knee-src" is not a legal package name,
# so rebuild a real `src` package under /kaggle/working. Resolve every path by
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

from src.data.slots import (  # noqa: E402
    GROUP,
    IMG,
    N_GROUP,
    N_SLOT,
    SLOTS,
    SeriesInfo,
    band_indices,
    classify_weighting,
    normalise_intensity,
    normalise_laterality,
    physical_crop,
    pick_slot_series,
    side_from_geometry,
    study_side,
)

# -------------------------------------------------------------------------

SHARD = int(os.environ.get("SHARD", "0"))
N_SHARD = int(os.environ.get("N_SHARD", "1"))
SPLIT = os.environ.get("SPLIT", "train")          # 'train' or 'test'
LIMIT = int(os.environ.get("LIMIT", "300"))          # >0 = smoke test on N studies
FLUSH_EVERY = 200
N_SLICE = GROUP * N_GROUP
OUT = "/kaggle/working"

SERIES_DIR = f"{COMP}/{SPLIT}_series"
meta = pd.read_csv(f"{COMP}/{SPLIT}.csv")
series_meta = pd.read_csv(f"{COMP}/{SPLIT}_series.csv")
log(f"{SPLIT}: {len(meta)} studies, {len(series_meta)} series")

studies = meta["StudyInstanceUID"].tolist()
if LIMIT:
    studies = studies[:LIMIT]
    log(f"LIMIT set: smoke test on {len(studies)} studies")
studies = studies[SHARD::N_SHARD] if N_SHARD > 1 else studies
log(f"shard {SHARD}/{N_SHARD}: {len(studies)} studies")

PLANE_OF = dict(
    zip(series_meta["SeriesInstanceUID"], series_meta["Anatomical_Plane"])
)

# Tags parsed per header. Restricting the set is not a micro-optimisation here:
# this notebook opens ~800 k files and full parsing of each is the difference
# between fitting in the 12 h cap and not.
HDR_TAGS = [
    "SeriesInstanceUID", "SeriesDescription", "SequenceName", "ScanOptions",
    "ScanningSequence", "RepetitionTime", "EchoTime", "Laterality",
    "PixelSpacing", "Rows", "Columns", "ImagePositionPatient",
    "ImageOrientationPatient", "InstanceNumber",
]


def _num(ds, name):
    try:
        value = getattr(ds, name, None)
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def read_header(path):
    import pydicom

    try:
        return pydicom.dcmread(
            path, stop_before_pixels=True, specific_tags=HDR_TAGS
        )
    except Exception:
        return None


def slice_key(ds):
    """Projection of the slice origin onto the slice normal.

    The only reliable ordering key: filenames are SOP Instance UIDs, assigned to
    be unique rather than ordered, so sorting by name scrambles the stack.
    Falls back to InstanceNumber, then 0.0, so degraded metadata still yields
    *an* order rather than raising.
    """
    try:
        orientation = [float(v) for v in ds.ImageOrientationPatient]
        position = [float(v) for v in ds.ImagePositionPatient]
        normal = np.cross(np.array(orientation[:3]), np.array(orientation[3:6]))
        return float(np.dot(np.array(position), normal))
    except Exception:
        value = _num(ds, "InstanceNumber")
        return value if value is not None else 0.0


def probe_study(study_uid):
    """One header per series -> [SeriesInfo], plus the per-series file lists."""
    infos, files_of = [], {}
    study_dir = os.path.join(SERIES_DIR, study_uid)
    if not os.path.isdir(study_dir):
        return infos, files_of

    for series_uid in sorted(os.listdir(study_dir)):
        files = glob.glob(os.path.join(study_dir, series_uid, "*.dcm"))
        if not files:
            continue
        ds = read_header(files[0])
        if ds is None:
            continue

        plane = PLANE_OF.get(series_uid)
        if not isinstance(plane, str) or plane not in ("Sagittal", "Coronal", "Axial"):
            continue

        text = " ".join(
            str(getattr(ds, tag, "") or "")
            for tag in ("SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence")
        )
        fluid, fatsat = classify_weighting(
            text, _num(ds, "RepetitionTime"), _num(ds, "EchoTime")
        )

        pixel_mm = None
        try:
            pixel_mm = float(ds.PixelSpacing[0])
        except Exception:
            pixel_mm = None

        # Both routes are computed every time, even when the tag is present and
        # settles it. The extra cost is one dot product on a header already in
        # memory, and it buys the only check available on whether the geometry
        # rule is right: agreement with the tag on the half of the corpus that
        # carries one. Without it the mirror is unfalsifiable.
        laterality = str(getattr(ds, "Laterality", "") or "").strip().upper()
        tag_side = laterality if laterality in ("L", "R") else None
        geom_side = side_from_geometry(
            getattr(ds, "ImageOrientationPatient", None),
            getattr(ds, "ImagePositionPatient", None),
            getattr(ds, "Rows", 0),
            getattr(ds, "Columns", 0),
            pixel_mm,
        )

        infos.append(
            SeriesInfo(
                uid=series_uid, plane=plane, fluid=fluid, fatsat=fatsat,
                n_files=len(files), side=tag_side or geom_side,
                pixel_mm=pixel_mm, tag_side=tag_side, geom_side=geom_side,
            )
        )
        files_of[series_uid] = files
    return infos, files_of


def read_slot(files, plane, side):
    """A chosen series -> uint8 (N_SLICE, IMG, IMG), laterality-normalised.

    Headers of every file are read to establish the true geometric order, then
    only the N_SLICE frames actually wanted are decoded. Header parsing is
    cheap relative to pixel decoding, so this is far faster than decoding the
    whole series and discarding most of it.
    """
    import cv2

    keyed = []
    for path in files:
        ds = read_header(path)
        if ds is not None:
            keyed.append((slice_key(ds), path, ds))
    if not keyed:
        return None, False

    keyed.sort(key=lambda t: t[0])
    picks = band_indices(len(keyed), N_SLICE)

    frames, cropped_any = [], 0
    for i in picks:
        _key, path, ds = keyed[i]
        try:
            import pydicom

            full = pydicom.dcmread(path)
            pixels = full.pixel_array.astype(np.float32)
        except Exception:
            # A slice that will not decode is replaced by its nearest already-
            # decoded neighbour rather than by zeros: a black frame inside a
            # 3-channel group is a strong, entirely artificial edge.
            frames.append(frames[-1] if frames else np.zeros((IMG, IMG), np.float32))
            continue

        pixel_mm = None
        try:
            pixel_mm = float(ds.PixelSpacing[0])
        except Exception:
            pixel_mm = None

        pixels, applied = physical_crop(pixels, pixel_mm)
        cropped_any += int(applied)
        frames.append(cv2.resize(pixels, (IMG, IMG), interpolation=cv2.INTER_AREA))

    volume = np.stack(frames)
    volume = normalise_laterality(volume, plane, side)
    return normalise_intensity(volume), cropped_any == len(picks)


cache = np.zeros((len(studies), N_SLOT, N_SLICE, IMG, IMG), dtype=np.uint8)
mask = np.zeros((len(studies), N_SLOT), dtype=np.uint8)
rows = []

log(f"cache tensor: {cache.nbytes / 1024**3:.2f} GB "
    f"({len(studies)} x {N_SLOT} x {N_SLICE} x {IMG}^2)")
log(f"slots: {[s[0] for s in SLOTS]}")


def flush():
    np.save(f"{OUT}/cache_{SPLIT}_{SHARD}.npy", cache)
    np.save(f"{OUT}/mask_{SPLIT}_{SHARD}.npy", mask)
    pd.DataFrame(rows).to_csv(f"{OUT}/index_{SPLIT}_{SHARD}.csv", index=False)


sides = {"L": 0, "R": 0, None: 0}
crop_ok = crop_total = 0
lat_agree = lat_disagree = lat_tag_only = lat_geom_only = 0

for n, study_uid in enumerate(studies):
    try:
        infos, files_of = probe_study(study_uid)
        side = study_side(infos)
        sides[side if side in ("L", "R") else None] += 1

        # Audit the two routes against each other, per series.
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

        picked = pick_slot_series(infos)

        row = {"StudyInstanceUID": study_uid, "side": side or ""}
        for slot_i, info in enumerate(picked):
            name = SLOTS[slot_i][0]
            if info is None:
                row[name] = ""
                continue
            volume, fully_cropped = read_slot(
                files_of[info.uid], info.plane, side
            )
            if volume is None:
                row[name] = ""
                continue
            cache[n, slot_i] = volume
            mask[n, slot_i] = 1
            row[name] = info.uid
            crop_ok += int(fully_cropped)
            crop_total += 1
        rows.append(row)
    except Exception as exc:                      # noqa: BLE001
        log(f"  study {study_uid} failed: {type(exc).__name__}: {exc}")
        rows.append({"StudyInstanceUID": study_uid, "side": ""})

    if (n + 1) % 20 == 0:
        rate = (n + 1) / (time.time() - T0)
        eta = (len(studies) - n - 1) / max(rate, 1e-6) / 60.0
        log(f"  {n + 1}/{len(studies)}  {rate:.2f} study/s  ETA {eta:.0f} min  "
            f"slots filled {int(mask[: n + 1].sum())}")
    if (n + 1) % FLUSH_EVERY == 0:
        flush()
        log(f"  flushed at {n + 1}")

flush()

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
        "The mirror would be applied to the wrong studies, which is WORSE "
        "than not mirroring. Do not train on this cache.")
log(f"physical crop applied on {crop_ok}/{crop_total} slots "
    f"({100.0 * crop_ok / max(crop_total, 1):.0f}%)")
if crop_total and crop_ok / crop_total < 0.5:
    log("!! crop mostly did NOT fit — CROP_MM is too large for this corpus")
