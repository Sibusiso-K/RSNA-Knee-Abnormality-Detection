"""KAGGLE FAT-SUPPRESSION ROUTING SCAN — CPU, free, ~11 min.

Settings: Accelerator **NONE (CPU)** | Internet OFF

Sizes one decision: **should slot routing take fat-suppression from the CSV
instead of from `_FATSAT_RX`?**

Established by `knee-t1-scan` and the local follow-up:

- The column named `Fluid_Sensitive` actually encodes **fat suppression**. It
  agrees with the header-derived fat-sat value on **97.31%** of series, versus
  80.19% if read as fluid-sensitivity. (`Fluid_Sensitive` and `Fat_Suppression`
  are identical in all 24,371 rows, so they carry one bit; this is which bit.)
- The 592 series where the CSV says fat-suppressed and the regex does not are
  **regex misses**, from three causes: the underscore bug (`\\bfs\\b`,
  `\\bstir\\b`, `\\btirm\\b` cannot match between underscores), missing
  vocabulary (`we` = water excitation, Philips `SMART FAT`), and series whose
  only fat-sat evidence is `ScanningSequence == IR`.

592 series is **18x** the T1 underscore bug, and unlike that one it lands on the
axis separating `SAG_FLUID_FS` from `SAG_FLUID_NOFS`.

**But 592 series is an upper bound on studies affected, not a count**, and that
is exactly the mistake the T1 chase nearly made: 10,361 structural series
turned out to move only 23 studies. `pick_slot_series` resolves ties by
`n_files`, so a series can change its fat-sat verdict and still lose its slot to
the same competitor it lost to before, changing nothing.

So this scan keeps `n_files` (which `knee-t1-scan` did not) and reports the only
numbers that justify a cache rebuild:

1. how many studies change which SERIES occupies a slot (different pixels), and
2. how many change whether a slot is filled at all (different mask).

Both matter. (1) is the larger effect and the easier one to overlook, because
the presence mask looks identical while the model is fed a different acquisition.
"""

import os
import sys
import glob
import shutil
import time
from collections import Counter

import pandas as pd

START = time.time()


def log(msg):
    print(f"[{time.time() - START:6.1f}s] {msg}", flush=True)


PKG = "/kaggle/working/pkg"
INPUT = "/kaggle/input"


def find_dir(marker, max_depth=6):
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
if _src is None or COMP is None:
    raise SystemExit(f"src={_src} comp={COMP} — attach knee-src and the competition")
if not os.path.exists(PKG + "/src"):
    os.makedirs(PKG, exist_ok=True)
    shutil.copytree(_src, PKG + "/src")
sys.path.insert(0, PKG)

from src.data.cache import read_header, _num                      # noqa: E402
from src.data.slots import (                                      # noqa: E402
    SLOTS, SeriesInfo, classify_weighting, pick_slot_series,
)

SPLIT = "train"
SERIES_DIR = f"{COMP}/{SPLIT}_series"
series_meta = pd.read_csv(f"{COMP}/{SPLIT}_series.csv")
plane_of = dict(zip(series_meta.SeriesInstanceUID, series_meta.Anatomical_Plane))

# Either column works - they are identical in all 24,371 rows - but read the one
# whose NAME matches what it turned out to contain, so this line does not look
# like a bug to the next reader.
fatsat_csv_of = dict(zip(series_meta.SeriesInstanceUID,
                         series_meta.Fat_Suppression))
studies = sorted(series_meta.StudyInstanceUID.unique())
log(f"{len(series_meta):,} series across {len(studies):,} studies")

NAMES = [s[0] for s in SLOTS]
FLUID_SLOTS = [i for i, s in enumerate(SLOTS) if s[3] is not None]
log(f"slots affected by the fat-sat bit: {[NAMES[i] for i in FLUID_SLOTS]}")

rows, study_rows, examples = [], [], []

for n_done, study_uid in enumerate(studies):
    study_dir = os.path.join(SERIES_DIR, study_uid)
    if not os.path.isdir(study_dir):
        continue
    infos_now, infos_fix = [], []

    for series_uid in sorted(os.listdir(study_dir)):
        files = glob.glob(os.path.join(study_dir, series_uid, "*.dcm"))
        if not files:
            continue
        ds = read_header(files[0])
        if ds is None:
            continue
        plane = plane_of.get(series_uid)
        if not isinstance(plane, str) or plane not in ("Sagittal", "Coronal", "Axial"):
            continue

        text = " ".join(
            str(getattr(ds, tag, "") or "")
            for tag in ("SeriesDescription", "SequenceName", "ScanOptions",
                        "ScanningSequence")
        ).strip()
        fluid, fatsat_hdr = classify_weighting(
            text, _num(ds, "RepetitionTime"), _num(ds, "EchoTime")
        )
        flag = fatsat_csv_of.get(series_uid)
        # A series missing from the CSV keeps the header verdict rather than
        # being forced to a default - an absent flag is not evidence of absence.
        fatsat_fix = bool(flag == 1) if pd.notna(flag) else fatsat_hdr

        if fatsat_hdr != fatsat_fix and len(examples) < 20:
            examples.append((plane, fatsat_hdr, fatsat_fix, text[:70]))
        rows.append({"plane": plane, "fluid": fluid, "fatsat_hdr": fatsat_hdr,
                     "fatsat_fix": fatsat_fix, "changed": fatsat_hdr != fatsat_fix,
                     "n_files": len(files), "text": text})

        common = dict(uid=series_uid, plane=plane, fluid=fluid,
                      n_files=len(files), side=None, pixel_mm=None)
        infos_now.append(SeriesInfo(fatsat=fatsat_hdr, **common))
        infos_fix.append(SeriesInfo(fatsat=fatsat_fix, **common))

    now = pick_slot_series(infos_now)
    fix = pick_slot_series(infos_fix)
    rec = {"study": study_uid}
    for i, name in enumerate(NAMES):
        rec[f"{name}_now"] = now[i].uid if now[i] else None
        rec[f"{name}_fix"] = fix[i].uid if fix[i] else None
    study_rows.append(rec)

    if (n_done + 1) % 1000 == 0:
        log(f"{n_done + 1}/{len(studies)} studies")

df = pd.DataFrame(rows)
sdf = pd.DataFrame(study_rows)
log(f"scanned {len(df):,} series in {len(sdf):,} studies")

print("\n" + "=" * 70)
print("1. SERIES WHOSE FAT-SAT VERDICT CHANGES")
print("=" * 70)
ch = df[df.changed]
print(f"{len(ch):,} of {len(df):,} series ({len(ch)/max(1,len(df)):.2%})")
if len(ch):
    print("\ndirection:")
    print(f"  regex said NOT fat-sat, CSV says fat-sat: {int((~ch.fatsat_hdr & ch.fatsat_fix).sum()):,}")
    print(f"  regex said fat-sat, CSV says NOT        : {int((ch.fatsat_hdr & ~ch.fatsat_fix).sum()):,}")
    print("\nby plane:")
    print(ch.plane.value_counts().to_string())

print("\n" + "=" * 70)
print("2. SLOT FILL, PER SLOT")
print("=" * 70)
for i, name in enumerate(NAMES):
    now_f = sdf[f"{name}_now"].notna()
    fix_f = sdf[f"{name}_fix"].notna()
    gained = int((~now_f & fix_f).sum())
    lost = int((now_f & ~fix_f).sum())
    swapped = int((now_f & fix_f &
                   (sdf[f"{name}_now"] != sdf[f"{name}_fix"])).sum())
    mark = "  <-- fat-sat axis" if i in FLUID_SLOTS else ""
    print(f"{name:16s} filled {now_f.mean():6.2%} -> {fix_f.mean():6.2%} | "
          f"gained {gained:4d} lost {lost:4d} SWAPPED {swapped:4d}{mark}")

print("\n" + "=" * 70)
print("3. STUDY-LEVEL IMPACT  <-- the deliverable")
print("=" * 70)
mask_ch = pd.Series(False, index=sdf.index)
ident_ch = pd.Series(False, index=sdf.index)
for name in NAMES:
    now_c, fix_c = sdf[f"{name}_now"], sdf[f"{name}_fix"]
    mask_ch |= now_c.notna() != fix_c.notna()
    ident_ch |= now_c.notna() & fix_c.notna() & (now_c != fix_c)
any_ch = mask_ch | ident_ch
print(f"studies where a slot's PRESENCE changes (mask differs) : "
      f"{int(mask_ch.sum()):,} ({mask_ch.mean():.2%})")
print(f"studies where a slot's SERIES changes (same mask, new  ")
print(f"  pixels - the easy one to miss)                       : "
      f"{int(ident_ch.sum()):,} ({ident_ch.mean():.2%})")
print(f"studies affected AT ALL                                : "
      f"{int(any_ch.sum()):,} ({any_ch.mean():.2%})")
print()
print("Compare: the T1 underscore bug moved 23 studies (0.52%) and was closed")
print("as immaterial. Rebuilding every cache and retraining 5 folds needs a")
print("number that clearly beats that.")

print("\n" + "=" * 70)
print("4. ACTUAL SERIES TEXT THAT CHANGES VERDICT")
print("=" * 70)
for plane, hdr, fix, text in examples:
    print(f"  {plane:9s} regex={str(hdr):5s} -> csv={str(fix):5s}  {text}")
print("\nmost common changed descriptions:")
for text, count in Counter(ch.text).most_common(15):
    print(f"  {count:5d}  {text[:78]}")

df.to_csv("/kaggle/working/fatsat_scan_series.csv", index=False)
sdf.to_csv("/kaggle/working/fatsat_scan_studies.csv", index=False)
log("wrote fatsat_scan_series.csv + fatsat_scan_studies.csv")
