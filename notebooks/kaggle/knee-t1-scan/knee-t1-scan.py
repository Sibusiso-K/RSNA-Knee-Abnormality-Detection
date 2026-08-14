"""KAGGLE T1 ROUTING AUDIT — CPU, free, no GPU/TPU quota.

Settings: Accelerator **NONE (CPU)** | Internet OFF

Measures one thing: **how often does `classify_weighting` misroute a structural
(T1) series into a fluid slot, and how much of that is the underscore bug?**

The bug, established locally by construction rather than guessed:
`_T1_RX` matches with `\\b`, and underscore is a word character in Python regex
(`\\w == [A-Za-z0-9_]`). So `\\bt1w\\b` cannot match `t1w_sag` — there is no
boundary between `w` and `_`. Underscore-separated descriptions are the Siemens
house style (`t1_tse_sag`), so this is not a hypothetical spelling.

Why it would matter: a misclassified T1 series is routed into a *fluid* slot,
leaving `COR_T1`/`SAG_T1` empty and masked out — and those two slots exist
because cartilage thinning and marrow signal, i.e. the three OA labels, read on
the structural sequences.

**What is NOT yet known, and is the entire point of this notebook: prevalence.**
`train_series.csv` sizes the population at risk (10,361 of 24,371 series are
structural, across all 4,407 studies) but says nothing about how they are named
— series text lives in the DICOM headers. A bug that is real by construction and
hits 3 series is not worth a cache rebuild; one that hits 3,000 is.

So the deliverable is the last table: **how many studies currently have an empty
T1 slot that the fix would fill.** That is the number the rebuild decision turns
on, not the raw regex miss count.

Cost: one header per series, 24,371 headers — not the 819,640-file full read.
"""

import os
import re
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

log(f"src : {_src}")
log(f"comp: {COMP}")

SPLIT = "train"
SERIES_DIR = f"{COMP}/{SPLIT}_series"
series_meta = pd.read_csv(f"{COMP}/{SPLIT}_series.csv")
plane_of = dict(zip(series_meta.SeriesInstanceUID, series_meta.Anatomical_Plane))
fluid_csv_of = dict(zip(series_meta.SeriesInstanceUID, series_meta.Fluid_Sensitive))
studies = sorted(series_meta.StudyInstanceUID.unique())
log(f"{len(series_meta):,} series across {len(studies):,} studies")

# --- the candidate fix ---------------------------------------------------
# Custom boundaries that treat underscore as a SEPARATOR rather than as part of
# the word. `(?<![a-z0-9])t1w?(?![a-z0-9])` matches t1 / t1w when flanked by
# anything that is not alphanumeric — so `_`, `-`, `/`, space and string edges
# all count, while `at1`, `t10` and `t1rho` correctly do not match.
T1_FIXED = re.compile(r"(?<![a-z0-9])t1w?(?![a-z0-9])", re.I)
T1_LOOSE = re.compile(r"t1", re.I)          # contains the characters at all
NAMES = [s[0] for s in SLOTS]
I_COR_T1, I_SAG_T1 = NAMES.index("COR_T1"), NAMES.index("SAG_T1")


def classify_fixed(text, tr, te):
    """`classify_weighting` with only the T1 detection swapped."""
    if T1_FIXED.search(text):
        fluid = False
        if re.search(r"\bstir\b|\btirm\b|\bspair\b", text, re.I):
            fluid = True
        return fluid, classify_weighting(text, tr, te)[1]
    return classify_weighting(text, tr, te)


# --- scan ----------------------------------------------------------------
rows = []
study_rows = []
examples = []
n_series = 0

for n_done, study_uid in enumerate(studies):
    study_dir = os.path.join(SERIES_DIR, study_uid)
    if not os.path.isdir(study_dir):
        continue
    infos_now, infos_fixed = [], []

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
        tr, te = _num(ds, "RepetitionTime"), _num(ds, "EchoTime")

        fluid_now, fatsat = classify_weighting(text, tr, te)
        fluid_fix, _ = classify_fixed(text, tr, te)
        n_series += 1

        flipped = fluid_now != fluid_fix
        rows.append({
            "plane": plane,
            "csv_fluid": fluid_csv_of.get(series_uid),
            "fluid_now": fluid_now,
            "fluid_fixed": fluid_fix,
            "flipped": flipped,
            "has_t1_chars": bool(T1_LOOSE.search(text)),
            "text": text,
        })
        if flipped and len(examples) < 25:
            examples.append((plane, text[:90]))

        common = dict(uid=series_uid, plane=plane, fatsat=fatsat,
                      n_files=len(files), side=None, pixel_mm=None)
        infos_now.append(SeriesInfo(fluid=fluid_now, **common))
        infos_fixed.append(SeriesInfo(fluid=fluid_fix, **common))

    slots_now = pick_slot_series(infos_now)
    slots_fix = pick_slot_series(infos_fixed)
    study_rows.append({
        "study": study_uid,
        "cor_t1_now": slots_now[I_COR_T1] is not None,
        "cor_t1_fix": slots_fix[I_COR_T1] is not None,
        "sag_t1_now": slots_now[I_SAG_T1] is not None,
        "sag_t1_fix": slots_fix[I_SAG_T1] is not None,
        "n_slots_now": sum(s is not None for s in slots_now),
        "n_slots_fix": sum(s is not None for s in slots_fix),
    })

    if (n_done + 1) % 500 == 0:
        log(f"{n_done + 1}/{len(studies)} studies, {n_series:,} series")

df = pd.DataFrame(rows)
sdf = pd.DataFrame(study_rows)
log(f"scanned {len(df):,} series in {len(sdf):,} studies")

# --- 1. the regex miss ---------------------------------------------------
print("\n" + "=" * 68)
print("1. HOW OFTEN DOES THE UNDERSCORE BUG FIRE?")
print("=" * 68)
flipped = df[df.flipped]
print(f"series whose weighting flips under the fix : {len(flipped):,} "
      f"({len(flipped) / max(1, len(df)):.2%} of {len(df):,})")
print("all of these are structural series currently read as fluid-sensitive.")
print("\nby plane:")
print(flipped.plane.value_counts().to_string() if len(flipped) else "  (none)")

# --- 2. agreement with the host's own flag -------------------------------
print("\n" + "=" * 68)
print("2. AGREEMENT WITH THE CSV Fluid_Sensitive FLAG")
print("=" * 68)
sub = df[df.csv_fluid.notna()].copy()
sub["csv_struct"] = sub.csv_fluid == 0
for label, col in (("current", "fluid_now"), ("fixed  ", "fluid_fixed")):
    agree = (sub["csv_struct"] != sub[col]).mean()
    missed = int(((sub["csv_struct"]) & (sub[col])).sum())
    print(f"{label}: agrees with CSV on {agree:.2%} of series | "
          f"structural series read as fluid: {missed:,}")

# --- 3. the number the rebuild decision turns on -------------------------
print("\n" + "=" * 68)
print("3. IMPACT AT THE SLOT LEVEL  <-- the deliverable")
print("=" * 68)
for name, now, fix in (("COR_T1", "cor_t1_now", "cor_t1_fix"),
                       ("SAG_T1", "sag_t1_now", "sag_t1_fix")):
    gained = int((~sdf[now] & sdf[fix]).sum())
    lost = int((sdf[now] & ~sdf[fix]).sum())
    print(f"{name}: filled now {sdf[now].mean():6.2%} -> fixed {sdf[fix].mean():6.2%} "
          f"| studies gaining the slot {gained:,} | losing it {lost:,}")
print(f"\nmean slots per study: now {sdf.n_slots_now.mean():.3f} "
      f"-> fixed {sdf.n_slots_fix.mean():.3f}")
changed = int((sdf.n_slots_now != sdf.n_slots_fix).sum())
print(f"studies whose slot COUNT changes at all: {changed:,} "
      f"({changed / max(1, len(sdf)):.2%})")

# --- 4. look at the actual strings ---------------------------------------
print("\n" + "=" * 68)
print("4. ACTUAL SERIES TEXT THAT FLIPS (read these, do not trust the counts alone)")
print("=" * 68)
for plane, text in examples:
    print(f"  {plane:9s} {text}")
if not examples:
    print("  (none — the bug does not fire on this corpus)")

print("\nmost common flipped descriptions:")
for text, count in Counter(flipped.text).most_common(15):
    print(f"  {count:5d}  {text[:80]}")

df.to_csv("/kaggle/working/t1_scan_series.csv", index=False)
sdf.to_csv("/kaggle/working/t1_scan_studies.csv", index=False)
log("wrote t1_scan_series.csv + t1_scan_studies.csv")
