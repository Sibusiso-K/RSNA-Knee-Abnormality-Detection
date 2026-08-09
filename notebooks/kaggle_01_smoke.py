"""KAGGLE SMOKE TEST — run this FIRST, before training anything.

Settings: Accelerator GPU T4 x2 | Internet ON | ~15 minutes

Answers the four questions that decide whether the training notebook is worth
starting, and that cannot be answered from a laptop with no DICOMs:

  1. Do all four transfer syntaxes actually decode here? (JPEG2000 and JPEG
     Lossless need codec libs; docs/03-data-guide.md flags this as the classic
     week-8 disaster that should be a week-4 fix.)
  2. How long does one study take to load? -> extrapolates to both the training
     epoch time and the 9-hour inference cap on ~1,300 test studies.
  3. Does a real volume come out sane — right shape, non-trivial intensity, and
     is the plane routing finding all three planes?
  4. Does the model do a forward+backward pass on a real batch on a T4?

If any of these fail, fix it here for pennies rather than 40 minutes into a
training run.
"""

import sys
import time

import numpy as np
import pandas as pd
import torch

# --- src bootstrap (identical in all three notebooks) --------------------
# Kaggle flattens the top-level folder of an uploaded dataset, so
# /kaggle/input/knee-src IS the contents of src/, not a folder containing it.
# Dataset dirs also can't be imported directly (the "-" in knee-src is not a
# legal package name). Rebuild a real `src` package under /kaggle/working so
# `from src.x import y` resolves identically here and locally — matching import
# paths is what stops train/inference code from quietly diverging.
import os
import shutil

PKG = "/kaggle/working/pkg"
if not os.path.exists(f"{PKG}/src"):
    os.makedirs(PKG, exist_ok=True)
    shutil.copytree("/kaggle/input/knee-src", f"{PKG}/src")
sys.path.insert(0, PKG)
# -------------------------------------------------------------------------

from src.data.dicom import PLANES, load_study, pick_series  # noqa: E402

COMP = "/kaggle/input/rsna-knee-abnormality-detection"
ID = "StudyInstanceUID"
N_SLICES, SIZE = 16, 224

train = pd.read_csv(f"{COMP}/train.csv")
series = pd.read_csv(f"{COMP}/train_series.csv")
print(f"studies {len(train)}  series {len(series)}")

# --- 1. transfer syntaxes -------------------------------------------------
import pydicom  # noqa: E402
from pathlib import Path  # noqa: E402

print("\n[1] transfer syntax decode check")
syntaxes: dict[str, list[bool]] = {}
for study_uid in train[ID].head(40):
    for dcm in Path(f"{COMP}/train_series/{study_uid}").rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
            name = str(ds.file_meta.TransferSyntaxUID.name)
        except Exception:
            continue
        if len(syntaxes.get(name, [])) >= 3:
            break
        try:
            pydicom.dcmread(str(dcm)).pixel_array
            ok = True
        except Exception as exc:
            ok = False
            print(f"    DECODE FAIL {name}: {exc}")
        syntaxes.setdefault(name, []).append(ok)
        break
for name, results in syntaxes.items():
    print(f"    {'OK  ' if all(results) else 'FAIL'} {name}  (n={len(results)})")

# --- 2 & 3. load timing and volume sanity ---------------------------------
print("\n[2/3] study load timing + volume sanity")
times, plane_hits = [], {p: 0 for p in PLANES}
sample = train[ID].head(8).tolist()
for study_uid in sample:
    picks = pick_series(series, study_uid)
    for plane, uid in picks.items():
        if uid is not None:
            plane_hits[plane] += 1
    t0 = time.time()
    volume = load_study(f"{COMP}/train_series", study_uid, series, N_SLICES, SIZE)
    times.append(time.time() - t0)
    nonzero = [float(volume[i].std()) for i in range(volume.shape[0])]
    print(
        f"    {study_uid[-8:]} {volume.shape} {times[-1]:5.1f}s  "
        f"per-plane std {[round(s, 3) for s in nonzero]}"
    )

mean_load = float(np.mean(times))
print(f"\n    mean {mean_load:.1f}s/study")
print(f"    plane coverage over {len(sample)} studies: {plane_hits}")
print(f"    -> one 4,400-study epoch, 2 workers: ~{mean_load * 4400 / 2 / 3600:.1f} h")
print(f"    -> 1,300 test studies, 2 workers:    ~{mean_load * 1300 / 2 / 3600:.1f} h "
      f"(inference cap is 9 h)")
if mean_load * 4400 / 2 / 3600 > 2:
    print("    WARNING: too slow to re-decode every epoch — cache to .npy first "
          "and publish as a Kaggle Dataset (docs/03-data-guide.md step 6).")

# --- 4. model step on real data -------------------------------------------
print("\n[4] model forward/backward on a real batch")
from src.model.net import KneeNet  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"    device: {device}  ({torch.cuda.get_device_name(0) if device == 'cuda' else 'no GPU'})")
model = KneeNet(pretrained=True).to(device)
batch = torch.from_numpy(
    np.stack([load_study(f"{COMP}/train_series", u, series, N_SLICES, SIZE) for u in sample[:2]])
).float().to(device)

t0 = time.time()
with torch.cuda.amp.autocast(enabled=device == "cuda"):
    out = model(batch)
    loss = out.mean()
loss.backward()
print(f"    logits {tuple(out.shape)} (expect (2, 12))  step {time.time() - t0:.1f}s")
if device == "cuda":
    print(f"    peak GPU mem: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB of ~15 GB")
print("\nsmoke test complete — read the numbers above before starting training.")
