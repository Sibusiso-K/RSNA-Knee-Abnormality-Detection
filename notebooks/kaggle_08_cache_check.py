"""KAGGLE CACHE AUDIT — CPU, free, ~2 minutes.

Settings: Accelerator **NONE (CPU)** | Internet OFF

Answers, before a single GPU-second is spent, the questions that decide whether
the training run means anything:

1. **Are the scanner fingerprints usable as CV groups?** This is the one that
   matters most. `docs/04-method.md` records that DICOM metadata alone reaches
   0.6516 macro AUC under random folds but 0.5981 under scanner-grouped folds.
   If the fingerprints came out near-unique, GroupKFold silently degrades to
   random KFold and hands back that 0.053 as free-looking score. It has already
   happened once in this project: raw `ImagingFrequency` produced 3,229 groups
   across 4,349 studies.

2. **Do the cache rows line up with the labels?** A merge that drops half the
   corpus still trains, still prints an AUC, and is worthless.

3. **Are the 58 gold studies present and holdable?** They are the only real
   ground truth and must be excluded from training.

4. **Is any study empty?** A row with no slots is 6 black images, and the
   encoder maps black to a confident feature vector rather than to "unknown".

Reading 1.85 MB of CSV to protect a run that costs an eighth of the weekly GPU
budget is the cheapest insurance available.
"""

import os
import sys
import glob
import shutil

import numpy as np
import pandas as pd

PKG = "/kaggle/working/pkg"
INPUT = "/kaggle/input"
ID = "StudyInstanceUID"


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
if _src and not os.path.exists(PKG + "/src"):
    os.makedirs(PKG, exist_ok=True)
    shutil.copytree(_src, PKG + "/src")
sys.path.insert(0, PKG)

from src.data.slots import SLOTS                              # noqa: E402
from src.labels import TARGETS                                 # noqa: E402
from src.model.validation import check_grouping, grouped_folds  # noqa: E402

cache_dir = find_dir("index_train_0.csv")
comp = find_dir("train.csv")
print(f"cache: {cache_dir}")
print(f"comp : {comp}")
if cache_dir is None:
    raise SystemExit("knee-cache output not attached")

index = pd.concat(
    [pd.read_csv(p) for p in sorted(glob.glob(f"{cache_dir}/index_train_*.csv"))],
    ignore_index=True,
)
masks = np.concatenate(
    [np.load(p) for p in sorted(glob.glob(f"{cache_dir}/mask_train_*.npy"))]
)
print(f"\nindex rows {len(index)} | mask {masks.shape}")
if len(index) != len(masks):
    raise SystemExit(f"MISMATCH: index {len(index)} vs mask {len(masks)}")

# --- 4. empty studies ----------------------------------------------------
empty = int((masks.sum(axis=1) == 0).sum())
print(f"\nstudies with ZERO usable slots: {empty} ({100.0 * empty / len(masks):.2f}%)")
if empty:
    print("  these are 6 black images each; the encoder reads black as a "
          "confident feature, not as 'unknown'. They must be dropped, not fed.")
print("slot fill per slot:")
for i, spec in enumerate(SLOTS):
    print(f"  {spec[0]:16s} {masks[:, i].mean():.3f}")
print(f"slots per study: mean {masks.sum(axis=1).mean():.2f} "
      f"min {masks.sum(axis=1).min()} max {masks.sum(axis=1).max()}")

# --- 1. fingerprints as CV groups ---------------------------------------
fp = index["fingerprint"].fillna("").replace("", "unknown")
counts = fp.value_counts()
print(f"\n=== scanner fingerprints ===")
print(f"distinct: {fp.nunique()} across {len(fp)} studies "
      f"(ratio {fp.nunique() / len(fp):.3f})")
print(f"singletons: {int((counts == 1).sum())}")
print(f"largest groups: {counts.head(8).tolist()}")
print(f"top-20 groups cover {100.0 * counts.head(20).sum() / len(fp):.1f}% of studies")
check_grouping(fp.values)

# --- 3. gold studies -----------------------------------------------------
gold_uids = set()
if comp:
    train_meta = pd.read_csv(os.path.join(comp, "train.csv"))
    gold_uids = set(train_meta[train_meta[TARGETS].notna().any(axis=1)][ID])
    present = len(gold_uids & set(index[ID]))
    print(f"\ngold studies: {len(gold_uids)} total, {present} present in the cache")
    if present != len(gold_uids):
        print("  !! some gold studies are missing from the cache — the "
              "independent check will be computed on fewer than 58")

# --- 2. label alignment --------------------------------------------------
label_path = None
for candidate in ("labels_blend_v1.csv", "labels_v1.csv"):
    hit = find_dir(candidate)
    if hit:
        label_path = os.path.join(hit, candidate)
        break
if label_path:
    labels = pd.read_csv(label_path)
    merged = index[[ID]].merge(labels, on=ID, how="inner")
    print(f"\nlabels: {os.path.basename(label_path)}")
    print(f"  cache {len(index)} | labels {len(labels)} | merged {len(merged)}")
    if len(merged) < 0.95 * len(index):
        print("  !! the merge drops >5% of the cache — a silent corpus cut")
    trainable = merged[~merged[ID].isin(gold_uids)]
    print(f"  training pool after holding out gold: {len(trainable)}")

    groups = index.set_index(ID).loc[trainable[ID], "fingerprint"]
    groups = groups.fillna("").replace("", "unknown").values
    folds = grouped_folds(groups, n_splits=5)
    sizes = pd.Series(folds).value_counts().sort_index().to_dict()
    print(f"  fold sizes: {sizes}")
    spread = max(sizes.values()) / max(min(sizes.values()), 1)
    print(f"  largest/smallest fold ratio: {spread:.2f}")
    if spread > 2.0:
        print("  !! folds are badly unbalanced — a few huge scanner groups "
              "dominate, and per-fold AUCs will not be comparable")
else:
    print("\n!! no label file attached — cannot check alignment")

print("\n=== audit complete ===")
