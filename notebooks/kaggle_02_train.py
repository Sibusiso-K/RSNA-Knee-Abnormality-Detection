"""KAGGLE TRAINING NOTEBOOK — paste as a single cell.

Settings: Accelerator GPU T4 x2 | Internet ON (to fetch timm weights) | ~8h cap

Attach as Datasets:
  - competition data (automatic)
  - report-derived labels -> /kaggle/input/knee-labels/labels_v1.csv
  - repo src/             -> /kaggle/input/knee-src/

Outputs one .pth per fold; publish /kaggle/working as a Dataset and point
kaggle_03_submit.py at it.

TWO RULES THIS FILE EXISTS TO ENFORCE:

1. **GroupKFold on the scanner fingerprint, never random KFold.** Random folds
   overstate macro AUC by ~0.053 through site memorisation (docs/04-method.md).
   Every number this script prints is a grouped number.
2. **The 58 gold studies are never trained on.** They are the only real ground
   truth in the competition; they are held out entirely and used to check
   whether report-derived labels are actually teaching the right thing.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# --- src bootstrap (see kaggle_01_smoke.py for the full explanation) -----
# Kaggle flattens the uploaded folder, and "knee-src" isn't a legal package
# name, so rebuild a real `src` package under /kaggle/working.
import os
import shutil

PKG = "/kaggle/working/pkg"
INPUT = "/kaggle/input"


def find_dir(marker, max_depth=5):
    """Find the directory containing `marker`, anywhere under /kaggle/input.

    Kaggle's mount layout is NOT what most example code assumes. This account
    gets a nested tree -- /kaggle/input/{competitions,datasets}/... -- while
    public snippets hardcode flat /kaggle/input/<slug>/. Kaggle also strips the
    top-level folder of an uploaded dataset. Hardcoding any of that is how a
    notebook dies on its first line after queueing for ten minutes.

    Searching by CONTENT survives all of it: nesting, renames, version lag, and
    datasets attached by hand in the UI.
    """
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


print("=== resolving input paths ===")
_src = find_dir("labels.py")
COMP = find_dir("train_series.csv")
print("  src :", _src)
print("  comp:", COMP)
if _src is None or COMP is None:
    raise SystemExit(
        "Missing input. Attach knee-src (Datasets) and the competition "
        "(Competitions) in the notebook sidebar, or fix kernel-metadata.json."
    )

if not os.path.exists(PKG + "/src"):
    os.makedirs(PKG, exist_ok=True)
    shutil.copytree(_src, PKG + "/src")
sys.path.insert(0, PKG)
# -------------------------------------------------------------------------

from src.data.dicom import load_study                        # noqa: E402
from src.labels import TARGETS                                # noqa: E402
from src.model.net import KneeNet, positive_weights           # noqa: E402
from src.model.validation import (  # noqa: E402
    build_fingerprints,
    check_grouping,
    grouped_folds,
    macro_auc,
)

LABELS = os.path.join(find_dir("labels_v1.csv") or "", "labels_v1.csv")
ID = "StudyInstanceUID"

N_SLICES, SIZE = 16, 224
# BATCH=2, not 4: the smoke test measured 7.7 GB peak on a T4 (~15 GB usable)
# for a batch of 2 including the backward pass. Batch 4 would sit right on the
# ceiling and OOM partway through an epoch — the expensive way to find out.
# Gradient accumulation recovers the effective batch size for free.
# EPOCHS=8, not 10+: measured 3,500 s/epoch on a T4 (session 11), so 8 epochs
# is ~7.8 h against Kaggle's 12 h kernel ceiling — headroom for a slow queue or
# a stray retry. 10 would be ~9.7 h and one bad epoch from losing the whole run
# with no checkpoint newer than the last improvement.
# Fold 0 was still improving at epoch 3 (loss 1.08->0.74, AUC rising every
# epoch), so this is unfinished training, not a plateau.
BATCH, ACCUM, EPOCHS, LR = 2, 4, 8, 3e-4   # effective batch = 8
N_FOLDS, TRAIN_FOLDS = 5, [0]      # widen once one fold's timing is known
BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"

train_meta = pd.read_csv(f"{COMP}/train.csv")
series = pd.read_csv(f"{COMP}/train_series.csv")
labels = pd.read_csv(LABELS)

# Hold out the gold studies. Not a nicety — training on 58 hand-labelled
# studies then reporting agreement with them would be circular.
gold_mask = train_meta[TARGETS].notna().any(axis=1)
gold_uids = set(train_meta.loc[gold_mask, ID])
print(f"gold studies held out: {len(gold_uids)}")

data = labels[~labels[ID].isin(gold_uids)].reset_index(drop=True)
print(f"training pool: {len(data)} studies")

print("building scanner fingerprints (header reads only)...", flush=True)
t0 = time.time()
fingerprints = build_fingerprints(f"{COMP}/train_series", series, data[ID].tolist())
groups = data[ID].map(fingerprints).fillna("unknown").values
print(f"  {len(set(groups))} distinct fingerprints in {time.time() - t0:.0f}s")
# Tripwire: a near-unique grouping key silently degrades GroupKFold to random
# KFold. Print this BEFORE spending GPU hours on a number that would be a lie.
check_grouping(groups)

data["fold"] = grouped_folds(groups, n_splits=N_FOLDS)
print(data["fold"].value_counts().sort_index().to_dict())


class KneeDataset(Dataset):
    def __init__(self, frame, augment: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        try:
            volume = load_study(f"{COMP}/train_series", row[ID], series, N_SLICES, SIZE)
        except Exception:
            volume = np.zeros((3, N_SLICES, SIZE, SIZE), dtype=np.float32)

        if self.augment:
            # Flips only. NOT left-right on the in-plane axis for sagittal:
            # medial/lateral is the difference between two distinct labels, so
            # a horizontal flip would silently relabel the study. Vertical
            # (anterior-posterior) flips are safe here.
            if np.random.rand() < 0.5:
                volume = volume[:, :, ::-1].copy()

        y = row[TARGETS].values.astype(np.float32)
        return torch.from_numpy(volume).float(), torch.from_numpy(y)


def run_fold(fold: int) -> float:
    train_df = data[data["fold"] != fold]
    valid_df = data[data["fold"] == fold]
    print(f"\n=== fold {fold}: train {len(train_df)} / valid {len(valid_df)} ===")

    train_loader = DataLoader(
        KneeDataset(train_df, augment=True), batch_size=BATCH, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        KneeDataset(valid_df), batch_size=BATCH, shuffle=False, num_workers=2
    )

    device = "cuda"
    model = KneeNet(backbone=BACKBONE, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weights(train_df[TARGETS].values).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    # total_steps counts OPTIMISER steps, not batches — with accumulation those
    # differ by ACCUM. Getting this wrong makes OneCycleLR run off the end of
    # its schedule mid-training and raise.
    steps_per_epoch = max(1, len(train_loader) // ACCUM)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=EPOCHS * steps_per_epoch + EPOCHS
    )
    scaler = torch.cuda.amp.GradScaler()

    best = 0.0
    for epoch in range(EPOCHS):
        model.train()
        t0, total = time.time(), 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                # Scale by ACCUM so accumulated gradients average rather than
                # sum — otherwise the effective LR is ACCUM times too large.
                loss = criterion(model(x), y) / ACCUM
            scaler.scale(loss).backward()
            if (step + 1) % ACCUM != 0:
                total += loss.item() * ACCUM
                continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            total += loss.item() * ACCUM
            if step % 100 == 0:
                print(
                    f"  e{epoch} s{step}/{len(train_loader)} "
                    f"loss {loss.item() * ACCUM:.4f}",
                    flush=True,
                )

        model.eval()
        preds, truths = [], []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for x, y in valid_loader:
                preds.append(torch.sigmoid(model(x.to(device))).float().cpu().numpy())
                truths.append(y.numpy())

        score, per_label = macro_auc(np.concatenate(truths), np.concatenate(preds))
        print(
            f"  epoch {epoch}: loss {total / max(len(train_loader), 1):.4f} "
            f"grouped-CV macro AUC {score:.4f}  ({time.time() - t0:.0f}s)"
        )
        print("   " + "  ".join(f"{k}:{v:.3f}" for k, v in per_label.items()))

        if score > best:
            best = score
            torch.save(
                {"model": model.state_dict(), "backbone": BACKBONE,
                 "fold": fold, "score": score},
                f"/kaggle/working/knee_fold{fold}.pth",
            )
            print(f"   saved (best {best:.4f})")

    return best


scores = [run_fold(f) for f in TRAIN_FOLDS]
print(f"\nfold scores: {[f'{s:.4f}' for s in scores]}")
print(f"mean grouped-CV macro AUC: {np.mean(scores):.4f}")
print("\nBeat these floors or the model is not reading the images:")
print("  0.500  constant baseline      0.598  DICOM-metadata-only (site-grouped)")
