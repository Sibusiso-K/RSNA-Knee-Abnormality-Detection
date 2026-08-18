"""RECOMPUTE OOF from already-trained checkpoints — no training, inference only.

Settings on Kaggle: Accelerator GPU T4 | Internet OFF | DINOv2 attached

Why this exists: knee-train-pseudo-5f (5-fold, alpha 0.5 pseudo-labels, mean CV
0.8296) was built from knee-train-6slice-5fold rather than from the OOF-patched
knee-train-6slice-fs, so it never wrote oof_fold*.csv. Retraining 5 folds again
just to get OOF predictions would cost the same ~3h TPU as the original run for
information the checkpoints already contain. This notebook instead loads each
fold's checkpoint and runs one inference pass over its own held-out validation
set - the same computation predict() already does inside training, at a
fraction of the cost of retraining.

Attach knee-train-pseudo-5f as a kernel_source so its 5 checkpoints resolve
under /kaggle/input.

The frame and fold assignment MUST be reproduced bit-for-bit or a checkpoint
gets scored against the wrong studies. GroupKFold has no randomness given a
fixed groups array (see src/model/validation.py), and the merge is an inner
join on StudyInstanceUID - so building the frame from the SAME cache and the
SAME label file used to train these checkpoints (labels_pseudo_a5.csv)
reproduces the identical split. This is checked, not assumed: each fold's
reproduced macro AUC is printed next to the training log's own recorded score,
and they should match.
"""

import os
import sys
import glob
import time
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


PKG = "/kaggle/working/pkg"
INPUT = "/kaggle/input"


#: Directories never descended into while searching by content.
#:
#: `train_series/` holds 819,640 DICOM files across ~22,000 nested directories.
#: Walking it costs REAL MONEY here: measured on the first TPU run, two
#: find_dir calls spent ~1,100 s each traversing it - ~37 minutes of a 20 h/week
#: quota, before a single training step. The markers we look for are never
#: inside it.
SKIP_DIRS = {"train_series", "test_series", ".git", "__pycache__", "pkg"}


def find_dir(marker, max_depth=6):
    """Locate a directory by CONTENT. See kaggle_02_train.py for why."""
    roots = [INPUT, ".", "..", "/teamspace/studios/this_studio", "/data"]
    for root in roots:
        if not os.path.isdir(root):
            continue
        stack = [(root, 0)]
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
                if entry in SKIP_DIRS:
                    continue
                path = os.path.join(directory, entry)
                if os.path.isdir(path):
                    stack.append((path, depth + 1))
    return None


_src = find_dir("labels.py")
if _src and os.path.isdir("/kaggle"):
    if not os.path.exists(PKG + "/src"):
        os.makedirs(PKG, exist_ok=True)
        shutil.copytree(_src, PKG + "/src")
    sys.path.insert(0, PKG)
else:
    sys.path.insert(0, os.path.abspath("."))

from src.data.slots import IMG, N_SLOT, SLOTS          # noqa: E402
from src.labels import TARGETS                          # noqa: E402
from src.model.slotnet import SlotNet                    # noqa: E402
from src.model.validation import check_grouping, grouped_folds  # noqa: E402

ID = "StudyInstanceUID"

# --- configuration ---------------------------------------------------------
# Must match knee-train-pseudo-5f exactly, or the reproduced fold split and
# the checkpoint's own weights disagree about what "fold 0" means.
BATCH = int(os.environ.get("BATCH", "8"))
UNFREEZE_LAST = 6
POOL = os.environ.get("POOL", "cls_mean_focal")
TRAIN_FOLDS = [int(f) for f in os.environ.get("FOLDS", "0,1,2,3,4").split(",")]
N_FOLDS = 5
TRAIN_SIZE = int(os.environ.get("SIZE", str(IMG)))
UNDECIDED = 0.05

# --- device: CUDA or CPU only, deliberately never XLA -----------------------
# This is a GPU-only inference notebook (T4). Earlier versions of this script
# auto-detected torch_xla the same way the trainer does, which broke here: the
# T4 docker image still bundles torch_xla, so `import torch_xla` succeeds even
# without TPU hardware, falls back to PJRT_DEVICE=CPU, and hands back an XLA
# CPU device. torch.load(..., map_location=<that xla device>) then fails with
# "don't know how to restore data location of torch.storage.UntypedStorage
# (tagged with xla:0)" - a real run, not a transient error, confirmed by the
# fold split reproducing exactly (870/870/870/870/869, 151 fingerprints)
# before the crash. Training already moved every tensor to CPU before saving
# (`state = {k: v.cpu() for k, v in state.items()}`), so there is nothing here
# that needs XLA-aware loading. Not detecting it at all is the fix.
XLA = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"device: {device}  torch {torch.__version__}  XLA={XLA}")
if device.type == "cuda":
    log(f"gpu: {torch.cuda.get_device_name(0)}  "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# --- inputs ------------------------------------------------------------------
cache_dir = find_dir("index_train_0.csv")
if cache_dir is None:
    raise SystemExit("knee-cache not found. Attach knee-cache-6slice.")
log(f"cache: {cache_dir}")

index = pd.concat(
    [pd.read_csv(p) for p in sorted(glob.glob(f"{cache_dir}/index_train_*.csv"))],
    ignore_index=True,
)


def load_shards(pattern, mmap=False):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no shards matched {pattern}")
    if len(paths) == 1:
        return np.load(paths[0], mmap_mode="r" if mmap else None)
    return np.concatenate([np.load(p) for p in paths])


try:
    with open("/proc/meminfo") as fh:
        avail_gb = int(
            dict(l.split(":", 1) for l in fh if ":" in l)["MemAvailable"].split()[0]
        ) / 1024 ** 2
except Exception:
    avail_gb = float("inf")

MMAP = avail_gb < 14.0
if MMAP:
    log("!! low host memory - falling back to mmap")

cache = load_shards(f"{cache_dir}/cache_train_*.npy", mmap=MMAP)
mask = load_shards(f"{cache_dir}/mask_train_*.npy")
log(f"cache {cache.shape} {cache.nbytes / 1024**3:.2f} GB "
    f"({'mmap' if MMAP else 'RAM'}) | index {len(index)}")

from src.data.slots import GROUP  # noqa: E402

N_GROUPS = max(1, cache.shape[2] // GROUP)
if cache.shape[2] % GROUP:
    raise SystemExit(f"cache has {cache.shape[2]} slices/slot, not a multiple of {GROUP}")
log(f"slices/slot {cache.shape[2]} = {N_GROUPS} group(s) of {GROUP}")


def take_group(rows, g):
    return cache[rows][:, :, g * GROUP:(g + 1) * GROUP]


if not (len(index) == len(cache) == len(mask)):
    raise SystemExit(
        f"cache/index length mismatch: index {len(index)} cache {len(cache)} "
        f"mask {len(mask)}"
    )

# TRAIN labels: must be the SAME file knee-train-pseudo-5f used, so the
# frame/fold split reproduces exactly. Not the point of this run — the point
# is the fresh OOF predictions — but the split has to match or fold 0's
# checkpoint gets evaluated on the wrong studies.
LABEL_CANDIDATES = ("labels_pseudo_a5.csv",)
label_path = None
for candidate in LABEL_CANDIDATES:
    hit = find_dir(candidate)
    if hit:
        label_path = os.path.join(hit, candidate)
        break
if label_path is None:
    raise SystemExit(f"No labels found. Attach one of {LABEL_CANDIDATES}")
log(f"TRAIN labels (for fold split only): {label_path}")
labels = pd.read_csv(label_path)

# VALID labels: the fixed yardstick, same as every other run in this project.
VALID_LABEL = "labels_blend_v1.csv"
_vdir = find_dir(VALID_LABEL)
if _vdir is None:
    raise SystemExit(f"attach {VALID_LABEL} - it is the fixed yardstick for this run")
valid_labels = pd.read_csv(os.path.join(_vdir, VALID_LABEL))
log(f"VALID labels: {os.path.join(_vdir, VALID_LABEL)}  (fixed yardstick)")

train_csv = find_dir("train.csv")
gold_uids = set()
if train_csv:
    train_meta = pd.read_csv(os.path.join(train_csv, "train.csv"))
    gold = train_meta[train_meta[TARGETS].notna().any(axis=1)]
    gold_uids = set(gold[ID])
    log(f"gold studies held out: {len(gold_uids)}")

# --- align cache rows to labels, exactly as the trainer did ----------------
frame = index[[ID, "fingerprint"]].copy()
frame["row"] = np.arange(len(frame))
frame = frame.merge(labels, on=ID, how="inner")
frame = frame.merge(
    valid_labels.rename(columns={t: f"{t}__val" for t in TARGETS}),
    on=ID, how="inner",
)
log(f"studies with both pixels and labels: {len(frame)}")

is_gold = frame[ID].isin(gold_uids).values
frame = frame[~is_gold].reset_index(drop=True)
log(f"training pool (gold excluded): {len(frame)}")

groups = frame["fingerprint"].fillna("unknown").replace("", "unknown").values
log(f"{len(set(groups))} distinct scanner fingerprints")
check_grouping(groups)
frame["fold"] = grouped_folds(groups, n_splits=N_FOLDS)
log(f"fold sizes: {frame['fold'].value_counts().sort_index().to_dict()}")


def macro_auc(y_true, y_pred, drop_undecided=True):
    from sklearn.metrics import roc_auc_score

    per_label = {}
    for j, target in enumerate(TARGETS):
        y, p = y_true[:, j], y_pred[:, j]
        keep = ~np.isnan(y)
        if drop_undecided:
            keep &= np.abs(y - 0.5) >= UNDECIDED
        y_bin = (y[keep] > 0.5).astype(int)
        if keep.sum() < 10 or len(set(y_bin)) < 2:
            continue
        per_label[target] = float(roc_auc_score(y_bin, p[keep]))
    macro = float(np.mean(list(per_label.values()))) if per_label else float("nan")
    return macro, per_label


def autocast():
    return torch.autocast("cuda", enabled=device.type == "cuda")


@torch.no_grad()
def predict(model, rows):
    model.eval()
    out = []
    for start in range(0, len(rows), BATCH):
        sel = rows[start:start + BATCH]
        m = torch.from_numpy(mask[sel]).float().to(device)
        acc = None
        for g in range(N_GROUPS):
            x = torch.from_numpy(take_group(sel, g)).to(device)
            with autocast():
                logits = model(x, m, TRAIN_SIZE).float()
            acc = logits if acc is None else acc + logits
        probs = torch.sigmoid(acc / N_GROUPS)
        out.append(probs.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)


def find_dinov2():
    for base in (INPUT, ".", ".."):
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            if "config.json" in files and "dinov2" in root.lower():
                return root
    raise SystemExit(
        "DINOv2 weights not found. Attach metaresearch/dinov2 (Models)."
    )


DINOV2 = find_dinov2()
log(f"encoder: {DINOV2}")


def extract_fold(fold):
    """One inference pass over fold's held-out studies. No training."""
    valid_df = frame[frame["fold"] == fold]
    valid_rows = valid_df["row"].values
    log(f"=== fold {fold}: {len(valid_rows)} held-out studies ===")

    ckpt_dir = None
    for base in (INPUT, ".", ".."):
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            if f"knee_slot_fold{fold}.pth" in files:
                ckpt_dir = root
                break
        if ckpt_dir:
            break
    if ckpt_dir is None:
        raise SystemExit(
            f"knee_slot_fold{fold}.pth not found - attach knee-train-pseudo-5f"
        )
    ckpt_path = os.path.join(ckpt_dir, f"knee_slot_fold{fold}.pth")
    # map_location="cpu", not `device`: training already moved every tensor to
    # CPU before saving, so there is no XLA- or CUDA-tagged storage in the
    # file to restore. Loading to CPU first and moving the constructed model
    # to `device` afterward sidesteps map_location entirely, which is the
    # actual fix for the "don't know how to restore data location" crash.
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    log(f"  loaded {ckpt_path}: training-log CV {blob.get('score', float('nan')):.4f} "
        f"epoch {blob.get('epoch')} labels {blob.get('labels')}")

    model = SlotNet(DINOV2, unfreeze_last=UNFREEZE_LAST, pool=POOL,
                     head=blob.get("head", "xattn"))
    model.load_state_dict(blob["model"])
    model.to(device)
    model.eval()

    pred = predict(model, valid_rows)
    valid_y = valid_df[[f"{t}__val" for t in TARGETS]].values.astype(np.float32)
    score, per_label = macro_auc(valid_y, pred)
    log(f"  reproduced AUC (labels_blend_v1 yardstick): {score:.4f}  "
        f"(training log recorded {blob.get('score', float('nan')):.4f})")

    oof = pd.DataFrame(pred, columns=TARGETS)
    oof.insert(0, ID, valid_df[ID].values)
    oof["fold"] = fold
    oof.to_csv(f"oof_fold{fold}.csv", index=False)
    return score


scores = [extract_fold(f) for f in TRAIN_FOLDS]
log(f"mean reproduced macro AUC: {np.mean(scores):.4f}")
