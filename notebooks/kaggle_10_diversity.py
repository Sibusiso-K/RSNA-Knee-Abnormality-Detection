"""KAGGLE DIVERSITY PROBE — does a second architecture add anything?

Settings: Accelerator **TPU** | Internet OFF

The question, and why it is worth a kernel of its own
-----------------------------------------------------
Three capacity levers in a row returned nothing: 8-vs-4 epochs on the old
pipeline, 20-vs-10 on this one, and DINOv2-base against small (-0.0005 on the
same fold). So a bigger or longer-trained member is not the answer.

But the public 0.899 is a rank-mean of SEVERAL members, and ours is five folds
of ONE configuration. That suggests the missing ingredient is diversity rather
than size — and base, despite scoring the same as small, is a different model
whose errors may be decorrelated.

Testing that by training base on all five folds would cost ~3.5 h of a 20 h
weekly budget on a hunch. It does not need to: small-fold0 and base-fold0 were
validated on the SAME held-out fold, so their predictions can simply be
combined and scored. Twenty minutes answers it.

What it prints
--------------
    small alone, base alone, and the rank-mean of the two
    plus the per-label correlation between them

If the ensemble beats both, diversity pays and base x 5 folds is worth the
3.5 h. If it does not, that is 3.5 h not spent, and the next lever is the
448 px cache rather than another architecture.
"""

import os
import sys
import glob
import time
import shutil

import numpy as np
import pandas as pd
import torch

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


PKG = "/kaggle/working/pkg"
INPUT = "/kaggle/input"
ID = "StudyInstanceUID"
SKIP_DIRS = {"train_series", "test_series", ".git", "__pycache__"}


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
            if entry in SKIP_DIRS:
                continue
            path = os.path.join(directory, entry)
            if os.path.isdir(path):
                stack.append((path, depth + 1))
    return None


_src = find_dir("labels.py")
if not os.path.exists(PKG + "/src"):
    os.makedirs(PKG, exist_ok=True)
    shutil.copytree(_src, PKG + "/src")
sys.path.insert(0, PKG)

from src.labels import TARGETS                                # noqa: E402
from src.model.slotnet import SlotNet                          # noqa: E402
from src.model.validation import check_grouping, grouped_folds  # noqa: E402

try:
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    XLA = True
except Exception:
    device, XLA, xm = torch.device("cpu"), False, None
log(f"device {device} XLA={XLA}")

cache_dir = find_dir("index_train_0.csv")
index = pd.read_csv(f"{cache_dir}/index_train_0.csv")
cache = np.load(glob.glob(f"{cache_dir}/cache_train_*.npy")[0])
mask = np.load(glob.glob(f"{cache_dir}/mask_train_*.npy")[0])
log(f"cache {cache.shape}")

label_dir = find_dir("labels_blend_v1.csv")
labels = pd.read_csv(f"{label_dir}/labels_blend_v1.csv")
comp = find_dir("train.csv")
train_meta = pd.read_csv(f"{comp}/train.csv")
gold_uids = set(train_meta[train_meta[TARGETS].notna().any(axis=1)][ID])

frame = index[[ID, "fingerprint"]].copy()
frame["row"] = np.arange(len(frame))
frame = frame.merge(labels, on=ID, how="inner")
frame = frame[~frame[ID].isin(gold_uids)].reset_index(drop=True)

groups = frame["fingerprint"].fillna("").replace("", "unknown").values
check_grouping(groups)
frame["fold"] = grouped_folds(groups, n_splits=5)

# Fold 0 EXACTLY as the training runs defined it. Both checkpoints were
# validated on this split, which is the whole reason they are comparable.
valid = frame[frame["fold"] == 0].reset_index(drop=True)
log(f"fold 0 validation: {len(valid)} studies")

valid_rows = valid["row"].values
Y = valid[TARGETS].values.astype(np.float32)

UNDECIDED = 0.05


def macro_auc(y_true, y_pred):
    from sklearn.metrics import roc_auc_score

    per = {}
    for j, target in enumerate(TARGETS):
        y, p = y_true[:, j], y_pred[:, j]
        keep = (~np.isnan(y)) & (np.abs(y - 0.5) >= UNDECIDED)
        y_bin = (y[keep] > 0.5).astype(int)
        if keep.sum() < 10 or len(set(y_bin)) < 2:
            continue
        per[target] = float(roc_auc_score(y_bin, p[keep]))
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def encoder_for(hidden):
    for root, dirs, files in os.walk(INPUT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        if "config.json" in files and "dinov2" in root.lower():
            import json

            with open(os.path.join(root, "config.json")) as fh:
                if int(json.load(fh).get("hidden_size", -1)) == hidden:
                    return root
    raise SystemExit(f"no mounted DINOv2 with hidden_size {hidden}")


@torch.no_grad()
def predict(net, rows, batch=8):
    net.eval()
    out = []
    for start in range(0, len(rows), batch):
        sel = rows[start : start + batch]
        x = torch.from_numpy(cache[sel]).to(device)
        m = torch.from_numpy(mask[sel]).float().to(device)
        p = torch.sigmoid(net(x, m).float())
        if XLA:
            xm.mark_step()
        out.append(p.cpu().numpy())
    return np.concatenate(out)


preds = {}
for path in sorted(glob.glob(f"{INPUT}/**/knee_slot_fold0.pth", recursive=True)):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    hidden = int(blob["model"]["vit.embeddings.cls_token"].shape[-1])
    name = f"{'base' if hidden == 768 else 'small'}"
    if name in preds:
        continue
    net = SlotNet(encoder_for(hidden), pool=blob.get("pool", "cls_mean_focal"))
    net.load_state_dict(blob["model"])
    net.to(device)
    log(f"{name}: hidden {hidden}, recorded CV {blob.get('score', float('nan')):.4f}")
    preds[name] = predict(net, valid_rows)
    score, _ = macro_auc(Y, preds[name])
    log(f"  {name} measured here: {score:.4f}")
    del net

if len(preds) < 2:
    raise SystemExit(f"need two members, found {list(preds)}")

log("\n=== does diversity pay? ===")
singles = {}
for name, p in preds.items():
    singles[name], _ = macro_auc(Y, p)
    log(f"  {name:6s} {singles[name]:.4f}")

# Rank-mean, the same combiner the submission uses: the metric reads order.
ranked = np.zeros_like(next(iter(preds.values())))
for p in preds.values():
    ranked += pd.DataFrame(p, columns=list(TARGETS)).rank(pct=True).to_numpy(np.float32)
ranked /= len(preds)
ens, per = macro_auc(Y, ranked)

best_single = max(singles.values())
log(f"  ensemble {ens:.4f}   (best single {best_single:.4f}, "
    f"delta {ens - best_single:+.4f})")
log("  per label: " + "  ".join(f"{k}:{v:.3f}" for k, v in sorted(per.items())))

# How decorrelated are they really? Two members that agree everywhere cannot
# help each other no matter how good either is.
names = list(preds)
corr = [
    float(pd.Series(preds[names[0]][:, j]).corr(
        pd.Series(preds[names[1]][:, j]), method="spearman"))
    for j in range(len(TARGETS))
]
log(f"  per-label Spearman between members: mean {np.mean(corr):.3f} "
    f"min {np.min(corr):.3f} max {np.max(corr):.3f}")

log("\n=== verdict ===")
if ens - best_single >= 0.010:
    log(f"Diversity PAYS (+{ens - best_single:.4f} from two members on one fold). "
        f"base x 5 folds is worth ~3.5 h.")
else:
    log(f"Diversity does NOT pay here ({ens - best_single:+.4f}). Do not spend "
        f"3.5 h on base x 5; the 448 px cache is the next lever instead.")
