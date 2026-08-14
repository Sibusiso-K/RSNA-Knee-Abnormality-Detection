"""TRAINING on the cached slot representation — Kaggle or anywhere else.

Settings on Kaggle: Accelerator GPU T4 x2 | Internet OFF | DINOv2 attached

Attach:
  - knee-cache-v1  -> cache_train_*.npy, mask_train_*.npy, index_train_*.csv
  - knee-labels    -> the label CSV chosen by scripts/compare_labels.py
  - metaresearch/dinov2 (Models)

**This script never opens a DICOM.** Everything it needs — pixels, presence
mask, scanner fingerprint — is in the cache and its index. That is deliberate:
the 570 GB of DICOMs exist only on Kaggle, so a training job that needs them is
chained to Kaggle's 30 h/week. This one runs unchanged on Lightning, Colab or a
rented box, because the whole input is ~9 GB.

Two rules carried over from kaggle_02_train.py, both load-bearing:

1. **GroupKFold on the scanner fingerprint, never random KFold.** Random folds
   overstate macro AUC by ~0.053 through site memorisation.
2. **The 58 gold studies are never trained on.** They are the only real ground
   truth here and are reported as an independent check, not used for selection —
   58 studies cannot resolve differences below ~0.02.
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
#: find_dir calls spent ~1,100 s each traversing it — ~37 minutes of a 20 h/week
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

# --- configuration -------------------------------------------------------
# Matches the public strong-pipeline recipe where it is proven, so the things
# we changed (labels, laterality, grouped CV) are the variables under test.
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BATCH = int(os.environ.get("BATCH", "8"))          # studies; each is 6 slot images
LR_HEAD = 1e-3
LR_BACKBONE = 8e-6      # adapt, do not retrain — a single 1e-3 destroys DINOv2
UNFREEZE_LAST = 6
WEIGHT_DECAY = 0.02
VARIANT = os.environ.get("VARIANT", "small")
POOL = os.environ.get("POOL", "cls_mean_focal")
TRAIN_FOLDS = [int(f) for f in os.environ.get("FOLDS", "0,1,2,3,4").split(",")]
N_FOLDS = 5
TRAIN_SIZE = int(os.environ.get("SIZE", str(IMG)))
AUG_ROT_DEG, AUG_SCALE, AUG_SHIFT, AUG_INTENSITY = 8.0, 0.08, 0.05, 0.10

# A cell this close to 0.5 is the labeller saying "the report does not address
# this", not a weak positive. Excluded from the validation AUC: scoring against
# a coerced 0 would measure how well the model reproduces silence.
UNDECIDED = 0.05

# --- device: CUDA, XLA/TPU, or CPU --------------------------------------
# One script for all three. The model, labels, folds and gold holdout are the
# parts that must not diverge between runs, so they stay shared; only the four
# lines that differ per backend are branched.
#
# TPU is not a fallback here. Measured 2026-08-12 on a Kaggle v5e-8: both
# risky ops lower (grid_sample, topk), and one core sustains 63.6 study/s
# warm after a ~10 s compile. The 20 h/week TPU quota is also on a completely
# separate budget from the GPU's 30 h.
XLA = False
xm = None
if os.environ.get("USE_XLA", "auto") != "0":
    try:
        import torch_xla.core.xla_model as _xm

        xm = _xm
        device = xm.xla_device()
        XLA = True
    except Exception:
        XLA = False

if not XLA:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log(f"device: {device}  torch {torch.__version__}  XLA={XLA}")
if XLA:
    import torch_xla

    log(f"torch_xla {getattr(torch_xla, '__version__', '?')} | "
        f"cores visible {len(xm.get_xla_supported_devices())} (using one)")
elif device.type == "cuda":
    log(f"gpu: {torch.cuda.get_device_name(0)}  "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# --- inputs --------------------------------------------------------------
cache_dir = find_dir("index_train_0.csv")
if cache_dir is None:
    raise SystemExit("knee-cache not found. Attach it, or run kaggle_05_cache.py.")
log(f"cache: {cache_dir}")

index = pd.concat(
    [pd.read_csv(p) for p in sorted(glob.glob(f"{cache_dir}/index_train_*.csv"))],
    ignore_index=True,
)
def load_shards(pattern, mmap=False):
    """Concatenate shards — but never copy a single one.

    `np.concatenate` always allocates a new array, so wrapping a lone 8.96 GB
    shard in it doubles peak RAM to ~18 GB to produce a byte-identical result.
    With one shard (the current layout) this returns the array as loaded.
    """
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
    log(f"host memory available: {avail_gb:.1f} GB")
except Exception:
    avail_gb = float("inf")

# Fall back to a memory map when the cache would not comfortably fit. Random
# gathers from a network-mounted mmap are much slower than from RAM, so this
# is a last resort that keeps the run alive rather than a default.
MMAP = avail_gb < 14.0
if MMAP:
    log("!! low host memory — falling back to mmap; expect slower steps")

cache = load_shards(f"{cache_dir}/cache_train_*.npy", mmap=MMAP)
mask = load_shards(f"{cache_dir}/mask_train_*.npy")
log(f"cache {cache.shape} {cache.nbytes / 1024**3:.2f} GB "
    f"({'mmap' if MMAP else 'RAM'}) | index {len(index)}")

# The encoder takes THREE channels - a 2.5D triplet - so a cache holding more
# slices per slot holds several GROUPS of three, not one wide image. Training
# draws one group per step (which also acts as augmentation along the stack)
# and inference averages over all of them. With one group the two coincide,
# which is why this was invisible until the 6-slice cache arrived and the ViT
# was handed six channels.
from src.data.slots import GROUP                             # noqa: E402

N_GROUPS = max(1, cache.shape[2] // GROUP)
if cache.shape[2] % GROUP:
    raise SystemExit(f"cache has {cache.shape[2]} slices/slot, not a multiple of {GROUP}")
log(f"slices/slot {cache.shape[2]} = {N_GROUPS} group(s) of {GROUP}")


def take_group(rows, g):
    """Slice GROUP consecutive channels out of the cached slices."""
    return cache[rows][:, :, g * GROUP:(g + 1) * GROUP]
if not (len(index) == len(cache) == len(mask)):
    raise SystemExit(
        f"cache/index length mismatch: index {len(index)} cache {len(cache)} "
        f"mask {len(mask)} — a partial shard was published"
    )

LABEL_CANDIDATES = ("labels_blend_v1.csv", "llm_labels_v4_blend.csv", "labels_v1.csv")
label_path = None
for candidate in LABEL_CANDIDATES:
    hit = find_dir(candidate)
    if hit:
        label_path = os.path.join(hit, candidate)
        break
if label_path is None:
    raise SystemExit(f"No labels found. Attach one of {LABEL_CANDIDATES}")
log(f"LABELS: {label_path}")
labels = pd.read_csv(label_path)

train_csv = find_dir("train.csv")
gold_uids = set()
if train_csv:
    train_meta = pd.read_csv(os.path.join(train_csv, "train.csv"))
    gold = train_meta[train_meta[TARGETS].notna().any(axis=1)]
    gold_uids = set(gold[ID])
    gold_truth = gold.set_index(ID)[TARGETS]
    log(f"gold studies held out: {len(gold_uids)}")

# --- align cache rows to labels -----------------------------------------
frame = index[[ID, "fingerprint"]].copy()
frame["row"] = np.arange(len(frame))
frame = frame.merge(labels, on=ID, how="inner")
log(f"studies with both pixels and labels: {len(frame)}")

is_gold = frame[ID].isin(gold_uids).values
gold_frame = frame[is_gold].reset_index(drop=True)
frame = frame[~is_gold].reset_index(drop=True)
log(f"training pool {len(frame)} | gold held out {len(gold_frame)}")

groups = frame["fingerprint"].fillna("unknown").replace("", "unknown").values
log(f"{len(set(groups))} distinct scanner fingerprints")
check_grouping(groups)
frame["fold"] = grouped_folds(groups, n_splits=N_FOLDS)
log(f"fold sizes: {frame['fold'].value_counts().sort_index().to_dict()}")

Y = frame[TARGETS].values.astype(np.float32)
log(f"label range [{np.nanmin(Y):.2f}, {np.nanmax(Y):.2f}], "
    f"undecided cells {np.mean(np.abs(Y - 0.5) < UNDECIDED):.1%}")


def augment(imgs):
    """Rigid jitter plus an intensity scale, per slot image, on the GPU.

    **Neither flip is used, and for different reasons.**

    A horizontal flip would reintroduce exactly the nuisance axis that the
    laterality normalisation was built to remove — undoing, once per batch, the
    thing that fixes five of the twelve targets.

    A vertical flip is not a nuisance axis at all, and this is the bug
    kaggle_02_train.py had: it flipped anterior-posterior on half of all
    samples. A knee is acquired in a canonical orientation and no study looks
    like its own vertical mirror. Where a finding sits in the frame is
    information, not noise — a Baker's cyst is identified by lying in the
    popliteal fossa and PF OA by the patella being anterior. Randomising that
    destroys the feature.
    """
    lead = imgs.shape[:-3]
    x = imgs.reshape(-1, *imgs.shape[-3:]).float()
    n = x.shape[0]

    rot = (torch.rand(n, device=x.device) - 0.5) * 2 * (AUG_ROT_DEG * np.pi / 180)
    # Zoom IN only: `border` padding repeats the edge outward, and the edge of a
    # 130 mm crop is where the popliteal fossa sits. Zooming out would fabricate
    # tissue precisely where a Baker's cyst is looked for.
    scale = 1.0 + torch.rand(n, device=x.device) * AUG_SCALE
    tx = (torch.rand(n, device=x.device) - 0.5) * 2 * AUG_SHIFT
    ty = (torch.rand(n, device=x.device) - 0.5) * 2 * AUG_SHIFT
    cos, sin = torch.cos(rot) / scale, torch.sin(rot) / scale
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                      align_corners=False)

    gain = 1.0 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 2 * AUG_INTENSITY
    return (x * gain).clamp(0, 255).reshape(*lead, *x.shape[-3:])


def macro_auc(y_true, y_pred, drop_undecided=True):
    """Macro AUC, skipping cells the report never addressed.

    A label at exactly 0.5 means the finding was not mentioned. Binarising it to
    0 and scoring against it measures agreement with silence, which for Baker's
    cyst is nearly the label and for Synovitis is nearly nothing (83.7% of that
    column is unaddressed). Dropping those cells scores the model where the
    report actually committed.
    """
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
    """Mixed precision per backend.

    CUDA gets fp16 with a GradScaler. XLA gets bf16, which has fp32's exponent
    range and therefore needs no loss scaling at all — the GradScaler exists to
    stop fp16 gradients underflowing, and bf16 does not underflow the same way.
    """
    if XLA:
        return torch.autocast("xla", dtype=torch.bfloat16)
    return torch.autocast("cuda", enabled=device.type == "cuda")


@torch.no_grad()
def predict(model, rows):
    model.eval()
    out = []
    for start in range(0, len(rows), BATCH):
        sel = rows[start : start + BATCH]
        m = torch.from_numpy(mask[sel]).float().to(device)
        acc = None
        for g in range(N_GROUPS):
            x = torch.from_numpy(take_group(sel, g)).to(device)
            with autocast():
                logits = model(x, m, TRAIN_SIZE).float()
            acc = logits if acc is None else acc + logits
        probs = torch.sigmoid(acc / N_GROUPS)
        if XLA:
            xm.mark_step()
        out.append(probs.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)


def find_dinov2():
    """Locate a mounted DINOv2 checkpoint directory, or fail loudly.

    Deliberately does NOT fall back to "any directory containing a
    config.json". An earlier version did, and that is a silent-wrong-answer
    bug: it would happily hand `AutoModel.from_pretrained` some unrelated
    model, which loads, trains, and produces a plausible-looking score from an
    encoder nobody chose. Not finding the weights must stop the run.
    """
    for base in (INPUT, ".", ".."):
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            if "config.json" in files and "dinov2" in root.lower():
                return root
    raise SystemExit(
        "DINOv2 weights not found. Attach metaresearch/dinov2 (Models), or "
        "place the checkpoint in a directory whose path contains 'dinov2'."
    )


DINOV2 = find_dinov2()
log(f"encoder: {DINOV2}")


def run_fold(fold):
    train_df = frame[frame["fold"] != fold]
    valid_df = frame[frame["fold"] == fold]
    log(f"\n=== fold {fold}: train {len(train_df)} / valid {len(valid_df)} ===")
    dinov2 = DINOV2

    model = SlotNet(dinov2, unfreeze_last=UNFREEZE_LAST, pool=POOL, head="xattn").to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Log the hidden size, not just the path. "small" and "base" mount at
    # similar-looking paths, differ by 2x in feature width and ~4x in
    # activation memory, and both run — so the only way to know afterwards
    # which one a checkpoint came from is to have written it down.
    log(f"encoder hidden size {model.vit.config.hidden_size} "
        f"({len(model.vit.encoder.layer)} blocks, last {UNFREEZE_LAST} open) "
        f"| trainable {trainable / 1e6:.1f}M | pool {POOL}")

    optimizer = torch.optim.AdamW(
        model.param_groups(LR_HEAD, LR_BACKBONE), weight_decay=WEIGHT_DECAY
    )
    steps = max(1, len(train_df) // BATCH)
    # max_lr MUST be a per-group list. A scalar overwrites every group's rate
    # and silently retrains the ViT at 1e-3, which defeats the whole design.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[LR_BACKBONE, LR_HEAD],
        total_steps=EPOCHS * steps + EPOCHS,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(not XLA) and device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()

    train_rows = train_df["row"].values
    train_y = train_df[TARGETS].values.astype(np.float32)
    valid_rows = valid_df["row"].values
    valid_y = valid_df[TARGETS].values.astype(np.float32)
    gold_rows = gold_frame["row"].values if len(gold_frame) else np.array([], int)

    best = 0.0
    for epoch in range(EPOCHS):
        model.train()
        order = np.random.permutation(len(train_rows))
        running, t_epoch = None, time.time()
        for step in range(steps):
            sel = order[step * BATCH : (step + 1) * BATCH]
            if len(sel) == 0:
                continue
            rows = train_rows[sel]
            g = np.random.randint(N_GROUPS)
            x = torch.from_numpy(take_group(rows, g)).to(device)
            m = torch.from_numpy(mask[rows]).float().to(device)
            y = torch.from_numpy(train_y[sel]).to(device)

            x = augment(x)
            with autocast():
                # Soft targets are used as-is. BCE against a target of 0.28
                # expresses "probably absent but the report did not say", which
                # is the information a hard 0 throws away.
                loss = criterion(model(x, m, TRAIN_SIZE), y)

            if XLA:
                loss.backward()
                # xm.optimizer_step inserts the cross-replica reduction (a
                # no-op on one core, correct on eight) and applies the update.
                xm.optimizer_step(optimizer)
                # mark_step is NOT optional and NOT implied by the line above:
                # xm.optimizer_step defaults to barrier=False and does not cut
                # the graph. Without this, XLA keeps tracing lazily and the
                # graph grows until something forces evaluation — which here
                # was loss.item() every 50 steps, so it compiled FIFTY steps of
                # training as one graph. Measured cost of omitting it: 12.2
                # s/step instead of 0.12, then
                #   "Ran out of memory in memory space hbm.
                #    Used 16.36G of 15.75G"
                # The probe did not hit this only because it passed
                # barrier=True, which calls mark_step internally.
                xm.mark_step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # Accumulate on-device. `.item()` forces a host sync, and on XLA
            # that stalls the pipeline every step for a number only printed
            # every fiftieth — which turns a compute-bound loop into a
            # latency-bound one.
            running = loss.detach() if running is None else running + loss.detach()
            if step % 50 == 0:
                log(f"  e{epoch} {step}/{steps} loss {float(loss.item()):.4f}")

        epoch_loss = float(running.item()) / max(steps, 1) if running is not None else float("nan")
        score, per_label = macro_auc(valid_y, predict(model, valid_rows))
        line = (f"  epoch {epoch}: loss {epoch_loss:.4f}  "
                f"grouped-CV macro AUC {score:.4f}  ({time.time() - t_epoch:.0f}s)")
        if len(gold_rows):
            gold_pred = predict(model, gold_rows)
            gold_y = gold_truth.loc[gold_frame[ID]].values.astype(np.float32)
            gold_score, _ = macro_auc(gold_y, gold_pred, drop_undecided=False)
            line += f"  | gold58 {gold_score:.4f}"
        log(line)
        log("   " + "  ".join(f"{k}:{v:.3f}" for k, v in sorted(per_label.items())))

        if score > best:
            best = score
            # On XLA the state dict holds device tensors. Move to CPU before
            # saving so the checkpoint loads anywhere - the submission notebook
            # runs on a GPU and must not need torch_xla to read this file.
            state = model.state_dict()
            if XLA:
                state = {k: v.cpu() for k, v in state.items()}
            torch.save(
                {
                    "model": state,
                    "device": "xla" if XLA else device.type,
                    "encoder": os.path.basename(dinov2),
                    "variant": VARIANT, "pool": POOL, "size": TRAIN_SIZE,
                    "slots": [s[0] for s in SLOTS],
                    "fold": fold, "score": score, "epoch": epoch,
                    "labels": os.path.basename(label_path),
                    "epochs": EPOCHS, "n_groups": int(len(set(groups))),
                },
                f"knee_slot_fold{fold}.pth",
            )
            log(f"   saved (best {best:.4f})")
    return best


scores = [run_fold(f) for f in TRAIN_FOLDS]
log(f"\nfold scores: {[f'{s:.4f}' for s in scores]}")
log(f"mean grouped-CV macro AUC: {np.mean(scores):.4f}")
log("\nFloors — beat these or the model is not reading the images:")
log("  0.500 constant   0.598 DICOM-metadata-only (site-grouped)   "
    "0.775 our EfficientNet 2.5D baseline")
