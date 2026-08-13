"""KAGGLE SUBMISSION NOTEBOOK — slot pipeline.

Settings that must be right or the submit button stays greyed out:
  Accelerator: GPU T4 x2  |  Internet: OFF  |  Output: submission.csv

Attach:
  - the competition data (automatic)
  - knee-src           -> src/
  - knee-slot-model-v1 -> knee_slot_fold*.pth
  - metaresearch/dinov2 (Models)

The test cache is built HERE, at inference time, through `src/data/cache.py` —
the same module that built the training cache. That is not a convenience: the
two caches are produced weeks apart in different notebooks, and if the decode
paths ever diverge the weights are applied to pixels they were never trained
on, with nothing in the pipeline to complain. One module, one code path.

Cost: ~1,300 test studies at the measured 0.57 study/s is ~38 minutes, against
the 9 h notebook cap. The Efficiency track charges full wall time, but
docs/01-competition.md establishes that ~0.045 AUC is worth about an hour, so
buying resolution with minutes here is the right side of that trade.

DEGRADED-MODE BEHAVIOUR IS DELIBERATE: a study that fails to decode keeps 0.5
rather than raising. A submission that scores 0.5 on one row still scores; a
notebook that throws scores nothing and burns a slot. Every fallback is
counted and printed, so a silent degradation cannot masquerade as a result.
"""

import os
import sys
import glob
import time
import shutil
import traceback

import numpy as np
import pandas as pd
import torch

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


PKG = "/kaggle/working/pkg"
INPUT = "/kaggle/input"
ID = "StudyInstanceUID"


#: Never descended into while searching by content. `test_series/` holds the
#: hidden test DICOMs across thousands of nested directories; walking it cost
#: ~1,100 s per call on the training side, and here that comes straight off the
#: 9 h submission cap. The markers we look for are never inside it.
SKIP_DIRS = {"train_series", "test_series", ".git", "__pycache__"}


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
            if entry in SKIP_DIRS:
                continue
            path = os.path.join(directory, entry)
            if os.path.isdir(path):
                stack.append((path, depth + 1))
    return None


COMP = find_dir("test_series.csv")
_src = find_dir("labels.py")
if COMP is None:
    raise SystemExit("competition data not attached")
if _src and not os.path.exists(PKG + "/src"):
    os.makedirs(PKG, exist_ok=True)
    shutil.copytree(_src, PKG + "/src")
sys.path.insert(0, PKG)

from src.data.cache import N_SLICE, build_study      # noqa: E402
from src.data.slots import IMG, N_SLOT               # noqa: E402
from src.labels import TARGETS                        # noqa: E402

test = pd.read_csv(f"{COMP}/test.csv")
test_series = pd.read_csv(f"{COMP}/test_series.csv")
submission = pd.DataFrame({ID: test[ID]})
for target in TARGETS:
    submission[target] = 0.5

log(f"test: {len(test)} studies, {len(test_series)} series")


def write_and_exit(reason):
    """Always leave a valid submission.csv behind, whatever went wrong."""
    submission.to_csv("submission.csv", index=False)
    log(f"WROTE fallback submission.csv ({reason})")
    print(pd.read_csv("submission.csv").head())
    raise SystemExit(0)


try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")

    from src.model.slotnet import SlotNet

    def find_dinov2(hidden=None):
        """Locate a mounted DINOv2 whose width matches the checkpoint.

        Matching on `hidden_size` rather than taking the first hit is the fix
        for a real failure: the submission kernel mounted DINOv2-**base**
        (768) while the checkpoint had been trained on **small** (384), and
        load_state_dict died on every parameter. That one failed loudly only
        because the shapes disagree — if two variants ever share a width, the
        first-hit version would load the wrong weights silently and score a
        plausible-looking number from an encoder nobody chose.
        """
        candidates = []
        for root, dirs, files in os.walk(INPUT):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            if "config.json" in files and "dinov2" in root.lower():
                try:
                    import json

                    with open(os.path.join(root, "config.json")) as fh:
                        size = int(json.load(fh).get("hidden_size", -1))
                except Exception:
                    size = -1
                candidates.append((root, size))
        if not candidates:
            return None
        log(f"dinov2 candidates: {candidates}")
        if hidden is not None:
            for root, size in candidates:
                if size == hidden:
                    return root
            log(f"!! no mounted DINOv2 has hidden_size {hidden}; "
                f"attach the matching variant")
            return None
        return candidates[0][0]

    checkpoints = sorted(glob.glob(f"{INPUT}/**/knee_slot_fold*.pth", recursive=True))
    if not checkpoints:
        write_and_exit("no checkpoints found")
    log(f"checkpoints: {[os.path.basename(c) for c in checkpoints]}")

    # The checkpoint decides which encoder is correct, not the metadata.
    probe = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    want_hidden = int(probe["model"]["vit.embeddings.cls_token"].shape[-1])
    log(f"checkpoint expects encoder hidden size {want_hidden}")

    dinov2 = find_dinov2(want_hidden)
    if dinov2 is None:
        write_and_exit(f"no mounted DINOv2 with hidden_size {want_hidden}")

    models = []
    for path in checkpoints:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        # Build the model from the checkpoint's own recorded configuration, not
        # from this file's defaults. A checkpoint trained with a different pool
        # or a different prior setting loads with every shape matching and is
        # quietly a different model.
        net = SlotNet(dinov2, pool=blob.get("pool", "cls_mean_focal"))
        net.load_state_dict(blob["model"])
        net.eval().to(device)
        models.append(net)
        log(f"  {os.path.basename(path)}: fold {blob.get('fold')} "
            f"CV {blob.get('score', float('nan')):.4f} "
            f"labels {blob.get('labels')} pool {blob.get('pool')}")

    plane_of = dict(zip(test_series["SeriesInstanceUID"],
                        test_series["Anatomical_Plane"]))

    BATCH = 8
    preds = np.full((len(test), len(TARGETS)), 0.5, dtype=np.float32)
    # One prediction matrix per member, combined by rank at the end.
    member_preds = [
        np.full((len(test), len(TARGETS)), 0.5, dtype=np.float32)
        for _ in models
    ]
    scored = np.zeros(len(test), dtype=bool)
    failures = empty = 0
    sides = {"L": 0, "R": 0, "": 0}
    crop_ok = crop_total = 0

    for start in range(0, len(test), BATCH):
        chunk = test.iloc[start : start + BATCH]
        volumes, masks, rows = [], [], []

        for offset, (_, row) in enumerate(chunk.iterrows()):
            try:
                vol, msk, meta, _infos, (ok, total) = build_study(
                    f"{COMP}/test_series", row[ID], plane_of
                )
                if msk.sum() == 0:
                    empty += 1          # keeps its 0.5 default
                    continue
                volumes.append(vol)
                masks.append(msk)
                rows.append(start + offset)
                scored[start + offset] = True
                sides[meta["side"] if meta["side"] in ("L", "R") else ""] += 1
                crop_ok += ok
                crop_total += total
            except Exception:
                failures += 1           # keeps its 0.5 default

        if not volumes:
            continue

        x = torch.from_numpy(np.stack(volumes)).to(device)
        m = torch.from_numpy(np.stack(masks)).float().to(device)
        with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
            # Keep every member separate here; they are combined by RANK across
            # the whole test set once all batches are in, not averaged per
            # batch. A rank is a position within a column, so it cannot be
            # computed on 8 studies at a time.
            for member, net in enumerate(models):
                member_preds[member][rows] = (
                    torch.sigmoid(net(x, m).float()).cpu().numpy()
                )

        if start % 80 == 0:
            rate = (start + BATCH) / max(time.time() - T0, 1e-6)
            log(f"  {start}/{len(test)}  {rate:.2f} study/s  "
                f"failures {failures} empty {empty}")

    # --- combine members by RANK, per column ----------------------------
    # The metric is macro AUC over twelve independent per-label AUCs, and AUC
    # reads order only. Averaging probabilities lets a member that happens to
    # be more confident dominate one that merely ranks better; ranking first
    # makes members from different folds - or different configurations
    # entirely - directly comparable. This is what the public 0.899 notebook
    # does, and it is the OPPOSITE of the right rule for combining LABELS,
    # where BCE needs calibrated targets rather than queue positions.
    #
    # Only rows that were actually scored take part. A failed study keeps its
    # 0.5 default, and ranking that 0.5 in with real predictions would give it
    # a spurious mid-table position in every column.
    if scored.any():
        ranked = np.zeros((int(scored.sum()), len(TARGETS)), dtype=np.float32)
        for member in member_preds:
            frame = pd.DataFrame(member[scored], columns=list(TARGETS))
            ranked += frame.rank(pct=True).to_numpy(dtype=np.float32)
        preds[scored] = ranked / max(len(member_preds), 1)
    log(f"combined {len(member_preds)} member(s) by per-column rank mean "
        f"over {int(scored.sum())} scored studies")

    for i, target in enumerate(TARGETS):
        submission[target] = preds[:, i]
    submission.to_csv("submission.csv", index=False)

    log(f"WROTE submission.csv — {submission.shape}")
    log(f"  decode failures {failures} | no usable slot {empty}")
    log(f"  laterality: L={sides['L']} R={sides['R']} unknown={sides['']}")
    log(f"  crop applied {crop_ok}/{crop_total} "
        f"({100.0 * crop_ok / max(crop_total, 1):.0f}%)")

    # A submission that is still mostly 0.5 means inference silently did
    # nothing — the exact failure that produced a constant submission earlier
    # in this project. Say so loudly rather than letting the score explain it.
    default_rows = int((np.abs(preds - 0.5) < 1e-6).all(axis=1).sum())
    if default_rows:
        log(f"!! {default_rows}/{len(test)} rows are still the 0.5 default")
    if default_rows > 0.1 * len(test):
        log("!! more than 10% of rows are untouched — treat this submission "
            "as diagnostic, not as a result")

except SystemExit:
    raise
except Exception:
    traceback.print_exc()
    write_and_exit("inference raised — see traceback above")

print(pd.read_csv("submission.csv").head())
