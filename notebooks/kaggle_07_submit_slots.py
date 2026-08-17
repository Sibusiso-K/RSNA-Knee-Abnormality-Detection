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
SKIP_DIRS = {"train_series", "test_series", ".git", "__pycache__", "pkg"}


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
from src.data.slots import GROUP, IMG, N_SLOT        # noqa: E402

N_GROUPS = max(1, N_SLICE // GROUP)
if N_SLICE % GROUP:
    raise SystemExit(f"test cache {N_SLICE} slices/slot is not a multiple of {GROUP}")
from src.labels import TARGETS                        # noqa: E402

# Decoding depends on packages that differ between Kaggle's CPU/GPU image and
# its TPU image, and submissions run with internet OFF so nothing can be
# installed at runtime. Report what is actually present before doing any work:
# a missing JPEG-2000 backend turns every study into a silent decode failure
# and a constant 0.5 submission.
for _mod in ("pydicom", "cv2", "gdcm", "pylibjpeg", "openjpeg", "PIL"):
    try:
        _m = __import__(_mod)
        log(f"  {_mod:10s} {getattr(_m, '__version__', 'present')}")
    except Exception as _exc:                              # noqa: BLE001
        log(f"  {_mod:10s} MISSING ({type(_exc).__name__})")
try:
    import pydicom

    log(f"  pydicom handlers: "
        f"{[h.__name__.split('.')[-1] for h in pydicom.config.pixel_data_handlers if h.is_available()]}")
except Exception as _exc:                                  # noqa: BLE001
    log(f"  pydicom handler probe failed: {_exc}")

test = pd.read_csv(f"{COMP}/test.csv")
test_series = pd.read_csv(f"{COMP}/test_series.csv")
submission = pd.DataFrame({ID: test[ID]})
for target in TARGETS:
    submission[target] = 0.5

log(f"test: {len(test)} studies, {len(test_series)} series")
log(f"test cache: {N_SLICE} slices/slot = {N_GROUPS} group(s) of {GROUP}")


def write_and_exit(reason):
    """Always leave a valid submission.csv behind, whatever went wrong."""
    submission.to_csv("submission.csv", index=False)
    log(f"WROTE fallback submission.csv ({reason})")
    print(pd.read_csv("submission.csv").head())
    raise SystemExit(0)


try:
    # CUDA, XLA/TPU or CPU. XLA matters here for a practical reason: five
    # members on CPU already ran close to the 9 h submission cap, because the
    # cost is (studies x slots x MEMBERS) encoder passes and members are the
    # axis we want to grow. On TPU the same work is minutes, so the ensemble
    # size stops being limited by the scoring budget.
    XLA = False
    xm = None
    try:
        import torch_xla.core.xla_model as _xm

        xm = _xm
        device = xm.xla_device()
        XLA = True
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}  XLA={XLA}")

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

    # Both naming conventions: ours (knee_slot_fold*.pth) and the public
    # members (m_*.pt). A mixed ensemble is the point, so the glob cannot be
    # tied to our own filenames.
    checkpoints = sorted(
        glob.glob(f"{INPUT}/**/knee_slot_fold*.pth", recursive=True)
        + glob.glob(f"{INPUT}/**/m_*.pt", recursive=True)
        + glob.glob(f"{INPUT}/**/champ_fold*.pt", recursive=True)
    )
    if not checkpoints:
        write_and_exit("no checkpoints found")
    log(f"checkpoints: {[os.path.basename(c) for c in checkpoints]}")

    # Encoder is resolved PER CHECKPOINT, not once for the batch.
    #
    # The whole point of a mixed ensemble is that members differ, and the
    # cheapest useful difference is encoder width - our sixth member is
    # DINOv2-base (768) alongside five smalls (384). Resolving the encoder from
    # checkpoints[0] and reusing it would build every member on that width and
    # die on the first mismatch. Each checkpoint names its own.
    def normalise(blob):
        """Any checkpoint -> (state_dict, pool, prior, head).

        Public members published by other competitors use the same SlotHead
        parameter names as ours - head.slot_emb / query / proj / out - and the
        same six slots in the same order, but call the encoder `backbone`
        where we call it `vit`. Their manifest also records pool="cls_mean"
        and prior=False, so their checkpoints carry no head.slot_prior; ours
        registers that buffer unconditionally and it is all zeros when the
        prior is off, which is exactly what prior=False means.

        So the mapping is a rename plus one legitimately-absent buffer. That
        is checked rather than assumed below: any OTHER missing key, or any
        unexpected one, means the architectures genuinely differ and the run
        must stop rather than load a partly-initialised model and score it.
        """
        import re as _re

        sd = blob["model"] if "model" in blob else blob
        # Different publishers name the encoder differently: ours `vit`,
        # pilkwang `backbone`, stevenleehans `encoder.module` (a DataParallel
        # wrapper that survived the save). The head parameters are identical in
        # all three, so a prefix rewrite is the whole of the mapping.
        sd = {_re.sub(r"^(backbone\.|encoder\.module\.|encoder\.)", "vit.", k): v
              for k, v in sd.items()}
        cfg = blob.get("config") or {}
        pool = cfg.get("pool") or blob.get("pool")
        if pool is None:
            # No recorded config (champ members). The head's input width says
            # it: proj.0 takes dim*parts, so 2x encoder width is cls_mean and
            # 3x is cls_mean_focal. Guessing wrong is a shape error rather than
            # a silent one, but inferring is better than guessing.
            # proj.0 is the LayerNorm (1-D weight); proj.1 is the Linear that
            # actually takes dim*parts. Reading the wrong one is an IndexError,
            # which is how this was found.
            w = sd["head.proj.1.weight"].shape[1]
            enc = next(v for k, v in sd.items()
                       if k.endswith("embeddings.cls_token")).shape[-1]
            pool = {2: "cls_mean", 3: "cls_mean_focal"}.get(w // enc, "cls_mean")
        prior = cfg.get("prior", blob.get("prior",
                                          any("slot_prior" in k for k in sd)))
        head = blob.get("head")
        if head is None:
            head = "xattn" if any(k.startswith("head.cross_attn") for k in sd) else "slot"
        return sd, pool, bool(prior), head

    models = []
    for path in checkpoints:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        _probe = blob["model"] if "model" in blob else blob
        hidden = int(next(v for k, v in _probe.items()
                          if k.endswith("embeddings.cls_token")).shape[-1])
        dinov2 = find_dinov2(hidden)
        if dinov2 is None:
            write_and_exit(f"no mounted DINOv2 with hidden_size {hidden}")
        # Build the model from the checkpoint's own recorded configuration, not
        # from this file's defaults. A checkpoint trained with a different pool
        # or a different prior setting loads with every shape matching and is
        # quietly a different model.
        # Which HEAD the checkpoint was trained with. Older checkpoints do
        # not record it, so it is inferred from the weights themselves: the
        # cross-attention head has parameters the slot head does not. Guessing
        # wrong is not subtle - load_state_dict fails outright - but defaulting
        # to "slot" and failing is exactly what happened here, so the model is
        # built from what the file contains rather than from a default.
        sd, pool, prior, head = normalise(blob)
        net = SlotNet(dinov2, pool=pool, prior=prior, head=head)
        result = net.load_state_dict(sd, strict=False)
        # slot_prior is a zeros buffer when prior=False, so its absence is
        # expected for members trained without the anatomy tilt. Nothing else is.
        allowed = {"head.slot_prior"} if not prior else set()
        bad_missing = set(result.missing_keys) - allowed
        if bad_missing or result.unexpected_keys:
            write_and_exit(
                f"{os.path.basename(path)} does not fit SlotNet: "
                f"missing {sorted(bad_missing)[:4]} unexpected "
                f"{sorted(result.unexpected_keys)[:4]}"
            )
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
            except Exception as exc:    # noqa: BLE001
                # The FIRST failure is printed in full, with a traceback.
                # Counting failures silently is how a run reports "decode
                # failures 3" and leaves you guessing whether the cause is a
                # missing decoder, a missing package or a bad path. One
                # traceback costs nothing and names it.
                if failures == 0:
                    log(f"first decode failure on {row[ID]}: "
                        f"{type(exc).__name__}: {exc}")
                    traceback.print_exc()
                failures += 1           # keeps its 0.5 default

        if not volumes:
            continue

        x = torch.from_numpy(np.stack(volumes)).to(device)
        m = torch.from_numpy(np.stack(masks)).float().to(device)
        autocast = (torch.autocast("xla", dtype=torch.bfloat16) if XLA
                    else torch.autocast("cuda", enabled=device.type == "cuda"))
        with torch.no_grad(), autocast:
            # Keep every member separate here; they are combined by RANK across
            # the whole test set once all batches are in, not averaged per
            # batch. A rank is a position within a column, so it cannot be
            # computed on 8 studies at a time.
            for member, net in enumerate(models):
                # The encoder takes a 2.5D TRIPLET. A test cache holding more
                # slices per slot holds several groups of three, and the models
                # were trained by drawing one group per step - so inference
                # averages logits over every group, exactly as training sampled
                # them. Feeding all slices as channels would not even have the
                # right shape; feeding a differently-sampled triplet would have
                # the right shape and be quietly wrong.
                acc = None
                for g in range(N_GROUPS):
                    xg = x[:, :, g * GROUP:(g + 1) * GROUP]
                    lg = net(xg, m).float()
                    acc = lg if acc is None else acc + lg
                probs = torch.sigmoid(acc / N_GROUPS)
                if XLA:
                    # Cut the graph per member. Without it XLA traces every
                    # member of every batch into one graph and compiles the lot
                    # at the first .cpu() - which on the training side turned
                    # 0.12 s/step into 12 s/step and then ran out of HBM.
                    xm.mark_step()
                member_preds[member][rows] = probs.cpu().numpy()

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
