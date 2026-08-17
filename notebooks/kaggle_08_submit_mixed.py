"""KAGGLE SUBMISSION NOTEBOOK — mixed-grid ensemble.

Settings: Accelerator **GPU** | Internet OFF | Output submission.csv

Why this exists next to `kaggle_07_submit_slots.py`
---------------------------------------------------
`kaggle_07` builds ONE test cache and feeds every member from it. That is
correct only while every member was trained on the same sampling grid, and it
stops being true the moment the ensemble mixes our six-slice models with the
twenty public twelve-slice ones — which is exactly the ensemble worth running,
because those two groups were trained on different labels, different seeds and
different schedules, and rank-mean pays for disagreement.

The grids do not nest. `band_indices` is a linspace across the central band, so
six points sit at fractions k/5 and twelve at k/11; they share only the two
endpoints. Handing a six-slice model triplets carved out of a twelve-slice
cache gives tensors of exactly the right shape holding slices 5/11 as far
apart as anything it ever saw. It scores badly and looks like a bad model.

So the cache is built per grid, from ONE pass over each study's headers
(`build_study_multi`), and each member is run on the grid it was trained on.

The compatibility gate
----------------------
Slice count is not the only thing that has to match. A member also declares its
image size and slice band, and a member whose fingerprint disagrees is SKIPPED
with a loud line rather than silently fed the wrong pixels. This is not
hypothetical: the `champ` members published alongside the twelve-slice ones are
224 px over band 0.35-0.65 with nine slices, and an earlier submission ran them
against a 336 px twelve-slice cache. Nothing raised — DINOv2 interpolates its
position embeddings and happily consumes the wrong scale — and five of that
run's twenty-five members were reading anatomy they were never trained on.

Degraded-mode behaviour is deliberate, as in `kaggle_07`: a study that fails to
decode keeps 0.5 rather than raising, and every fallback is counted and printed.
"""

import os
import sys
import glob
import time
import shutil
import traceback

from collections import Counter

import numpy as np
import pandas as pd
import torch

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


PKG = "/kaggle/working/pkg"
INPUT = "/kaggle/input"
ID = "StudyInstanceUID"

SKIP_DIRS = {"train_series", "test_series", ".git", "__pycache__", "pkg"}

#: Slices-per-slot for members whose checkpoint does not record it, keyed by the
#: mounted dataset directory. Our first six-slice family predates the field.
#: A member that matches nothing here AND records nothing is refused rather
#: than defaulted: the failure mode of a wrong guess is a plausible score from
#: the wrong anatomy, which is the hardest kind of bug to see.
SLICES_BY_DIR = {
    "knee-slot-6slice-v1": 6,
    "knee-slot-bc-v1": 6,
    "rsna-knee-weights": 12,
}

#: Relative weight per grid in the rank mean. A plain average is only right
#: when the members are interchangeable, and these are not.
#:
#: This constant was first set 12:1 over 6, on the published reputation of the
#: twelve-slice members — a holdout of 0.8381 and a widely-quoted ~0.899 on the
#: board. Both groups have now been run through this pipeline and scored on the
#: SAME ruler, and the reputation did not survive it:
#:
#:     20 public twelve-slice members, rank-mean   ->  LB 0.839
#:     25 (those plus 5 mismatched champ members)  ->  LB 0.844
#:      5 of ours, six-slice                       ->  LB 0.864
#:
#: So the weaker group is theirs, not ours, and the original weighting would
#: have handed 73% of the blend to the members that measure lower. Inverted.
#:
#: 2:1 towards ours puts our fifteen at ~60%: they decide, while twenty
#: genuinely independent members — different labels, different seeds, different
#: schedules — keep a real share. That share is not charity. Adding five
#: members that were fed the WRONG anatomy still moved 0.839 to 0.844, which
#: says the blend is paying for disagreement rather than for accuracy.
#:
#: The exact ratio is still a judgement. It cannot be validated offline: the
#: public members were trained on studies we hold out, so any CV computed for
#: them here is contaminated. Recorded as an assumption, not buried.
GRID_WEIGHT = {6: 2.0, 12: 1.0}


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

import src.data.cache as _cache                        # noqa: E402
from src.data.cache import build_study, build_study_multi   # noqa: E402
from src.data.slots import GROUP, IMG, N_SLOT, SLICE_BAND   # noqa: E402
from src.model.members import member_fingerprint, refuse_reason  # noqa: E402
from src.labels import TARGETS                          # noqa: E402

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


def write_and_exit(reason):
    """Always leave a valid submission.csv behind, whatever went wrong."""
    submission.to_csv("submission.csv", index=False)
    log(f"WROTE fallback submission.csv ({reason})")
    print(pd.read_csv("submission.csv").head())
    raise SystemExit(0)


try:
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
        if hidden is not None:
            for root, size in candidates:
                if size == hidden:
                    return root
            log(f"!! no mounted DINOv2 has hidden_size {hidden}")
            return None
        return candidates[0][0]

    # Pruned walk, not glob(recursive=True).
    #
    # `**` descends into everything under /kaggle/input, and that includes
    # test_series/ — thousands of nested study directories holding the DICOMs.
    # Finding 35 checkpoints this way measured 613 s on the three-study public
    # run, where test_series is nearly empty; on the private re-run it holds
    # ~1,300 studies and the same walk is the cost that took ~1,100 s per call
    # on the training side. That comes straight off the 9 h cap for no reason:
    # no checkpoint is ever inside it.
    PATTERNS = ("knee_slot_fold*.pth", "m_*.pt", "champ_fold*.pt")
    import fnmatch

    checkpoints = []
    for root, dirs, files in os.walk(INPUT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if any(fnmatch.fnmatch(name, p) for p in PATTERNS):
                checkpoints.append(os.path.join(root, name))
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        write_and_exit("no checkpoints found")
    log(f"checkpoints found: {len(checkpoints)}")

    def normalise(blob):
        """Any checkpoint -> (state_dict, pool, prior, head). See kaggle_07."""
        import re as _re

        sd = blob["model"] if "model" in blob else blob
        sd = {_re.sub(r"^(backbone\.|encoder\.module\.|encoder\.)", "vit.", k): v
              for k, v in sd.items()}
        cfg = blob.get("config") or {}
        pool = cfg.get("pool") or blob.get("pool")
        if pool is None:
            # proj.0 is the LayerNorm (1-D weight); proj.1 is the Linear that
            # takes dim*parts. 2x encoder width is cls_mean, 3x cls_mean_focal.
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

    WANT_BAND = tuple(round(float(b), 3) for b in SLICE_BAND)

    models, skipped = [], []
    for path in checkpoints:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        name = os.path.basename(path)
        slices, img, band = member_fingerprint(path, blob, SLICES_BY_DIR)

        # Refuse rather than guess. Feeding a member the wrong grid, size or
        # band costs a fraction of the ensemble and shows up as noise, not as
        # an error - so the gate has to be here, before the pixels are built.
        why = refuse_reason(slices, img, band, IMG, WANT_BAND, GROUP)
        if why:
            skipped.append((name, why))
            continue

        _probe = blob["model"] if "model" in blob else blob
        hidden = int(next(v for k, v in _probe.items()
                          if k.endswith("embeddings.cls_token")).shape[-1])
        dinov2 = find_dinov2(hidden)
        if dinov2 is None:
            write_and_exit(f"no mounted DINOv2 with hidden_size {hidden}")

        sd, pool, prior, head = normalise(blob)
        net = SlotNet(dinov2, pool=pool, prior=prior, head=head)
        result = net.load_state_dict(sd, strict=False)
        allowed = {"head.slot_prior"} if not prior else set()
        bad_missing = set(result.missing_keys) - allowed
        if bad_missing or result.unexpected_keys:
            write_and_exit(
                f"{name} does not fit SlotNet: missing {sorted(bad_missing)[:4]} "
                f"unexpected {sorted(result.unexpected_keys)[:4]}"
            )
        net.eval().to(device)
        models.append((net, int(slices), name))

    for name, why in skipped:
        log(f"  SKIPPED {name}: {why}")
    if not models:
        write_and_exit("every checkpoint was refused by the fingerprint gate")

    TAKES = tuple(sorted({s for _net, s, _n in models}))


    log(f"members: {len(models)} kept, {len(skipped)} skipped | "
        f"grids {dict(Counter(s for _n, s, _m in models))}")
    log(f"test cache grids: {TAKES} slices/slot, groups of {GROUP}")

    plane_of = dict(zip(test_series["SeriesInstanceUID"],
                        test_series["Anatomical_Plane"]))

    # --- prove the shared-decode path reproduces the single-grid one --------
    #
    # `build_study_multi` reads each header once and decodes the UNION of the
    # wanted indices, which is a different code path from the `build_study`
    # that produced the TRAINING caches. If it diverges by so much as the
    # intensity normalisation, every member is scored on pixels it never saw
    # and the run still finishes and still writes a plausible file.
    #
    # So it is checked, on real test studies, before any of them are used:
    # both paths, byte for byte, on the first few studies with usable slots.
    # Three studies cost under a minute against a 9 h cap.
    checked = 0
    for _, row in test.iterrows():
        if checked >= 3:
            break
        try:
            multi, m_mask, _r, _i, _c = build_study_multi(
                f"{COMP}/test_series", row[ID], plane_of, TAKES
            )
            if m_mask.sum() == 0:
                continue
            for k in TAKES:
                _cache.N_SLICE = k
                single, s_mask, _r2, _i2, _c2 = build_study(
                    f"{COMP}/test_series", row[ID], plane_of
                )
                if not np.array_equal(multi[k], single) or not np.array_equal(m_mask, s_mask):
                    write_and_exit(
                        f"shared-decode check FAILED at {k} slices/slot on "
                        f"{row[ID]} - the union decode does not reproduce "
                        f"build_study, so members would score on wrong pixels"
                    )
            checked += 1
        except Exception as exc:                            # noqa: BLE001
            log(f"  self-check skipped {row[ID]}: {type(exc).__name__}: {exc}")
    if checked == 0:
        write_and_exit("could not verify the shared-decode path on any study")
    log(f"shared-decode check passed on {checked} studies for grids {TAKES}")

    BATCH = 8
    preds = np.full((len(test), len(TARGETS)), 0.5, dtype=np.float32)
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
        volumes = {k: [] for k in TAKES}
        masks, rows = [], []

        for offset, (_, row) in enumerate(chunk.iterrows()):
            try:
                vols, msk, meta, _infos, (ok, total) = build_study_multi(
                    f"{COMP}/test_series", row[ID], plane_of, TAKES
                )
                if msk.sum() == 0:
                    empty += 1          # keeps its 0.5 default
                    continue
                for k in TAKES:
                    volumes[k].append(vols[k])
                masks.append(msk)
                rows.append(start + offset)
                scored[start + offset] = True
                sides[meta["side"] if meta["side"] in ("L", "R") else ""] += 1
                crop_ok += ok
                crop_total += total
            except Exception as exc:    # noqa: BLE001
                if failures == 0:
                    log(f"first decode failure on {row[ID]}: "
                        f"{type(exc).__name__}: {exc}")
                    traceback.print_exc()
                failures += 1           # keeps its 0.5 default

        if not rows:
            continue

        x = {k: torch.from_numpy(np.stack(v)).to(device) for k, v in volumes.items()}
        m = torch.from_numpy(np.stack(masks)).float().to(device)
        autocast = (torch.autocast("xla", dtype=torch.bfloat16) if XLA
                    else torch.autocast("cuda", enabled=device.type == "cuda"))
        with torch.no_grad(), autocast:
            for member, (net, slices, _name) in enumerate(models):
                # Each member reads ITS OWN grid and averages logits over the
                # groups of three within it, exactly as training sampled one
                # group per step. This is the line the whole notebook exists
                # for: `xk` is chosen by the member, not by the cache.
                xk = x[slices]
                n_groups = slices // GROUP
                acc = None
                for g in range(n_groups):
                    xg = xk[:, :, g * GROUP:(g + 1) * GROUP]
                    lg = net(xg, m).float()
                    acc = lg if acc is None else acc + lg
                probs = torch.sigmoid(acc / n_groups)
                if XLA:
                    xm.mark_step()
                member_preds[member][rows] = probs.cpu().numpy()

        if start % 80 == 0:
            rate = (start + BATCH) / max(time.time() - T0, 1e-6)
            log(f"  {start}/{len(test)}  {rate:.2f} study/s  "
                f"failures {failures} empty {empty}")

    # --- combine members by RANK, per column ----------------------------
    # AUC reads order only, so members are compared by position within a
    # column rather than by confidence. Only rows that were actually scored
    # take part; ranking a defaulted 0.5 in with real predictions would give
    # it a spurious mid-table position in every column.
    # Members are NOT equal, so the mean is weighted (see GRID_WEIGHT).
    if scored.any():
        ranked = np.zeros((int(scored.sum()), len(TARGETS)), dtype=np.float32)
        total_w = 0.0
        for member, (_net, slices, _name) in zip(member_preds, models):
            w = float(GRID_WEIGHT.get(slices, 1.0))
            frame = pd.DataFrame(member[scored], columns=list(TARGETS))
            ranked += w * frame.rank(pct=True).to_numpy(dtype=np.float32)
            total_w += w
        preds[scored] = ranked / max(total_w, 1e-6)
    log(f"combined {len(member_preds)} member(s) by per-column rank mean "
        f"over {int(scored.sum())} scored studies")
    for k in TAKES:
        n = sum(1 for _n, s, _m in models if s == k)
        share = n * GRID_WEIGHT.get(k, 1.0) / max(
            sum(c * GRID_WEIGHT.get(g, 1.0)
                for g, c in Counter(s for _n, s, _m in models).items()), 1e-6)
        log(f"  {k:2d}-slice group: {n} members, weight "
            f"{GRID_WEIGHT.get(k, 1.0)} -> {100 * share:.0f}% of the blend")

    for i, target in enumerate(TARGETS):
        submission[target] = preds[:, i]

    log(f"  decode failures {failures} | no usable slot {empty}")
    log(f"  sides L={sides['L']} R={sides['R']} unknown={sides['']}")
    log(f"  physical crop applied on {crop_ok}/{crop_total} slots")

    # The constant-submission tripwire. A run that decodes nothing still
    # writes a valid file of 0.5s and still "succeeds"; it scores ~0.5.
    spread = float(np.nanstd(preds, axis=0).mean())
    if spread < 1e-6:
        log("!! every prediction is still the 0.5 default - nothing was scored")

except SystemExit:
    raise
except Exception:                                          # noqa: BLE001
    traceback.print_exc()
    write_and_exit("unhandled exception")

submission.to_csv("submission.csv", index=False)
log(f"wrote submission.csv {submission.shape}")
print(submission.head())
