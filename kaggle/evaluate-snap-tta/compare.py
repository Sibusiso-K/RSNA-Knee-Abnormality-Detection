# Appended to the shared training input/validation definitions by build.py.
# docs/20 job 1: do existing late-epoch snapshots of the small family beat
# the deployed small checkpoint, under the exact production five-view TTA,
# with NO retraining? Two predeclared candidates, frozen before running:
#   Candidate A = 0.5 * TTA(E24) + 0.5 * TTA(B)   (fixed final-epoch swap)
#   Candidate B = 0.25*TTA(E20) + 0.25*TTA(E24) + 0.5*TTA(B)  (epoch average)
# against Anchor = 0.5 * TTA(S) + 0.5 * TTA(B), the 0.893 submission's
# per-fold formula. Base family and the translation set are unchanged.
import gc
import json
from pathlib import Path

from src.model.tta import TTA_SHIFT, TTA_VIEWS, TTA_WEIGHTS, tta_probs

assert device.type == "cuda", "evaluation requires a GPU"
assert TTA_SHIFT <= AUG_SHIFT
assert abs(sum(TTA_WEIGHTS) - 1.0) < 1e-12 and len(TTA_VIEWS) == len(TTA_WEIGHTS)

FOLDS = range(5)
GROUP_SIZE = GROUP  # from the shared training script: slice-group size (3)

# Deployed-anchor OOF this evaluator must reproduce before any candidate
# result means anything (kaggle/evaluate-tta/results/summary.json).
EXPECTED_ANCHOR_FOLD = {0: 0.8677146033238342, 1: 0.8399920045007625, 2: 0.8628445487187576,
                        3: 0.8583865354878745, 4: 0.8710175712596113}
EXPECTED_ANCHOR_MEAN = 0.8599910526581681
REPRO_TOL = 3e-4
N_BOOT = 500

SNAP_DATASET = "knee-slot-6slice-24ep-snap-v1"
small_dataset, small_hidden = "knee-slot-6slice-24ep-v1", 384
base_dataset, base_hidden = "knee-slot-base-24ep-v1", 768

checkpoints, encoders = {}, {}
for root, dirs, files in os.walk(INPUT):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    if "config.json" in files and "dinov2" in root.lower():
        cfg = json.loads(Path(root, "config.json").read_text())
        encoders[int(cfg.get("hidden_size", -1))] = root
    parts = Path(root).parts
    if small_dataset in parts:
        for name in files:
            if name.startswith("knee_slot_fold") and name.endswith(".pth") and "_snap_" not in name:
                fold = int(name.removeprefix("knee_slot_fold").removesuffix(".pth"))
                checkpoints["S", fold] = os.path.join(root, name)
    if base_dataset in parts:
        for name in files:
            if name.startswith("knee_slot_fold") and name.endswith(".pth") and "_snap_" not in name:
                fold = int(name.removeprefix("knee_slot_fold").removesuffix(".pth"))
                checkpoints["B", fold] = os.path.join(root, name)
    if SNAP_DATASET in parts:
        for name in files:
            if not (name.startswith("knee_slot_fold") and name.endswith(".pth") and "_snap_e" in name):
                continue
            rest = name[len("knee_slot_fold"):-len(".pth")]
            fold_str, epoch_str = rest.split("_snap_e")
            fold, epoch = int(fold_str), int(epoch_str)
            if epoch in (20, 24):
                checkpoints[f"E{epoch}", fold] = os.path.join(root, name)

expected_keys = {(k, f) for k in ("S", "B", "E20", "E24") for f in FOLDS}
missing = expected_keys - set(checkpoints)
assert not missing, f"missing checkpoints: {sorted(missing)}"
assert small_hidden in encoders and base_hidden in encoders, encoders
log(f"found all {len(checkpoints)} expected checkpoints across S/B/E20/E24 x 5 folds")


def load_member(key, fold):
    hidden = base_hidden if key == "B" else small_hidden
    blob = torch.load(checkpoints[key, fold], map_location="cpu", weights_only=False)
    assert blob["fold"] == fold, (key, fold, blob["fold"])
    if key == "B":
        assert blob["epochs"] == 24 and blob["head"] == "xattn"
    else:
        assert blob["head"] == "xattn"
    model = SlotNet(encoders[hidden], pool=blob["pool"], head=blob["head"],
                    unfreeze_last=UNFREEZE_LAST)
    model.load_state_dict(blob["model"], strict=True)
    model.eval().to(device)
    return model, blob


def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z.astype(np.float64)))).astype(np.float32)


targets, groups, ids, cache_rows = {}, {}, {}, {}
member_prob = {}       # (key, fold) -> (n,12) TTA probability
runtime = []
MEMBER_KEYS = ("S", "B", "E20", "E24")
for fold in FOLDS:
    sel = np.flatnonzero(frame["fold"].values == fold)
    rows_idx = frame.iloc[sel]["row"].values
    targets[fold] = Y[sel]
    groups[fold] = frame.iloc[sel]["fingerprint"].fillna("unknown").astype(str).values
    ids[fold] = frame.iloc[sel][ID].to_numpy(dtype=str)
    cache_rows[fold] = rows_idx

    loaded = {key: load_member(key, fold) for key in MEMBER_KEYS}
    n = len(rows_idx)
    out = {key: np.empty((n, len(TARGETS)), dtype=np.float32) for key in MEMBER_KEYS}
    member_seconds = {key: 0.0 for key in MEMBER_KEYS}

    # One pass over this fold's batches: translate each view ONCE per batch
    # (view_cache), run every member against it, then DROP the cache - it
    # must not accumulate past the batch it was built for, or cached float32
    # views from every earlier batch stay resident and exhaust the GPU (the
    # bug behind the first run's CUDA OOM: a cache keyed by batch start that
    # was never cleared).
    with torch.no_grad():
        for start in range(0, n, BATCH):
            batch_sel = rows_idx[start:start + BATCH]
            m = torch.from_numpy(mask[batch_sel]).float().to(device)
            xk = torch.from_numpy(cache[batch_sel]).to(device)
            view_cache = {}
            for key in MEMBER_KEYS:
                model, _blob = loaded[key]
                net = lambda x, mm, _model=model: _model(x, mm, TRAIN_SIZE)  # noqa: E731
                torch.cuda.synchronize(); t0 = time.time()
                with autocast():
                    probs = tta_probs(net, xk, m, GROUP_SIZE, view_cache=view_cache)
                torch.cuda.synchronize()
                member_seconds[key] += time.time() - t0
                out[key][start:start + BATCH] = probs.float().cpu().numpy()
            del view_cache, xk, m

    for key in MEMBER_KEYS:
        model, blob = loaded[key]
        member_prob[key, fold] = out[key]
        seconds = member_seconds[key]
        runtime.append(dict(key=key, fold=fold, n_studies=n, seconds=seconds,
                            s_per_study=seconds / n,
                            tpu_epoch=blob.get("epoch"), tpu_score=float(blob.get("score", -1))))
        log(f"fold {fold} {key}: {seconds:.1f}s ({seconds / n:.3f} s/study), "
            f"tpu_epoch={blob.get('epoch')}")
        del model, blob
    del loaded
    gc.collect()
    torch.cuda.empty_cache()

anchor, cand_a, cand_b = {}, {}, {}
for fold in FOLDS:
    anchor[fold] = 0.5 * member_prob["S", fold] + 0.5 * member_prob["B", fold]
    cand_a[fold] = 0.5 * member_prob["E24", fold] + 0.5 * member_prob["B", fold]
    cand_b[fold] = (0.25 * member_prob["E20", fold] + 0.25 * member_prob["E24", fold]
                    + 0.5 * member_prob["B", fold])

anchor_scores = {}
for fold in FOLDS:
    anchor_scores[fold], _ = macro_auc(targets[fold], anchor[fold])
    assert abs(anchor_scores[fold] - EXPECTED_ANCHOR_FOLD[fold]) < REPRO_TOL, \
        (fold, anchor_scores[fold], EXPECTED_ANCHOR_FOLD[fold])
mean_anchor = float(np.mean(list(anchor_scores.values())))
log(f"anchor mean {mean_anchor:.10f} vs expected {EXPECTED_ANCHOR_MEAN:.10f}")
assert abs(mean_anchor - EXPECTED_ANCHOR_MEAN) < REPRO_TOL

ARMS = {"anchor": anchor, "cand_a": cand_a, "cand_b": cand_b}
scores, per_target = {}, {}
for name, arm in ARMS.items():
    for fold in FOLDS:
        scores[name, fold], per_target[name, fold] = macro_auc(targets[fold], arm[fold])

rng = np.random.default_rng(0)
group_rows = {f: [np.flatnonzero(groups[f] == g) for g in np.unique(groups[f])] for f in FOLDS}
PAIRS = {"cand_a": ("anchor", "cand_a"), "cand_b": ("anchor", "cand_b")}
boot = {p: {f: [] for f in FOLDS} for p in PAIRS}
for _ in range(N_BOOT):
    idx = {}
    for f in FOLDS:
        pick = rng.integers(0, len(group_rows[f]), len(group_rows[f]))
        idx[f] = np.concatenate([group_rows[f][i] for i in pick])
    for p, (a, b) in PAIRS.items():
        for f in FOLDS:
            sa, _ = macro_auc(targets[f][idx[f]], ARMS[a][f][idx[f]])
            sb, _ = macro_auc(targets[f][idx[f]], ARMS[b][f][idx[f]])
            boot[p][f].append(sb - sa)


def ci(x):
    x = np.asarray(x)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


summary = dict(
    config=dict(views=[list(v) for v in TTA_VIEWS], weights=list(TTA_WEIGHTS),
                n_boot=N_BOOT, n_scanner_groups={str(f): int(len(group_rows[f])) for f in FOLDS}),
    anchor_reproduced=dict(mean=mean_anchor, expected_mean=EXPECTED_ANCHOR_MEAN),
    comparison={},
)
for p, (a, b) in PAIRS.items():
    row = {}
    for f in FOLDS:
        row[f"fold{f}_anchor"] = scores[a, f]
        row[f"fold{f}_candidate"] = scores[b, f]
        row[f"fold{f}_delta"] = scores[b, f] - scores[a, f]
        row[f"fold{f}_delta_ci95"] = ci(boot[p][f])
    deltas = [scores[b, f] - scores[a, f] for f in FOLDS]
    mean_boot = np.mean([boot[p][f] for f in FOLDS], axis=0)
    row["mean_anchor"] = float(np.mean([scores[a, f] for f in FOLDS]))
    row["mean_candidate"] = float(np.mean([scores[b, f] for f in FOLDS]))
    row["mean_delta"] = float(np.mean(deltas))
    row["mean_delta_ci95"] = ci(mean_boot)
    row["mean_delta_boot_frac_positive"] = float((mean_boot > 0).mean())
    row["folds_nonnegative"] = int(sum(d >= 0 for d in deltas))
    row["per_target_delta"] = {
        t: {**{f"fold{f}": per_target[b, f][t] - per_target[a, f][t] for f in FOLDS},
            "mean": float(np.mean([per_target[b, f][t] - per_target[a, f][t] for f in FOLDS]))}
        for t in per_target[a, 0]
    }
    summary["comparison"][p] = row

by_key = {}
for key in ("S", "B", "E20", "E24"):
    rs = [r for r in runtime if r["key"] == key]
    by_key[key] = float(np.mean([r["s_per_study"] for r in rs]))
N_HIDDEN_TEST = 1300
summary["runtime"] = dict(
    by_key_s_per_study=by_key,
    # Candidate B adds E20 as a fifth small-family member at deployment
    # (S, B, E24 already run today; E20 is the only NEW forward cost).
    projected_incremental_forward_seconds_candidate_b=5 * N_HIDDEN_TEST * by_key["E20"],
    note="Measured on a Kaggle T4 including mmap cache reads; excludes DICOM decode (unchanged, shared).",
)
summary["limitations"] = [
    "Both snapshot epochs were already inspected in experiment 1 (docs/16); treat as exploratory, not untouched validation.",
    "Candidate A/B were frozen before this run; no per-fold epoch or weight search was performed.",
    "Bootstrap resamples scanner-fingerprint groups within each fold, paired against the anchor; fold 0 has few groups.",
]
Path("summary.json").write_text(json.dumps(summary, indent=2))
Path("runtime.json").write_text(json.dumps(runtime, indent=2))
np.savez_compressed(
    "predictions.npz",
    **{f"ids_fold{f}": ids[f] for f in FOLDS},
    **{f"groups_fold{f}": groups[f] for f in FOLDS},
    **{f"cache_row_fold{f}": cache_rows[f] for f in FOLDS},
    **{f"target_fold{f}": targets[f] for f in FOLDS},
    **{f"{key}_tta_prob_fold{f}": member_prob[key, f] for key in ("S", "B", "E20", "E24") for f in FOLDS},
    views=np.asarray(TTA_VIEWS, dtype=np.float32), weights=np.asarray(TTA_WEIGHTS, dtype=np.float32),
)
log(json.dumps({p: {k: v for k, v in r.items() if k != "per_target_delta"}
                for p, r in summary["comparison"].items()}, indent=2))
log(json.dumps(summary["runtime"], indent=2))
