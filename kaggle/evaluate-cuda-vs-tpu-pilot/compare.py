# Appended to the shared training input/validation definitions by build.py.
# docs/23: paired CUDA-trained vs TPU-trained comparison, folds 0/1, under the
# exact production five-view TTA - first standalone, then as a swap-in for
# the small member of the deployed (0.5 small + 0.5 base) 0.893 ensemble.
#
# Both checkpoints started from verified bit-identical initial weights
# (docs/23, kaggle/verify-init-cuda and kaggle/verify-init-tpu) and share the
# same recipe - labels, six-slice cache, batch, optimizer, augmentation,
# 24-epoch schedule, per-fold seed (SEED+fold). The TPU-trained checkpoint is
# the "ordinary" leg of the EMA pilot (kaggle/train-ema-pilot) - EMA never
# writes back into that model, so it is a valid, unmodified TPU control.
import gc
import json
from pathlib import Path

from src.model.tta import TTA_SHIFT, TTA_VIEWS, TTA_WEIGHTS, tta_probs

assert device.type == "cuda", "evaluation requires a GPU"
assert TTA_SHIFT <= AUG_SHIFT
assert abs(sum(TTA_WEIGHTS) - 1.0) < 1e-12 and len(TTA_VIEWS) == len(TTA_WEIGHTS)

FOLDS = (0, 1)          # the pilot's scope only - do not extrapolate to 2-4
GROUP_SIZE = GROUP

TPU_DATASET = "knee-ema-pilot-v1"              # TPU-trained "ordinary" leg
CUDA_DATASET = "knee-cuda-vs-tpu-pilot-v1"     # CUDA-trained, this pilot
base_dataset, base_hidden = "knee-slot-base-24ep-v1", 768
small_hidden = 384

checkpoints, encoders = {}, {}
for root, dirs, files in os.walk(INPUT):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    if "config.json" in files and "dinov2" in root.lower():
        cfg = json.loads(Path(root, "config.json").read_text())
        encoders[int(cfg.get("hidden_size", -1))] = root
    parts = Path(root).parts
    if base_dataset in parts:
        for name in files:
            if name.startswith("knee_slot_fold") and name.endswith(".pth") and "_snap_" not in name:
                fold = int(name.removeprefix("knee_slot_fold").removesuffix(".pth"))
                checkpoints["base", fold] = os.path.join(root, name)
    if TPU_DATASET in parts:
        for name in files:
            if name.startswith("knee_slot_fold") and name.endswith("_ordinary_final.pth"):
                fold = int(name.removeprefix("knee_slot_fold").removesuffix("_ordinary_final.pth"))
                checkpoints["tpu", fold] = os.path.join(root, name)
    if CUDA_DATASET in parts:
        for name in files:
            if name.startswith("knee_slot_fold") and name.endswith("_cuda_final.pth"):
                fold = int(name.removeprefix("knee_slot_fold").removesuffix("_cuda_final.pth"))
                checkpoints["cuda", fold] = os.path.join(root, name)

expected_keys = {(k, f) for k in ("tpu", "cuda", "base") for f in FOLDS}
missing = expected_keys - set(checkpoints)
assert not missing, f"missing checkpoints: {sorted(missing)}"
assert small_hidden in encoders and base_hidden in encoders, encoders
log(f"found all {len(checkpoints)} expected checkpoints (tpu/cuda/base x {len(FOLDS)} folds)")


def load_member(key, fold):
    hidden = base_hidden if key == "base" else small_hidden
    blob = torch.load(checkpoints[key, fold], map_location="cpu", weights_only=False)
    assert blob["fold"] == fold, (key, fold, blob["fold"])
    assert blob["head"] == "xattn"
    if key == "tpu":
        assert blob.get("device") == "xla", f"tpu fold {fold}: device {blob.get('device')}"
        assert blob.get("ema") is False, f"tpu fold {fold}: ema flag {blob.get('ema')}"
    if key == "cuda":
        assert blob.get("device") == "cuda", f"cuda fold {fold}: device {blob.get('device')}"
    if key in ("tpu", "cuda"):
        assert blob.get("epochs") == 24 and blob.get("seed") == fold, \
            f"{key} fold {fold}: epochs {blob.get('epochs')} seed {blob.get('seed')}"
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
MEMBER_KEYS = ("tpu", "cuda", "base")
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

    # Same OOM-safe pattern as evaluate-snap-tta: translate once per batch,
    # run every member against it, then drop the cache before the next batch.
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
                            s_per_study=seconds / n))
        log(f"fold {fold} {key}: {seconds:.1f}s ({seconds / n:.3f} s/study)")
        del model, blob
    del loaded
    gc.collect()
    torch.cuda.empty_cache()

# --- 1. standalone: TPU-trained vs CUDA-trained, small family alone ------
standalone_scores = {}
for key in ("tpu", "cuda"):
    for fold in FOLDS:
        standalone_scores[key, fold], _ = macro_auc(targets[fold], member_prob[key, fold])

# --- 2. ensemble: swap the small member in the deployed 0.5/0.5 blend ----
ens = {}
for fold in FOLDS:
    ens["tpu", fold] = 0.5 * member_prob["tpu", fold] + 0.5 * member_prob["base", fold]
    ens["cuda", fold] = 0.5 * member_prob["cuda", fold] + 0.5 * member_prob["base", fold]
ens_scores = {}
for key in ("tpu", "cuda"):
    for fold in FOLDS:
        ens_scores[key, fold], _ = macro_auc(targets[fold], ens[key, fold])

# --- paired scanner-group bootstrap, both comparisons ---------------------
rng = np.random.default_rng(0)
group_rows = {f: [np.flatnonzero(groups[f] == g) for g in np.unique(groups[f])] for f in FOLDS}
N_BOOT = 500
boot_standalone = {f: [] for f in FOLDS}
boot_ens = {f: [] for f in FOLDS}
for _ in range(N_BOOT):
    idx = {}
    for f in FOLDS:
        pick = rng.integers(0, len(group_rows[f]), len(group_rows[f]))
        idx[f] = np.concatenate([group_rows[f][i] for i in pick])
    for f in FOLDS:
        sa, _ = macro_auc(targets[f][idx[f]], member_prob["tpu", f][idx[f]])
        sb, _ = macro_auc(targets[f][idx[f]], member_prob["cuda", f][idx[f]])
        boot_standalone[f].append(sb - sa)
        ea, _ = macro_auc(targets[f][idx[f]], ens["tpu", f][idx[f]])
        eb, _ = macro_auc(targets[f][idx[f]], ens["cuda", f][idx[f]])
        boot_ens[f].append(eb - ea)


def ci(x):
    x = np.asarray(x)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def summarize(scores, boot):
    row = {}
    for f in FOLDS:
        row[f"fold{f}_tpu"] = scores["tpu", f]
        row[f"fold{f}_cuda"] = scores["cuda", f]
        row[f"fold{f}_delta"] = scores["cuda", f] - scores["tpu", f]
        row[f"fold{f}_delta_ci95"] = ci(boot[f])
    deltas = [scores["cuda", f] - scores["tpu", f] for f in FOLDS]
    mean_boot = np.mean([boot[f] for f in FOLDS], axis=0)
    row["mean_tpu"] = float(np.mean([scores["tpu", f] for f in FOLDS]))
    row["mean_cuda"] = float(np.mean([scores["cuda", f] for f in FOLDS]))
    row["mean_delta"] = float(np.mean(deltas))
    row["mean_delta_ci95"] = ci(mean_boot)
    row["mean_delta_boot_frac_positive"] = float((mean_boot > 0).mean())
    row["folds_nonnegative"] = int(sum(d >= 0 for d in deltas))
    return row


summary = dict(
    config=dict(folds=list(FOLDS), n_boot=N_BOOT,
                n_scanner_groups={str(f): int(len(group_rows[f])) for f in FOLDS},
                views=[list(v) for v in TTA_VIEWS], weights=list(TTA_WEIGHTS)),
    standalone=summarize(standalone_scores, boot_standalone),
    ensemble=summarize(ens_scores, boot_ens),
)
summary["runtime"] = runtime
summary["limitations"] = [
    "Two-fold pilot (docs/15/docs/20/docs/23 scope) - not the full five folds.",
    "Initial weights verified bit-identical across backends for this pilot only "
    "(docs/23); the historical deployed/TPU checkpoints' own initial state was "
    "never recorded and remains unverified.",
    "CUDA and TPU/XLA differ in default matmul precision and kernel "
    "implementations beyond the verified-identical starting point, so this is "
    "a deployment-backend experiment, not a fully isolated hardware comparison.",
    "Bootstrap resamples scanner-fingerprint groups within each fold, paired "
    "against the TPU control; fold 0 has few groups.",
]
Path("summary.json").write_text(json.dumps(summary, indent=2))
np.savez_compressed(
    "predictions.npz",
    **{f"ids_fold{f}": ids[f] for f in FOLDS},
    **{f"groups_fold{f}": groups[f] for f in FOLDS},
    **{f"target_fold{f}": targets[f] for f in FOLDS},
    **{f"{key}_tta_prob_fold{f}": member_prob[key, f] for key in MEMBER_KEYS for f in FOLDS},
)
log(json.dumps({"standalone": {k: v for k, v in summary["standalone"].items()},
                "ensemble": {k: v for k, v in summary["ensemble"].items()}}, indent=2))
