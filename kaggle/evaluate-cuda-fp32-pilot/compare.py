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

FOLDS = (0,)            # docs/25: the fp32 diagnostic trained fold 0 only
GROUP_SIZE = GROUP

TPU_DATASET = "knee-ema-pilot-v1"              # TPU-trained "ordinary" leg
CUDA_DATASET = "knee-cuda-vs-tpu-pilot-v1"     # CUDA-trained fp16 (docs/24)
CUDA32_DATASET = "knee-cuda-fp32-pilot-v1"     # CUDA-trained fp32 (docs/25)
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
    if CUDA32_DATASET in parts:
        for name in files:
            if name.startswith("knee_slot_fold") and name.endswith("_cuda32_final.pth"):
                fold = int(name.removeprefix("knee_slot_fold").removesuffix("_cuda32_final.pth"))
                checkpoints["cuda32", fold] = os.path.join(root, name)

expected_keys = {(k, f) for k in ("tpu", "cuda", "cuda32", "base") for f in FOLDS}
missing = expected_keys - set(checkpoints)
assert not missing, f"missing checkpoints: {sorted(missing)}"
assert small_hidden in encoders and base_hidden in encoders, encoders
log(f"found {len(checkpoints)} checkpoints: {sorted(checkpoints)}")


def load_member(key, fold):
    hidden = base_hidden if key == "base" else small_hidden
    blob = torch.load(checkpoints[key, fold], map_location="cpu", weights_only=False)
    assert blob["fold"] == fold, (key, fold, blob["fold"])
    assert blob["head"] == "xattn"
    if key == "tpu":
        assert blob.get("device") == "xla", f"tpu fold {fold}: device {blob.get('device')}"
        assert blob.get("ema") is False, f"tpu fold {fold}: ema flag {blob.get('ema')}"
    if key in ("cuda", "cuda32"):
        assert blob.get("device") == "cuda", f"{key} fold {fold}: device {blob.get('device')}"
    if key == "cuda32":
        assert next(iter(blob["model"].values())).dtype == torch.float32
    if key in ("tpu", "cuda", "cuda32"):
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
MEMBER_KEYS = ("tpu", "cuda", "cuda32", "base")
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

# --- comparisons: three paired contrasts, standalone and in the blend ----
PAIRS = {"cuda32_vs_tpu": ("tpu", "cuda32"), "cuda32_vs_cuda16": ("cuda", "cuda32"),
         "cuda16_vs_tpu": ("tpu", "cuda")}
SMALL_KEYS = ("tpu", "cuda", "cuda32")
standalone_scores, ens_scores, ens = {}, {}, {}
for key in SMALL_KEYS:
    for fold in FOLDS:
        standalone_scores[key, fold], _ = macro_auc(targets[fold], member_prob[key, fold])
        ens[key, fold] = 0.5 * member_prob[key, fold] + 0.5 * member_prob["base", fold]
        ens_scores[key, fold], _ = macro_auc(targets[fold], ens[key, fold])

rng = np.random.default_rng(0)
group_rows = {f: [np.flatnonzero(groups[f] == g) for g in np.unique(groups[f])] for f in FOLDS}
N_BOOT = 500
boot = {(p, kind): {f: [] for f in FOLDS} for p in PAIRS for kind in ("standalone", "ensemble")}
for _ in range(N_BOOT):
    for f in FOLDS:
        pick = rng.integers(0, len(group_rows[f]), len(group_rows[f]))
        idx = np.concatenate([group_rows[f][i] for i in pick])
        s = {k: macro_auc(targets[f][idx], member_prob[k, f][idx])[0] for k in SMALL_KEYS}
        e = {k: macro_auc(targets[f][idx], ens[k, f][idx])[0] for k in SMALL_KEYS}
        for p, (a, b) in PAIRS.items():
            boot[p, "standalone"][f].append(s[b] - s[a])
            boot[p, "ensemble"][f].append(e[b] - e[a])


def ci(x):
    x = np.asarray(x)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


summary = dict(
    config=dict(folds=list(FOLDS), n_boot=N_BOOT,
                n_scanner_groups={str(f): int(len(group_rows[f])) for f in FOLDS},
                views=[list(v) for v in TTA_VIEWS], weights=list(TTA_WEIGHTS)),
    standalone={k: {f"fold{f}": standalone_scores[k, f] for f in FOLDS} for k in SMALL_KEYS},
    ensemble={k: {f"fold{f}": ens_scores[k, f] for f in FOLDS} for k in SMALL_KEYS},
    comparisons={},
)
for p, (a, b) in PAIRS.items():
    for kind, scores in (("standalone", standalone_scores), ("ensemble", ens_scores)):
        summary["comparisons"][f"{p}_{kind}"] = {
            f"fold{f}": {"a": scores[a, f], "b": scores[b, f], "delta": scores[b, f] - scores[a, f],
                         "delta_ci95": ci(boot[p, kind][f]),
                         "boot_frac_positive": float((np.asarray(boot[p, kind][f]) > 0).mean())}
            for f in FOLDS}
summary["runtime"] = runtime
summary["limitations"] = [
    "Fold 0 only; one fp32 run. Diagnostic of the docs/24 gap, not a deployment candidate.",
    "fp32 applies to training only; evaluation uses the unchanged production TTA path (fp16 autocast inference).",
    "All three small checkpoints share verified-identical initial weights (docs/23) and the same recipe.",
]
Path("summary.json").write_text(json.dumps(summary, indent=2))
np.savez_compressed(
    "predictions.npz",
    **{f"ids_fold{f}": ids[f] for f in FOLDS},
    **{f"groups_fold{f}": groups[f] for f in FOLDS},
    **{f"target_fold{f}": targets[f] for f in FOLDS},
    **{f"{key}_tta_prob_fold{f}": member_prob[key, f] for key in MEMBER_KEYS for f in FOLDS},
)
log(json.dumps(summary["comparisons"], indent=2))
