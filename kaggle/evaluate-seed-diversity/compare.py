# Appended to the shared training input/validation definitions by build.py.
# Experiment 3 (docs/15): incremental ensemble value of a second 24-epoch
# small seed, folds 0 and 1 only. Exploratory - these folds were also used
# to train and pick every checkpoint involved.
import gc
import json
from pathlib import Path

from scipy.stats import spearmanr

assert device.type == "cuda", "evaluation requires a GPU"

FOLDS = (0, 1)
FAMILIES = {
    # family: (dataset directory name, encoder hidden size, expected seed or None)
    "small": ("knee-slot-6slice-24ep-v1", 384, None),   # pre-dates SEED recording
    "base": ("knee-slot-base-24ep-v1", 768, None),
    "seed1": ("knee-slot-seed1-pilot-v1", 384, 1),
}
# From kaggle/evaluate-fresh24 v5 (summary.json): the deployed small+base
# 50/50 probability blend on CUDA. This evaluator must reproduce it before
# anything else in here means anything (docs/15 acceptance protocol #1).
EXPECTED_BASELINE = {0: 0.8668725809428489, 1: 0.8370606353860833}
BASELINE_TOL = 1e-4
WEIGHTS = (0.0, 0.10, 0.20)   # predeclared: weight of the new seed
N_BOOT = 500

checkpoints, encoders = {}, {}
for root, dirs, files in os.walk(INPUT):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    if "config.json" in files and "dinov2" in root.lower():
        cfg = json.loads(Path(root, "config.json").read_text())
        encoders[int(cfg.get("hidden_size", -1))] = root
    for family, (dataset, _hidden, _seed) in FAMILIES.items():
        if dataset not in Path(root).parts:
            continue
        for name in files:
            if name.startswith("knee_slot_fold") and name.endswith(".pth") and "_snap_" not in name:
                fold = int(name.removeprefix("knee_slot_fold").removesuffix(".pth"))
                if fold in FOLDS:
                    assert (family, fold) not in checkpoints, f"duplicate {family} {fold}"
                    checkpoints[family, fold] = os.path.join(root, name)
assert set(checkpoints) == {(f, k) for f in FAMILIES for k in FOLDS}, sorted(checkpoints)
assert 384 in encoders and 768 in encoders, encoders
log(f"checkpoints: { {k: os.path.basename(v) for k, v in checkpoints.items()} }")

preds, targets, valid_idx = {}, {}, {}
provenance = []
for fold in FOLDS:
    sel = np.flatnonzero(frame["fold"].values == fold)
    valid_idx[fold] = sel
    targets[fold] = Y[sel]
    rows_idx = frame.iloc[sel]["row"].values
    for family, (_dataset, hidden, expected_seed) in FAMILIES.items():
        blob = torch.load(checkpoints[family, fold], map_location="cpu", weights_only=False)
        assert blob["fold"] == fold and blob["epochs"] == 24 and blob["head"] == "xattn"
        assert blob["labels"] == "labels_blend_v1.csv" and blob["size"] == TRAIN_SIZE
        if expected_seed is not None:
            assert blob["seed"] == expected_seed, (family, blob["seed"])
        model = SlotNet(encoders[hidden], pool=blob["pool"], head=blob["head"],
                        unfreeze_last=UNFREEZE_LAST)
        model.load_state_dict(blob["model"], strict=True)
        model.eval().to(device)
        started = time.time()
        preds[family, fold] = predict(model, rows_idx)
        score, per_target = macro_auc(Y[sel], preds[family, fold])
        provenance.append(dict(family=family, fold=fold, seed=blob.get("seed"),
                               tpu_epoch=blob["epoch"], tpu_recorded_score=float(blob["score"]),
                               cuda_score=score, per_target=per_target,
                               seconds=time.time() - started))
        log(f"{family:6s} fold {fold}: CUDA AUC {score:.6f}  (TPU-recorded {blob['score']:.6f}, "
            f"epoch {blob['epoch']}, seed {blob.get('seed')})")
        del model, blob
        gc.collect()
        torch.cuda.empty_cache()


def blend(fold, w_new):
    old = 0.5 * (preds["small", fold] + preds["base", fold])   # deployed: uniform raw-probability mean
    return (1.0 - w_new) * old + w_new * preds["seed1", fold]


# 1. Reproduce the baseline before comparing anything against it.
for fold in FOLDS:
    got, _ = macro_auc(targets[fold], blend(fold, 0.0))
    log(f"baseline reproduction fold {fold}: {got:.10f} vs expected {EXPECTED_BASELINE[fold]:.10f}")
    assert abs(got - EXPECTED_BASELINE[fold]) < BASELINE_TOL, \
        f"baseline not reproduced on fold {fold}: {got} vs {EXPECTED_BASELINE[fold]}"

# 2. Predeclared comparison.
results = {}
for fold in FOLDS:
    for w in WEIGHTS:
        score, per_target = macro_auc(targets[fold], blend(fold, w))
        results[fold, w] = dict(score=score, per_target=per_target)
        log(f"fold {fold} w_new={w:.2f}: {score:.6f}")

# 3. Diversity diagnostics (supporting evidence, not the decision metric).
def mean_corr(a, b):
    pear = [np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(a.shape[1])]
    spear = [spearmanr(a[:, j], b[:, j])[0] for j in range(a.shape[1])]
    return float(np.mean(pear)), float(np.mean(spear))

diversity = {}
for fold in FOLDS:
    diversity[fold] = {
        "seed1_vs_small": mean_corr(preds["seed1", fold], preds["small", fold]),
        "small_vs_base": mean_corr(preds["small", fold], preds["base", fold]),
        "seed1_vs_base": mean_corr(preds["seed1", fold], preds["base", fold]),
    }
    log(f"fold {fold} mean per-target (pearson, spearman): {diversity[fold]}")

# 4. Paired scanner-group bootstrap of the ensemble delta vs the baseline.
rng = np.random.default_rng(0)
groups = {f: frame.iloc[valid_idx[f]]["fingerprint"].fillna("unknown").astype(str).values
          for f in FOLDS}
group_rows = {f: [np.flatnonzero(groups[f] == g) for g in np.unique(groups[f])] for f in FOLDS}
boot = {w: {f: [] for f in FOLDS} for w in WEIGHTS if w > 0}
for _ in range(N_BOOT):
    idx = {}
    for f in FOLDS:
        pick = rng.integers(0, len(group_rows[f]), len(group_rows[f]))
        idx[f] = np.concatenate([group_rows[f][i] for i in pick])
    for w in boot:
        for f in FOLDS:
            base_s, _ = macro_auc(targets[f][idx[f]], blend(f, 0.0)[idx[f]])
            new_s, _ = macro_auc(targets[f][idx[f]], blend(f, w)[idx[f]])
            boot[w][f].append(new_s - base_s)


def ci(x):
    x = np.asarray(x)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


summary = dict(
    baseline_reproduced={f: EXPECTED_BASELINE[f] for f in FOLDS},
    standalone={f"{p['family']}_fold{p['fold']}": p["cuda_score"] for p in provenance},
    diversity={str(f): diversity[f] for f in FOLDS},
    comparison={},
)
for w in WEIGHTS:
    row = {}
    for f in FOLDS:
        row[f"fold{f}_score"] = results[f, w]["score"]
        row[f"fold{f}_delta"] = results[f, w]["score"] - results[f, 0.0]["score"]
    row["mean_score"] = float(np.mean([results[f, w]["score"] for f in FOLDS]))
    row["mean_delta"] = row["mean_score"] - float(np.mean([results[f, 0.0]["score"] for f in FOLDS]))
    if w > 0:
        for f in FOLDS:
            row[f"fold{f}_delta_ci95"] = ci(boot[w][f])
        mean_boot = np.mean([boot[w][f] for f in FOLDS], axis=0)
        row["mean_delta_ci95"] = ci(mean_boot)
        row["mean_delta_boot_frac_positive"] = float((mean_boot > 0).mean())
    summary["comparison"][f"w_new_{w:.2f}"] = row
summary["limitations"] = [
    "Folds 0 and 1 only; exploratory. Every checkpoint here was trained and epoch-selected on these same folds.",
    "Existing small/base checkpoints have no recorded seed (pre-date SEED recording).",
    "Bootstrap resamples scanner-fingerprint groups within each fold (paired across arms).",
]
Path("summary.json").write_text(json.dumps(summary, indent=2))
Path("provenance.json").write_text(json.dumps(provenance, indent=2))
np.savez_compressed(
    "predictions.npz",
    **{f"{fam}_fold{f}": preds[fam, f] for fam in FAMILIES for f in FOLDS},
    **{f"target_fold{f}": targets[f] for f in FOLDS},
    **{f"ids_fold{f}": frame.iloc[valid_idx[f]][ID].to_numpy(dtype=str) for f in FOLDS},
)
log(json.dumps(summary["comparison"], indent=2))
