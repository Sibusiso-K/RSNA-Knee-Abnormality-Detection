# Appended to the shared training input/validation definitions by build.py.
# docs/15 step 4: inference-only translation TTA on the deployed small+base
# 24-epoch checkpoints. One predeclared view set (src/model/tta.py), fixed
# equal weights, applied ONCE - nothing here is tuned on these folds.
import gc
import json
from pathlib import Path

from src.model.tta import TTA_SHIFT, TTA_VIEWS, TTA_WEIGHTS, translate_images

assert device.type == "cuda", "evaluation requires a GPU"
assert TTA_SHIFT <= AUG_SHIFT, "TTA shift must stay inside the training augmentation range"
assert abs(sum(TTA_WEIGHTS) - 1.0) < 1e-12 and len(TTA_VIEWS) == len(TTA_WEIGHTS)

FOLDS = range(5)
FAMILIES = {
    "small": ("knee-slot-6slice-24ep-v1", 384),
    "base": ("knee-slot-base-24ep-v1", 768),
}
# evaluate-fresh24 v5 (CUDA, identity-only): what this evaluator must
# reproduce BEFORE any TTA number means anything.
EXPECTED = {
    "small": {0: 0.8553954416888035, 1: 0.8250857423073402, 2: 0.8557949259542834,
              3: 0.8473895675548629, 4: 0.8597334605933492},
    "base": {0: 0.8630616606830902, 1: 0.8321196156057974, 2: 0.854345472363285,
             3: 0.8510313612064943, 4: 0.8657413642198831},
    "blend": {0: 0.8668725809428489, 1: 0.8370606353860833, 2: 0.8621253135060387,
              3: 0.8565548711722881, 4: 0.869505500626961},
}
EXPECTED_MEAN_BLEND = 0.858423780326844
REPRO_TOL = 1e-4
N_BOOT = 500

checkpoints, encoders = {}, {}
for root, dirs, files in os.walk(INPUT):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    if "config.json" in files and "dinov2" in root.lower():
        cfg = json.loads(Path(root, "config.json").read_text())
        encoders[int(cfg.get("hidden_size", -1))] = root
    for family, (dataset, _hidden) in FAMILIES.items():
        if dataset not in Path(root).parts:
            continue
        for name in files:
            if name.startswith("knee_slot_fold") and name.endswith(".pth") and "_snap_" not in name:
                fold = int(name.removeprefix("knee_slot_fold").removesuffix(".pth"))
                assert (family, fold) not in checkpoints, f"duplicate {family} {fold}"
                checkpoints[family, fold] = os.path.join(root, name)
assert set(checkpoints) == {(f, k) for f in FAMILIES for k in FOLDS}, sorted(checkpoints)
assert 384 in encoders and 768 in encoders, encoders


@torch.no_grad()
def predict_views(model, rows):
    """(n, n_views, 12) group-averaged LOGITS, one column per TTA view.

    Same structure as predict(): per batch, per slice group, model logits,
    averaged over groups - the only difference is the loop over views. The
    raw group image batch is loaded once and reused for every view, so the
    cost of a view is a forward pass, not another cache read.
    """
    model.eval()
    out = []
    passes = 1 if ALL_GROUPS else N_GROUPS
    for start in range(0, len(rows), BATCH):
        sel = rows[start:start + BATCH]
        m = torch.from_numpy(mask[sel]).float().to(device)
        acc = [None] * len(TTA_VIEWS)
        for g in range(passes):
            x0 = torch.from_numpy(take_input(sel, g)).to(device)
            for vi, (tx, ty) in enumerate(TTA_VIEWS):
                x = translate_images(x0, tx, ty)
                with autocast():
                    logits = model(x, m, TRAIN_SIZE).float()
                acc[vi] = logits if acc[vi] is None else acc[vi] + logits
        out.append(torch.stack([a / passes for a in acc], dim=1).cpu().numpy())
    return np.concatenate(out)


def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z.astype(np.float64)))).astype(np.float32)


logits, ident_ref, targets, groups, ids, cache_rows = {}, {}, {}, {}, {}, {}
runtime, provenance = [], []
for fold in FOLDS:
    sel = np.flatnonzero(frame["fold"].values == fold)
    rows_idx = frame.iloc[sel]["row"].values
    targets[fold] = Y[sel]
    groups[fold] = frame.iloc[sel]["fingerprint"].fillna("unknown").astype(str).values
    ids[fold] = frame.iloc[sel][ID].to_numpy(dtype=str)
    cache_rows[fold] = rows_idx
    for family, (_dataset, hidden) in FAMILIES.items():
        blob = torch.load(checkpoints[family, fold], map_location="cpu", weights_only=False)
        assert blob["fold"] == fold and blob["epochs"] == 24 and blob["head"] == "xattn"
        assert blob["labels"] == "labels_blend_v1.csv" and blob["size"] == TRAIN_SIZE
        model = SlotNet(encoders[hidden], pool=blob["pool"], head=blob["head"],
                        unfreeze_last=UNFREEZE_LAST)
        model.load_state_dict(blob["model"], strict=True)
        model.eval().to(device)

        torch.cuda.synchronize(); t0 = time.time()
        ident_ref[family, fold] = predict(model, rows_idx)          # the baseline path, untouched
        torch.cuda.synchronize(); t_ident = time.time() - t0
        t0 = time.time()
        logits[family, fold] = predict_views(model, rows_idx)
        torch.cuda.synchronize(); t_tta = time.time() - t0

        # Identity view must be the baseline, numerically.
        ident_from_views = sigmoid(logits[family, fold][:, 0])
        max_diff = float(np.abs(ident_from_views - ident_ref[family, fold]).max())
        assert max_diff < 1e-5, f"identity view != predict() for {family} fold {fold}: {max_diff}"

        n = len(rows_idx)
        runtime.append(dict(family=family, fold=fold, n_studies=n,
                            identity_seconds=t_ident, tta_seconds=t_tta,
                            identity_s_per_study=t_ident / n, tta_s_per_study=t_tta / n,
                            tta_over_identity=t_tta / t_ident))
        provenance.append(dict(family=family, fold=fold, tpu_epoch=blob["epoch"],
                               tpu_recorded_score=float(blob["score"]),
                               identity_view_max_abs_diff_vs_predict=max_diff))
        s_id, _ = macro_auc(Y[sel], ident_ref[family, fold])
        log(f"{family:5s} fold {fold}: identity AUC {s_id:.6f}  "
            f"(expected {EXPECTED[family][fold]:.6f})  identity {t_ident:.1f}s  "
            f"TTAx{len(TTA_VIEWS)} {t_tta:.1f}s")
        assert abs(s_id - EXPECTED[family][fold]) < REPRO_TOL, \
            f"baseline not reproduced: {family} fold {fold}: {s_id} vs {EXPECTED[family][fold]}"
        del model, blob
        gc.collect()
        torch.cuda.empty_cache()


def tta_prob(family, fold):
    w = np.asarray(TTA_WEIGHTS, dtype=np.float64)
    z = (logits[family, fold].astype(np.float64) * w[None, :, None]).sum(axis=1)
    return sigmoid(z)


def arm(name, fold):
    """Predictions for one comparison arm on one fold."""
    if name == "small_id":
        return ident_ref["small", fold]
    if name == "base_id":
        return ident_ref["base", fold]
    if name == "blend_id":
        return 0.5 * (ident_ref["small", fold] + ident_ref["base", fold])
    if name == "small_tta":
        return tta_prob("small", fold)
    if name == "base_tta":
        return tta_prob("base", fold)
    if name == "blend_tta":
        return 0.5 * (tta_prob("small", fold) + tta_prob("base", fold))
    raise KeyError(name)


PAIRS = {"small": ("small_id", "small_tta"), "base": ("base_id", "base_tta"),
         "blend": ("blend_id", "blend_tta")}

# 1. Baseline reproduction (deployed blend, identity-only).
blend_id_scores = {}
for fold in FOLDS:
    blend_id_scores[fold], _ = macro_auc(targets[fold], arm("blend_id", fold))
    assert abs(blend_id_scores[fold] - EXPECTED["blend"][fold]) < REPRO_TOL, \
        (fold, blend_id_scores[fold], EXPECTED["blend"][fold])
mean_blend_id = float(np.mean(list(blend_id_scores.values())))
log(f"identity blend mean {mean_blend_id:.10f} vs expected {EXPECTED_MEAN_BLEND:.10f}")
assert abs(mean_blend_id - EXPECTED_MEAN_BLEND) < REPRO_TOL

# 2. Scores, fold deltas and per-target deltas.
scores, per_target = {}, {}
for name in ("small_id", "small_tta", "base_id", "base_tta", "blend_id", "blend_tta"):
    for fold in FOLDS:
        scores[name, fold], per_target[name, fold] = macro_auc(targets[fold], arm(name, fold))

# Per-view standalone AUC (diagnostic only: shows whether any single view is
# an outlier; nothing is selected from it).
per_view_auc = {}
for family in FAMILIES:
    for fold in FOLDS:
        for vi, view in enumerate(TTA_VIEWS):
            s, _ = macro_auc(targets[fold], sigmoid(logits[family, fold][:, vi]))
            per_view_auc[f"{family}_fold{fold}_view{vi}_{view}"] = s

# 3. Paired scanner-group bootstrap (same resample for both arms).
rng = np.random.default_rng(0)
group_rows = {f: [np.flatnonzero(groups[f] == g) for g in np.unique(groups[f])] for f in FOLDS}
boot = {p: {f: [] for f in FOLDS} for p in PAIRS}
for _ in range(N_BOOT):
    idx = {}
    for f in FOLDS:
        pick = rng.integers(0, len(group_rows[f]), len(group_rows[f]))
        idx[f] = np.concatenate([group_rows[f][i] for i in pick])
    for p, (a, b) in PAIRS.items():
        for f in FOLDS:
            sa, _ = macro_auc(targets[f][idx[f]], arm(a, f)[idx[f]])
            sb, _ = macro_auc(targets[f][idx[f]], arm(b, f)[idx[f]])
            boot[p][f].append(sb - sa)


def ci(x):
    x = np.asarray(x)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


summary = dict(
    config=dict(views=[list(v) for v in TTA_VIEWS], weights=list(TTA_WEIGHTS),
                tta_shift=TTA_SHIFT, train_aug_shift=AUG_SHIFT, n_boot=N_BOOT,
                n_scanner_groups={str(f): int(len(group_rows[f])) for f in FOLDS}),
    baseline_reproduced=dict(blend_mean=mean_blend_id, expected_blend_mean=EXPECTED_MEAN_BLEND),
    comparison={}, per_view_auc=per_view_auc,
)
for p, (a, b) in PAIRS.items():
    row = {}
    for f in FOLDS:
        row[f"fold{f}_identity"] = scores[a, f]
        row[f"fold{f}_tta"] = scores[b, f]
        row[f"fold{f}_delta"] = scores[b, f] - scores[a, f]
        row[f"fold{f}_delta_ci95"] = ci(boot[p][f])
    deltas = [scores[b, f] - scores[a, f] for f in FOLDS]
    mean_boot = np.mean([boot[p][f] for f in FOLDS], axis=0)
    row["mean_identity"] = float(np.mean([scores[a, f] for f in FOLDS]))
    row["mean_tta"] = float(np.mean([scores[b, f] for f in FOLDS]))
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

by_family = {}
for fam in FAMILIES:
    rs = [r for r in runtime if r["family"] == fam]
    by_family[fam] = dict(
        identity_s_per_study=float(np.mean([r["identity_s_per_study"] for r in rs])),
        tta_s_per_study=float(np.mean([r["tta_s_per_study"] for r in rs])),
    )
N_HIDDEN_TEST = 1300     # approximate size of the hidden test set (docs/12)
proj_identity = sum(5 * N_HIDDEN_TEST * by_family[f]["identity_s_per_study"] for f in FAMILIES)
proj_tta = sum(5 * N_HIDDEN_TEST * by_family[f]["tta_s_per_study"] for f in FAMILIES)
summary["runtime"] = dict(
    by_family=by_family,
    projected_forward_seconds_10_members_1300_studies=dict(
        identity=proj_identity, tta=proj_tta, extra=proj_tta - proj_identity),
    note=("Measured on a Kaggle T4 including mmap cache reads; the submission "
          "kernel additionally decodes DICOM (dominant cost), which TTA does not repeat."),
)
summary["limitations"] = [
    "One predeclared view set, evaluated once; no amounts were tuned on these folds.",
    "Every checkpoint was trained and epoch-selected on the fold it is scored on (OOF is not an untouched test set).",
    "Bootstrap resamples scanner-fingerprint groups within each fold, paired across arms; fold 0 has few groups.",
]
Path("summary.json").write_text(json.dumps(summary, indent=2))
Path("runtime.json").write_text(json.dumps(runtime, indent=2))
Path("provenance.json").write_text(json.dumps(provenance, indent=2))
np.savez_compressed(
    "predictions.npz",
    **{f"ids_fold{f}": ids[f] for f in FOLDS},
    **{f"groups_fold{f}": groups[f] for f in FOLDS},
    **{f"cache_row_fold{f}": cache_rows[f] for f in FOLDS},
    **{f"target_fold{f}": targets[f] for f in FOLDS},
    **{f"{fam}_identity_prob_fold{f}": ident_ref[fam, f] for fam in FAMILIES for f in FOLDS},
    **{f"{fam}_view_logits_fold{f}": logits[fam, f] for fam in FAMILIES for f in FOLDS},
    **{f"{fam}_tta_prob_fold{f}": tta_prob(fam, f) for fam in FAMILIES for f in FOLDS},
    views=np.asarray(TTA_VIEWS, dtype=np.float32), weights=np.asarray(TTA_WEIGHTS, dtype=np.float32),
)
log(json.dumps({p: {k: v for k, v in r.items() if k != "per_target_delta"}
                for p, r in summary["comparison"].items()}, indent=2))
log(json.dumps(summary["runtime"], indent=2))
