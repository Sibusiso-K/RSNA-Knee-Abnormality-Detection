# Appended to the shared training input/validation definitions by build.py.
import gc
import json
from pathlib import Path

assert device.type == "cuda", "evaluation requires a GPU"
assert cache.shape[1:3] == (6, 6), cache.shape
# The raw train metadata contains 58 gold studies, but the exact label file
# used for every checkpoint omits one of them. The training script merges
# pixels with labels before it identifies gold rows, so these models used the
# remaining 57. Requiring 58 here would evaluate a different cohort.
assert len(gold_frame) == 57, "gold holdout must match the checkpoint training cohort"
assert len(frame) == 4349, "training coverage must match the checkpoint runs"
assert os.path.basename(label_path) == "labels_blend_v1.csv"
assert not frame[ID].duplicated().any()

families = {"small": ("knee-slot-6slice-24ep-v1", 384),
            "base": ("knee-slot-base-24ep-v1", 768)}
checkpoints, encoders = {}, {}
for root, dirs, files in os.walk(INPUT):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    if "config.json" in files and "dinov2" in root.lower():
        cfg = json.loads(Path(root, "config.json").read_text())
        encoders[int(cfg.get("hidden_size", -1))] = root
    for family, (dataset, hidden) in families.items():
        if dataset not in Path(root).parts:
            continue
        for name in files:
            if name.startswith("knee_slot_fold") and name.endswith(".pth"):
                fold = int(name.removeprefix("knee_slot_fold").removesuffix(".pth"))
                assert (family, fold) not in checkpoints, "duplicate model"
                checkpoints[family, fold] = os.path.join(root, name)
assert set(checkpoints) == {(family, f) for family in families for f in range(5)}

oof = {family: np.full((len(frame), len(TARGETS)), np.nan, np.float32) for family in families}
gold_predictions = {family: [] for family in families}
rows = []

for family, (_dataset, hidden) in families.items():
    for fold in range(5):
        path = checkpoints[family, fold]
        blob = torch.load(path, map_location="cpu", weights_only=False)
        assert blob["fold"] == fold and blob["epochs"] == 24
        assert blob["slices_per_slot"] == 6 and blob["size"] == TRAIN_SIZE
        assert blob["labels"] == "labels_blend_v1.csv" and blob["head"] == "xattn"
        model = SlotNet(encoders[hidden], pool=blob["pool"], head=blob["head"])
        model.load_state_dict(blob["model"], strict=True)
        model.eval().to(device)
        sel = np.flatnonzero(frame["fold"].values == fold)
        started = time.time()
        prediction = predict(model, frame.iloc[sel]["row"].values)
        oof[family][sel] = prediction
        score, per_target = macro_auc(Y[sel], prediction)
        gold_pred = predict(model, gold_frame["row"].values)
        gold_predictions[family].append(gold_pred)
        # `blob["score"]` was measured by XLA on TPU.  This evaluation and
        # submission both use CUDA, and the architecture has a measured,
        # systematic backend shift: the identical fold-0 model is bit-exact
        # on TPU yet differs by +0.0348 AUC on CUDA.  Keep the TPU score as
        # provenance, never as a CUDA reproducibility assertion.
        row = dict(family=family, fold=fold, gpu_score=score,
                   tpu_recorded_score=float(blob["score"]),
                   backend_delta=score-float(blob["score"]), seconds=time.time()-started,
                   per_target=per_target)
        rows.append(row)
        log(f"CUDA OOF {family} fold {fold}: AUC {score:.6f}, TPU recorded "
            f"{blob['score']:.6f}, backend delta {row['backend_delta']:+.6f}, "
            f"{row['seconds']:.1f}s")
        Path("fold_metrics.json").write_text(json.dumps(rows, indent=2))
        np.savez_compressed(f"{family}_fold{fold}.npz", ids=frame.iloc[sel][ID].to_numpy(dtype=str),
                            prediction=prediction, target=Y[sel], gold_prediction=gold_pred)
        assert np.isfinite(score), "CUDA OOF score is invalid"
        del model, blob
        gc.collect()
        torch.cuda.empty_cache()

def ranks(p):
    return pd.DataFrame(p).rank(pct=True).to_numpy(dtype=np.float32)

assert all(np.isfinite(p).all() for p in oof.values())
comparisons = []
for fold in range(5):
    sel = np.flatnonzero(frame["fold"].values == fold)
    small, base = oof["small"][sel], oof["base"][sel]
    candidates = {"small": small, "base": base,
                  "blend50_rank": (ranks(small)+ranks(base))/2,
                  "blend50_probability": (small+base)/2}
    for name, prediction in candidates.items():
        score, per_target = macro_auc(Y[sel], prediction)
        comparisons.append(dict(fold=fold, candidate=name, score=score, per_target=per_target))

gold_y = gold_truth.loc[gold_frame[ID]].values.astype(np.float32)
gold_ranked = {family: np.stack([ranks(p) for p in predictions])
               for family, predictions in gold_predictions.items()}
gold_candidates = {family: predictions.mean(axis=0) for family, predictions in gold_ranked.items()}
gold_candidates["blend50_rank"] = (gold_candidates["small"] + gold_candidates["base"])/2
gold_metrics = {}
for name, prediction in gold_candidates.items():
    score, per_target = macro_auc(gold_y, prediction, drop_undecided=False)
    gold_metrics[name] = dict(score=score, per_target=per_target)

summary = dict(folds=comparisons, gold=gold_metrics,
               mean_fold_auc={name: float(np.mean([r["score"] for r in comparisons if r["candidate"] == name]))
                              for name in ("small", "base", "blend50_rank", "blend50_probability")},
               limitations=["GPU/CUDA OOF is the deployment-aligned baseline; TPU/XLA checkpoint scores are provenance only.",
                            "Checkpoints were selected on these validation folds; OOF is not an untouched holdout.",
                            "OOF blend uses two held-out members per study; deployment uses ten members.",
                            "Gold contains only 58 studies; no per-target blend fitting on gold."])
Path("summary.json").write_text(json.dumps(summary, indent=2))
np.savez_compressed("oof_predictions.npz", ids=frame[ID].to_numpy(dtype=str),
                    fold=frame["fold"].values, target=Y, small=oof["small"], base=oof["base"],
                    gold_ids=gold_frame[ID].to_numpy(dtype=str), gold_target=gold_y,
                    gold_small=np.stack(gold_predictions["small"]), gold_base=np.stack(gold_predictions["base"]))
log(json.dumps(summary["mean_fold_auc"]))
log(f"gold58: {json.dumps(gold_metrics)}")
