# Appended to the shared training input/validation definitions by build.py.
# Diagnostic only: characterise the evaluate-fresh24 mismatch across every
# checkpoint before attempting a fix. Nothing here is asserted away - every
# check evaluate-fresh24 makes is still made and LOGGED, just not fatal.
import gc
import hashlib
import json
from pathlib import Path

assert device.type == "cuda", "evaluation requires a GPU"
assert cache.shape[1:3] == (6, 6), cache.shape

with open(label_path, "rb") as fh:
    label_hash = hashlib.sha256(fh.read()).hexdigest()
log(f"label_path resolved to: {label_path}")
log(f"label file sha256: {label_hash}")
log(f"cache_dir resolved to: {cache_dir}")
log(f"train_csv resolved to: {find_dir('train.csv')}")
log(f"gold_frame: {len(gold_frame)} rows | frame: {len(frame)} rows")
log(f"frame duplicated ids: {int(frame[ID].duplicated().sum())}")

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
log(f"encoders found: { {h: p for h, p in encoders.items()} }")

rows = []
mismatches = []
for family, (_dataset, hidden) in families.items():
    for fold in range(5):
        path = checkpoints[family, fold]
        blob = torch.load(path, map_location="cpu", weights_only=False)
        checks = dict(
            fold_matches=blob["fold"] == fold, epochs_24=blob["epochs"] == 24,
            slices_6=blob["slices_per_slot"] == 6, size_matches=blob["size"] == TRAIN_SIZE,
            labels_match=blob["labels"] == "labels_blend_v1.csv", head_xattn=blob["head"] == "xattn",
        )
        log(f"{family} fold {fold} checkpoint checks: {checks}")

        model = SlotNet(encoders[hidden], pool=blob["pool"], head=blob["head"],
                        unfreeze_last=UNFREEZE_LAST)
        model.load_state_dict(blob["model"], strict=True)
        model.eval().to(device)
        sel = np.flatnonzero(frame["fold"].values == fold)
        started = time.time()
        prediction = predict(model, frame.iloc[sel]["row"].values)
        score, per_target = macro_auc(Y[sel], prediction)
        delta = score - float(blob["score"])
        row = dict(family=family, fold=fold, score=score, recorded_score=float(blob["score"]),
                   delta=delta, seconds=time.time() - started, checks=checks,
                   n_valid=int(len(sel)))
        rows.append(row)
        log(f"OOF {family} fold {fold}: AUC {score:.6f}, recorded {blob['score']:.6f}, "
            f"delta {delta:+.6f}, n={len(sel)}, {row['seconds']:.1f}s")
        Path("fold_metrics.json").write_text(json.dumps(rows, indent=2))
        # Save the exact validation ID set for this fold/family so a later,
        # narrower investigation can diff it against anything else (a
        # different label snapshot, a different cache) without a full rerun.
        np.savez_compressed(
            f"{family}_fold{fold}_ids.npz",
            ids=frame.iloc[sel][ID].to_numpy(dtype=str),
            fingerprints=frame.iloc[sel]["fingerprint"].to_numpy(dtype=str),
        )
        if abs(delta) >= 0.003:
            mismatches.append(row)
        del model, blob
        gc.collect()
        torch.cuda.empty_cache()

log(f"\n{len(mismatches)}/10 checkpoints mismatch by >= 0.003")
log(json.dumps([{k: v for k, v in r.items() if k != "checks"} for r in rows], indent=2))
Path("summary.json").write_text(json.dumps(dict(rows=rows, n_mismatch=len(mismatches)), indent=2))
