# Appended to the shared training input/validation definitions by build.py.
# CUDA comparison of the three UNDECIDED_WEIGHT pilot legs (folds 0,1 only).
import gc
import json
from pathlib import Path

assert device.type == "cuda", "evaluation requires a GPU"

DATASET = "knee-slot-uw-pilot-v1"
LEGS = {"control": 1.0, "w025": 0.25, "w000": 0.0}
VARIANTS = ("best", "e16", "e20", "e24")

encoder = None
checkpoints = {}  # (leg, fold, variant) -> path
for root, dirs, files in os.walk(INPUT):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    if "config.json" in files and "dinov2" in root.lower():
        cfg = json.loads(Path(root, "config.json").read_text())
        if int(cfg.get("hidden_size", -1)) == 384:
            encoder = root
    if DATASET not in Path(root).parts:
        continue
    for name in files:
        leg = next((L for L in LEGS if name.startswith(f"{L}_knee_slot_fold")), None)
        if leg is None or not name.endswith(".pth"):
            continue
        rest = name[len(f"{leg}_knee_slot_fold"):-len(".pth")]
        if "_snap_e" in rest:
            fold_str, epoch_str = rest.split("_snap_e")
            fold, variant = int(fold_str), f"e{int(epoch_str)}"
        else:
            fold, variant = int(rest), "best"
        key = (leg, fold, variant)
        assert key not in checkpoints, f"duplicate {key}"
        checkpoints[key] = os.path.join(root, name)

assert encoder, "small (384-dim) DINOv2 not found"
expected = {(leg, f, v) for leg in LEGS for f in (0, 1) for v in VARIANTS}
missing = expected - set(checkpoints)
assert not missing, f"missing checkpoints: {sorted(missing)}"
log(f"encoder: {encoder}")
log(f"found all {len(checkpoints)} expected checkpoints")

rows = []
for leg, undecided_weight in LEGS.items():
    for fold in (0, 1):
        sel = np.flatnonzero(frame["fold"].values == fold)
        rows_idx = frame.iloc[sel]["row"].values
        y = Y[sel]
        for variant in VARIANTS:
            path = checkpoints[leg, fold, variant]
            blob = torch.load(path, map_location="cpu", weights_only=False)
            assert blob["fold"] == fold, (blob["fold"], fold)
            assert blob["seed"] == SEED, "checkpoint seed does not match this evaluator's SEED"
            model = SlotNet(encoder, pool=blob["pool"], head=blob["head"], unfreeze_last=UNFREEZE_LAST)
            model.load_state_dict(blob["model"], strict=True)
            model.eval().to(device)
            started = time.time()
            pred = predict(model, rows_idx)
            score, per_target = macro_auc(y, pred)
            row = dict(
                leg=leg, undecided_weight=undecided_weight, fold=fold, variant=variant,
                tpu_epoch=blob["epoch"], tpu_recorded_score=float(blob["score"]),
                cuda_score=score, seconds=time.time() - started,
            )
            rows.append(row)
            log(f"{leg:8s} fold {fold} {variant}: TPU epoch {blob['epoch']}  "
                f"TPU score {blob['score']:.6f}  CUDA score {score:.6f}  "
                f"({row['seconds']:.1f}s)")
            Path("uw_comparison.json").write_text(json.dumps(rows, indent=2))
            del model, blob
            gc.collect()
            torch.cuda.empty_cache()

# Per fold, compare each leg's OWN best-on-TPU checkpoint under CUDA scoring
# - the deployment-relevant comparison, since "best" is what a real run
# would select and submit.
summary = []
for fold in (0, 1):
    fold_best = {r["leg"]: r for r in rows if r["fold"] == fold and r["variant"] == "best"}
    control_score = fold_best["control"]["cuda_score"]
    summary.append(dict(
        fold=fold,
        control_cuda_score=control_score,
        w025_cuda_score=fold_best["w025"]["cuda_score"],
        w025_delta=fold_best["w025"]["cuda_score"] - control_score,
        w000_cuda_score=fold_best["w000"]["cuda_score"],
        w000_delta=fold_best["w000"]["cuda_score"] - control_score,
    ))
log(json.dumps(summary, indent=2))
Path("summary.json").write_text(json.dumps(summary, indent=2))
