# Appended to the shared training input/validation definitions by build.py.
# CUDA-side epoch-ranking comparison: for each fold, does CUDA agree with
# whichever epoch this fold's TPU training run picked as "best"?
import gc
import json
from pathlib import Path

assert device.type == "cuda", "evaluation requires a GPU"

DATASET = "knee-slot-6slice-24ep-snap-v1"
encoder = None
checkpoints = {}  # (fold, variant) -> path; variant in {"best", "e16", "e20", "e24"}
for root, dirs, files in os.walk(INPUT):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    if "config.json" in files and "dinov2" in root.lower():
        cfg = json.loads(Path(root, "config.json").read_text())
        if int(cfg.get("hidden_size", -1)) == 384:
            encoder = root
    if DATASET not in Path(root).parts:
        continue
    for name in files:
        if not (name.startswith("knee_slot_fold") and name.endswith(".pth")):
            continue
        rest = name[len("knee_slot_fold"):-len(".pth")]
        if "_snap_e" in rest:
            fold_str, epoch_str = rest.split("_snap_e")
            fold, variant = int(fold_str), f"e{int(epoch_str)}"
        else:
            fold, variant = int(rest), "best"
        assert (fold, variant) not in checkpoints, f"duplicate {fold} {variant}"
        checkpoints[fold, variant] = os.path.join(root, name)

assert encoder, "small (384-dim) DINOv2 not found"
VARIANTS = ("best", "e16", "e20", "e24")
expected = {(f, v) for f in range(5) for v in VARIANTS}
missing = expected - set(checkpoints)
assert not missing, f"missing checkpoints: {sorted(missing)}"
log(f"encoder: {encoder}")
log(f"found all {len(checkpoints)} expected checkpoints")

rows = []
for fold in range(5):
    sel = np.flatnonzero(frame["fold"].values == fold)
    rows_idx = frame.iloc[sel]["row"].values
    y = Y[sel]
    for variant in VARIANTS:
        path = checkpoints[fold, variant]
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
            fold=fold, variant=variant, tpu_epoch=blob["epoch"],
            tpu_recorded_score=float(blob["score"]), cuda_score=score,
            seconds=time.time() - started,
        )
        rows.append(row)
        log(f"fold {fold} {variant}: TPU epoch {blob['epoch']}  TPU score {blob['score']:.6f}  "
            f"CUDA score {score:.6f}  ({row['seconds']:.1f}s)")
        Path("epoch_ranking.json").write_text(json.dumps(rows, indent=2))
        del model, blob
        gc.collect()
        torch.cuda.empty_cache()

# Per fold: does the epoch TPU chose as "best" also win under CUDA scoring?
summary = []
for fold in range(5):
    fold_rows = [r for r in rows if r["fold"] == fold]
    tpu_pick = next(r for r in fold_rows if r["variant"] == "best")
    cuda_pick = max(fold_rows, key=lambda r: r["cuda_score"])
    summary.append(dict(
        fold=fold,
        tpu_best_epoch=tpu_pick["tpu_epoch"],
        tpu_best_cuda_score=tpu_pick["cuda_score"],
        cuda_best_variant=cuda_pick["variant"],
        cuda_best_score=cuda_pick["cuda_score"],
        agrees=cuda_pick["variant"] == "best",
        cuda_gain_if_switched=cuda_pick["cuda_score"] - tpu_pick["cuda_score"],
    ))
n_agree = sum(1 for s in summary if s["agrees"])
log(f"\nTPU/CUDA agree on best epoch: {n_agree}/5 folds")
log(json.dumps(summary, indent=2))
Path("summary.json").write_text(json.dumps(dict(rows=summary, n_agree=n_agree), indent=2))
