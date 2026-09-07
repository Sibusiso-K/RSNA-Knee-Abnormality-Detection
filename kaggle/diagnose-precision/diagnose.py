# Appended to the shared training input/validation definitions by build.py.
# small fold 0 ONLY, run three ways: fp32 (no autocast), fp16 (what this GPU
# eval path defaults to), bf16 (what TPU training/scoring actually used).
import gc
import json
from pathlib import Path

assert device.type == "cuda", "evaluation requires a GPU"

path = None
for root, dirs, files in os.walk(INPUT):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    if "knee-slot-6slice-24ep-v1" in Path(root).parts and "knee_slot_fold0.pth" in files:
        path = os.path.join(root, "knee_slot_fold0.pth")
        break
assert path, "small fold0 checkpoint not found"

encoder = None
for root, dirs, files in os.walk(INPUT):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    if "config.json" in files and "dinov2" in root.lower():
        cfg = json.loads(Path(root, "config.json").read_text())
        if int(cfg.get("hidden_size", -1)) == 384:
            encoder = root
            break
assert encoder, "small (384-dim) DINOv2 not found"

blob = torch.load(path, map_location="cpu", weights_only=False)
log(f"checkpoint: fold {blob['fold']} recorded_score {blob['score']:.6f} epoch {blob['epoch']}")
model = SlotNet(encoder, pool=blob["pool"], head=blob["head"], unfreeze_last=UNFREEZE_LAST)
model.load_state_dict(blob["model"], strict=True)
model.eval().to(device)

sel = np.flatnonzero(frame["fold"].values == blob["fold"])
rows = frame.iloc[sel]["row"].values
y = Y[sel]


@torch.no_grad()
def predict_with(mode):
    """mode: 'fp32' | 'fp16' | 'bf16' - the SAME predict() logic, only the
    autocast context differs, to isolate precision as the one variable."""
    out = []
    for start in range(0, len(rows), BATCH):
        r = rows[start:start + BATCH]
        m = torch.from_numpy(mask[r]).float().to(device)
        passes = 1 if ALL_GROUPS else N_GROUPS
        acc = None
        for g in range(passes):
            x = torch.from_numpy(take_input(r, g)).to(device)
            if mode == "fp32":
                ctx = torch.autocast("cuda", enabled=False)
            elif mode == "fp16":
                ctx = torch.autocast("cuda", dtype=torch.float16)
            else:
                ctx = torch.autocast("cuda", dtype=torch.bfloat16)
            with ctx:
                logits = model(x, m, TRAIN_SIZE).float()
            acc = logits if acc is None else acc + logits
        probs = torch.sigmoid(acc / passes)
        out.append(probs.cpu().numpy())
    return np.concatenate(out)


results = {}
for mode in ("fp32", "fp16", "bf16"):
    started = time.time()
    pred = predict_with(mode)
    score, _ = macro_auc(y, pred)
    results[mode] = dict(score=score, seconds=time.time() - started)
    log(f"{mode}: AUC {score:.6f}  ({results[mode]['seconds']:.1f}s)")

results["recorded_score"] = float(blob["score"])
log(json.dumps(results, indent=2))
Path("precision_results.json").write_text(json.dumps(results, indent=2))
del model, blob
gc.collect()
torch.cuda.empty_cache()
