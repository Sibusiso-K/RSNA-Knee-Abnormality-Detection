# Appended to the shared training input/validation definitions by build.py.
# small fold 0 ONLY, on whatever device auto-detected (should be XLA/TPU) -
# using the SHARED predict() unmodified, so this is exactly what training's
# own per-epoch validation call did.
import json
from pathlib import Path

log(f"device for this run: {device}  XLA={XLA}")
assert XLA, "this diagnostic must run on TPU to test the backend hypothesis"

path = None
encoder = None
for root, dirs, files in os.walk(INPUT):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    if "knee-slot-6slice-24ep-v1" in Path(root).parts and "knee_slot_fold0.pth" in files:
        path = os.path.join(root, "knee_slot_fold0.pth")
    if "config.json" in files and "dinov2" in root.lower():
        cfg = json.loads(Path(root, "config.json").read_text())
        if int(cfg.get("hidden_size", -1)) == 384:
            encoder = root
assert path, "small fold0 checkpoint not found"
assert encoder, "small (384-dim) DINOv2 not found"

blob = torch.load(path, map_location="cpu", weights_only=False)
log(f"checkpoint: fold {blob['fold']} recorded_score {blob['score']:.6f} epoch {blob['epoch']}")
model = SlotNet(encoder, pool=blob["pool"], head=blob["head"], unfreeze_last=UNFREEZE_LAST)
model.load_state_dict(blob["model"], strict=True)
model.eval().to(device)

sel = np.flatnonzero(frame["fold"].values == blob["fold"])
rows = frame.iloc[sel]["row"].values
started = time.time()
prediction = predict(model, rows)
score, per_target = macro_auc(Y[sel], prediction)
delta = score - float(blob["score"])
log(f"TPU/XLA OOF: AUC {score:.6f}  recorded {blob['score']:.6f}  "
    f"delta {delta:+.6f}  ({time.time() - started:.1f}s)")
Path("tpu_result.json").write_text(json.dumps(
    dict(score=score, recorded_score=float(blob["score"]), delta=delta, per_target=per_target),
    indent=2,
))
