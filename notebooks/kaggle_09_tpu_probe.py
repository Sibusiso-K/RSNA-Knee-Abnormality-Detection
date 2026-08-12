"""KAGGLE TPU PROBE — does SlotNet run on XLA at all, and how fast?

Settings: Accelerator **TPU VM v3-8** | Internet OFF

TPU quota is 20 h/week and, unlike the GPU quota, it is currently untouched.
That makes it the only accelerator available before the GPU refresh on
2026-08-15. It is also the one we have never used, so this probe exists to
find out what breaks BEFORE a real run is committed to it.

Four things are genuinely uncertain on XLA and each would sink a training run
in a different way:

1. **`F.grid_sample`** — the augmentation warp. XLA support exists but has
   historically been slow or absent. If it is either, augmentation moves to
   the host or gets replaced.
2. **`torch.topk`** — the focal pooling tail. Should lower fine; a fallback to
   `cls_mean` costs a third of the feature width if not.
3. **Recompilation.** XLA traces a graph per unique shape. Our shapes are
   static by construction, but the LAST batch of an epoch is short unless it is
   dropped, and one ragged shape means a full recompile every epoch.
4. **Step time.** A v3-8 has 8 cores. This probe measures ONE core, so the
   real throughput is up to 8x what it reports — but only if the 8-core spawn
   path also works, which is a separate question this does not answer.

It costs minutes against a 20 h budget. The alternative is discovering any of
the four an hour into a real run.
"""

import os
import sys
import time
import shutil

import numpy as np

PKG = "/kaggle/working/pkg"
INPUT = "/kaggle/input"
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def find_dir(marker, max_depth=6):
    if not os.path.isdir(INPUT):
        return None
    stack = [(INPUT, 0)]
    while stack:
        directory, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            entries = os.listdir(directory)
        except OSError:
            continue
        if marker in entries:
            return directory
        for entry in entries:
            path = os.path.join(directory, entry)
            if os.path.isdir(path):
                stack.append((path, depth + 1))
    return None


log("=== 1. environment ===")
import torch                                              # noqa: E402
import torch.nn.functional as F                            # noqa: E402

log(f"torch {torch.__version__}")

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    log(f"torch_xla {getattr(torch_xla, '__version__', 'unknown')}")
except Exception as exc:                                   # noqa: BLE001
    raise SystemExit(f"torch_xla unavailable: {type(exc).__name__}: {exc}")

try:
    devices = xm.get_xla_supported_devices()
    log(f"xla devices visible: {len(devices)} -> {devices}")
except Exception as exc:                                   # noqa: BLE001
    log(f"get_xla_supported_devices failed: {exc}")
    devices = []

device = xm.xla_device()
log(f"using {device}")

# --- 2. the two risky ops, in isolation ----------------------------------
log("\n=== 2. risky ops ===")


def timed(name, fn):
    """Run once to trace/compile, then time. Both numbers matter on XLA.

    The first call includes graph compilation, which on XLA can be seconds and
    is paid once per unique shape. Reporting only the warm number would hide a
    recompilation problem; reporting only the cold one would look catastrophic.
    """
    try:
        t = time.time()
        out = fn()
        xm.mark_step()
        if hasattr(out, "cpu"):
            out.cpu()
        cold = time.time() - t

        t = time.time()
        for _ in range(3):
            out = fn()
            xm.mark_step()
        if hasattr(out, "cpu"):
            out.cpu()
        warm = (time.time() - t) / 3.0
        log(f"  {name:22s} cold {cold:7.2f}s  warm {warm:7.3f}s  OK")
        return True
    except Exception as exc:                               # noqa: BLE001
        log(f"  {name:22s} FAILED: {type(exc).__name__}: {exc}")
        return False


x = torch.randn(48, 3, 336, 336, device=device)

grid_ok = timed("F.grid_sample", lambda: F.grid_sample(
    x,
    F.affine_grid(
        torch.eye(2, 3, device=device).unsqueeze(0).repeat(48, 1, 1),
        x.shape, align_corners=False,
    ),
    mode="bilinear", padding_mode="border", align_corners=False,
))

patch = torch.randn(48, 576, 384, device=device)
topk_ok = timed("torch.topk (focal)", lambda: patch.topk(72, dim=1).values.mean(1))

# --- 3. SlotNet forward + backward ---------------------------------------
log("\n=== 3. SlotNet ===")
_src = find_dir("labels.py")
if _src is None:
    raise SystemExit("knee-src not attached")
if not os.path.exists(PKG + "/src"):
    os.makedirs(PKG, exist_ok=True)
    shutil.copytree(_src, PKG + "/src")
sys.path.insert(0, PKG)

dinov2 = None
for root, dirs, files in os.walk(INPUT):
    dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
    if "config.json" in files and "dinov2" in root.lower():
        dinov2 = root
        break
if dinov2 is None:
    raise SystemExit("DINOv2 not attached")
log(f"encoder: {dinov2}")

from src.model.slotnet import SlotNet                      # noqa: E402
from src.data.slots import N_SLOT                          # noqa: E402

POOL = "cls_mean_focal" if topk_ok else "cls_mean"
if not topk_ok:
    log("!! topk failed — falling back to cls_mean, which costs a third of "
        "the feature width")

model = SlotNet(dinov2, unfreeze_last=6, pool=POOL).to(device)
log(f"hidden {model.vit.config.hidden_size} | pool {POOL} | "
    f"trainable {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")

optimizer = torch.optim.AdamW(model.param_groups(1e-3, 8e-6), weight_decay=0.02)
criterion = torch.nn.BCEWithLogitsLoss()

BATCH = int(os.environ.get("BATCH", "8"))
imgs = torch.randint(0, 255, (BATCH, N_SLOT, 3, 336, 336), dtype=torch.uint8)
mask = torch.ones(BATCH, N_SLOT)
target = torch.rand(BATCH, 12)

log(f"batch {BATCH} studies = {BATCH * N_SLOT} slot images per step")

step_times = []
for step in range(6):
    t = time.time()
    x = imgs.to(device)
    m = mask.to(device)
    y = target.to(device)
    loss = criterion(model(x, m), y)
    loss.backward()
    # xm.optimizer_step, not optimizer.step: on XLA this is what inserts the
    # cross-replica gradient reduction and the mark_step. Calling the plain
    # optimizer works on ONE core and silently does not synchronise on eight.
    xm.optimizer_step(optimizer, barrier=True)
    optimizer.zero_grad(set_to_none=True)
    step_times.append(time.time() - t)
    log(f"  step {step}: {step_times[-1]:.2f}s  loss {loss.item():.4f}")

warm = float(np.median(step_times[2:]))
log(f"\nwarm step: {warm:.2f}s for {BATCH} studies -> {BATCH / warm:.2f} study/s (1 core)")

# 4,349 training studies, 10 epochs. One core; a working 8-core spawn would
# divide this, but that is not what was measured here.
one_core_h = 4349 * 10 / (BATCH / warm) / 3600
log(f"projected fold @10 epochs, ONE core: {one_core_h:.1f} h")
log(f"  (if the 8-core spawn works: ~{one_core_h / 8:.1f} h)")
log(f"  TPU budget is 20 h/week.")

log("\n=== verdict ===")
log(f"grid_sample {'OK' if grid_ok else 'BROKEN -> augment on host or drop'}")
log(f"topk        {'OK' if topk_ok else 'BROKEN -> pool=cls_mean'}")
if one_core_h > 20:
    log("!! one core cannot finish a fold inside the weekly TPU budget. "
        "The 8-core spawn path is REQUIRED, not an optimisation.")
