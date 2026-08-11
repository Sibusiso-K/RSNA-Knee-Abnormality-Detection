#!/usr/bin/env bash
# DINOv2 recipe, copied from what the public 0.899 baseline actually does.
#
# Our EfficientNetV2-S pipeline measures 0.7767 and the free public notebook
# measures 0.899. The difference is not tuning: they adapt a self-supervised
# ViT with a near-frozen encoder instead of retraining an ImageNet CNN. Every
# preprocessing tweak we have tried returned ~0.002; this is the only
# identified difference plausibly worth ~0.1.
#
# Three coupled changes, all required together:
#   1. BACKBONE -> the mounted DINOv2 dir. A path (not a timm name) is what
#      switches KneeNet to DinoEncoder.
#   2. Discriminative LR: backbone 8e-6, head 1e-3. A single 3e-4 across a
#      self-supervised ViT destroys the pretrained features - the whole point
#      is to adapt, not retrain.
#   3. SIZE stays 224. DINOv2 is patch-14, so the side must be a multiple of
#      14: 224 = 14x16 and 336 = 14x24, but 256 is NOT. The train-hires 256px
#      config is invalid for this encoder and DinoEncoder raises on it.
#
# Slices/epochs held at 16/4 so this isolates the backbone change against the
# 0.7746 and 0.7767 runs.
set -euo pipefail
sed -e 's|^BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"|BACKBONE = "/kaggle/input/dinov2/pytorch/base/1"|' \
    -e 's|^BATCH, ACCUM, EPOCHS, LR = 2, 4, 8, 3e-4|BATCH, ACCUM, EPOCHS, LR = 2, 4, 4, 1e-3|' \
    -e 's|^    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)|    optimizer = torch.optim.AdamW([\n        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 8e-6},\n        {"params": list(model.pools.parameters()) + list(model.head.parameters()), "lr": LR},\n    ], weight_decay=1e-2)|' \
    -e 's|optimizer, max_lr=LR, total_steps=EPOCHS \* steps_per_epoch + EPOCHS|optimizer, max_lr=[8e-6, LR], total_steps=EPOCHS * steps_per_epoch + EPOCHS|'     "$1/notebooks/kaggle_02_train.py" > "$2/script.py"
grep -q 'BACKBONE = "/kaggle/input/dinov2' "$2/script.py" || { echo "backbone patch missed" >&2; exit 1; }
grep -q '"lr": 8e-6'                       "$2/script.py" || { echo "discriminative LR patch missed" >&2; exit 1; }
grep -q 'EPOCHS, LR = 2, 4, 4, 1e-3'       "$2/script.py" || { echo "epoch/LR patch missed" >&2; exit 1; }
grep -q '^N_SLICES, SIZE = 16, 224'        "$2/script.py" || { echo "SIZE must stay 224 (patch-14)" >&2; exit 1; }
# OneCycleLR with a SCALAR max_lr overwrites every param group's lr, which would
# silently undo the discriminative rates above and retrain the ViT at 1e-3.
grep -q 'max_lr=\[8e-6, LR\]'             "$2/script.py" || { echo "max_lr must be a per-group list" >&2; exit 1; }
