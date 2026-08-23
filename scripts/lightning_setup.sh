#!/usr/bin/env bash
# Bootstrap a Lightning AI Studio (or Colab, or any rented box) to train.
#
# Why this can exist at all: the training job needs the cache, its index and a
# label CSV — about 9 GB — and never opens a DICOM. The 570 GB of DICOMs stay
# on Kaggle. That is the whole reason kaggle_05_cache.py was worth building.
#
# Run this ON the Studio, not locally.
#
#   bash scripts/lightning_setup.sh
#   python notebooks/kaggle_06_train_slots.py
#
# Kaggle credentials: create a token at kaggle.com/settings -> API, then upload
# kaggle.json to the Studio and `export KAGGLE_CONFIG_DIR=<its folder>`. Do not
# paste the key into this file or any file that git tracks.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${KNEE_DATA_DIR:-$REPO/data}"
mkdir -p "$DATA"

echo "=== 1. dependencies ==="
pip install -q --upgrade pip
pip install -q torch torchvision transformers pandas numpy scikit-learn opencv-python-headless kaggle

echo "=== 2. GPU ==="
python - <<'PY'
import torch
if not torch.cuda.is_available():
    print("!! no CUDA. Attach a GPU to the Studio before training.")
else:
    p = torch.cuda.get_device_properties(0)
    gb = p.total_memory / 1024**3
    print(f"{p.name}  {gb:.1f} GB")
    # The cache is ~9 GB and is loaded into RAM, not VRAM, so the binding
    # constraint here is activations. Batch 8 studies = 48 slot images through
    # DINOv2-small at 336px fits comfortably in 16 GB; below that, halve BATCH.
    if gb < 15:
        print("!! under 15 GB: run with BATCH=4 (and expect ~2x the wall time)")
PY

echo "=== 3. cache (~9 GB, from the Kaggle kernel output) ==="
if [ ! -f "$DATA/index_train_0.csv" ]; then
  python -m kaggle kernels output sibusisokhumalo11/knee-cache -p "$DATA"
else
  echo "already present, skipping"
fi

echo "=== 4. labels ==="
# The blend chosen by scripts/compare_labels.py. Re-derive rather than commit
# the CSV: the label sets are public and versioned by their authors, and a
# stale copy that silently disagrees with the comparison is worse than a
# rebuild that takes ten seconds.
if [ ! -f "$DATA/labels_blend_v1.csv" ]; then
  TMP="$(mktemp -d)"
  for d in stevenleehans/rsna-knee-llm-report-labels \
           pilkwang/rsna-knee-llm-labels \
           lixin73/rsna-knee-llm-report-labels-sol56; do
    python -m kaggle datasets download "$d" -p "$TMP/$(basename "$d")" --unzip -q
  done
  PYTHONPATH="$REPO" python "$REPO/scripts/compare_labels.py" \
    --dir "$TMP" --train "$DATA/train.csv" --out "$DATA/labels_blend_v1.csv"
fi

echo "=== 5. ready ==="
echo "  EPOCHS=10 FOLDS=0 python notebooks/kaggle_06_train_slots.py"
echo
echo "Free-tier note: ~15 Lightning credits is roughly 22 T4-hours a month."
echo "One fold at 10 epochs on cached pixels is the unit to budget against —"
echo "measure fold 0 before committing credits to all five."
