"""KAGGLE SUBMISSION NOTEBOOK — paste as a single cell.

Settings that must be right or the submit button stays greyed out:
  Accelerator: GPU T4 x2   |   Internet: OFF   |   Output: submission.csv

Attach as Datasets:
  - the competition data (automatic)
  - your trained checkpoints  -> /kaggle/input/knee-model-v1/
  - timm weights (offline)    -> /kaggle/input/timm-weights/   (only if the
    checkpoint doesn't already carry them; we build with pretrained=False and
    load our own state_dict, so usually not needed)

DEGRADED-MODE BEHAVIOUR IS DELIBERATE: if a study fails to load, or no
checkpoint is present at all, this writes 0.5 for that row instead of raising.
A submission that scores 0.5 still scores; a notebook that throws scores
nothing and burns a submission slot. Every fallback is logged so a silent
degradation can't masquerade as a real result.
"""

import os
import sys
import traceback

import numpy as np
import pandas as pd
import torch

COMP = "/kaggle/input/rsna-knee-abnormality-detection"
CKPT_DIR = "/kaggle/input/knee-model-v1"
SRC = "/kaggle/input/knee-src"  # optional: repo src/ as a dataset

if os.path.isdir(SRC):
    sys.path.insert(0, SRC)

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion",
    "Fracture",
]
ID = "StudyInstanceUID"
N_SLICES, SIZE, BATCH = 16, 224, 2

test = pd.read_csv(f"{COMP}/test.csv")
series = pd.read_csv(f"{COMP}/test_series.csv")
print(f"test studies: {len(test)}, series rows: {len(series)}")

submission = pd.DataFrame({ID: test[ID]})
for label in TARGETS:
    submission[label] = 0.5


def _write_and_exit(reason: str) -> None:
    submission.to_csv("submission.csv", index=False)
    print(f"WROTE FALLBACK submission.csv ({reason}) — shape {submission.shape}")


checkpoints = []
if os.path.isdir(CKPT_DIR):
    checkpoints = sorted(
        os.path.join(CKPT_DIR, f) for f in os.listdir(CKPT_DIR) if f.endswith(".pth")
    )
print(f"checkpoints found: {len(checkpoints)}")

if not checkpoints:
    # Pipeline-validation path: proves the notebook produces a scoreable file
    # end to end before any model exists. Expected score 0.500.
    _write_and_exit("no checkpoints — constant-0.5 baseline")
else:
    try:
        from src.data.dicom import load_study
        from src.model.net import KneeNet

        device = "cuda" if torch.cuda.is_available() else "cpu"
        models = []
        for path in checkpoints:
            state = torch.load(path, map_location=device)
            net = KneeNet(
                backbone=state.get("backbone", "tf_efficientnetv2_s.in21k_ft_in1k"),
                pretrained=False,
            )
            net.load_state_dict(state["model"])
            net.eval().to(device)
            models.append(net)
        print(f"loaded {len(models)} model(s) on {device}")

        preds = np.full((len(test), len(TARGETS)), 0.5, dtype=np.float32)
        failures = 0

        with torch.no_grad():
            for start in range(0, len(test), BATCH):
                chunk = test.iloc[start : start + BATCH]
                volumes, rows = [], []
                for offset, (_, row) in enumerate(chunk.iterrows()):
                    try:
                        volumes.append(
                            load_study(
                                f"{COMP}/test_series", row[ID], series, N_SLICES, SIZE
                            )
                        )
                        rows.append(start + offset)
                    except Exception:
                        failures += 1  # row keeps its 0.5 default

                if not volumes:
                    continue

                batch = torch.from_numpy(np.stack(volumes)).float().to(device)
                # Average probabilities, not logits: folds are separately
                # calibrated, and AUC only cares about ranking anyway.
                probs = torch.stack(
                    [torch.sigmoid(m(batch)) for m in models]
                ).mean(0)
                preds[rows] = probs.cpu().numpy()

                if start % 100 == 0:
                    print(f"  {start}/{len(test)}", flush=True)

        for i, label in enumerate(TARGETS):
            submission[label] = preds[:, i]
        submission.to_csv("submission.csv", index=False)
        print(f"WROTE submission.csv — shape {submission.shape}, load failures: {failures}")

    except Exception:
        traceback.print_exc()
        _write_and_exit("inference raised — see traceback above")

print(pd.read_csv("submission.csv").head())
