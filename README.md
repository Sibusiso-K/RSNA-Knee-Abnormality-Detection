# RSNA Knee Abnormality Detection

Working repo for the Kaggle competition: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

## The competition

Multimodal (MRI imaging + paired radiology report text) classification of 12 clinically important knee
abnormalities: `ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion,
Synovitis, Baker's, Contusion, Fracture`.

- **Metric**: macro-averaged AUC-ROC across the 12 targets.
- **Format**: code competition — submissions run as a Kaggle Notebook, <=9hr runtime, no internet access
  at submission time (public pretrained weights/data may be used if bundled in advance).
- **Timeline**: entry/team-merge deadline **Oct 15, 2026**, final submission **Oct 22, 2026**.
- **Submission format**:
  ```
  StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
  <uid_1>,0.5,0.5,...
  ```

## Project layout

```
data/               # raw + processed competition data (gitignored, not committed)
notebooks/           # exploration notebooks
src/                 # reusable code (data loading, model, training, inference)
submissions/         # generated submission.csv files (gitignored)
```

## Setup

1. Create and activate a virtualenv, then install deps:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Get a Kaggle API token: Kaggle → account settings → "Create New Token" → downloads `kaggle.json`.
   Place it at `C:\Users\<you>\.kaggle\kaggle.json`. You must have joined the competition on the
   Kaggle site first (accept the rules) or the download will 403.
3. Download the competition data:
   ```bash
   bash scripts/download_data.sh
   ```
4. Run the trivial baseline to confirm the pipeline works end-to-end:
   ```bash
   python src/baseline.py
   ```
   This writes `submissions/submission.csv` with a constant 0.5 for every target — just a format sanity
   check, not a real model.

## Status

- [x] Repo scaffolded
- [ ] Data downloaded
- [ ] EDA done
- [ ] First real baseline model
- [ ] First Kaggle submission
