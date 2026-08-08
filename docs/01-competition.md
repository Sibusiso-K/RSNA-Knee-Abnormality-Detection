# Competition intel — RSNA Knee Abnormality Detection

Gathered 2026-08-08 (comp started 2026-07-30). Sources: Overview/Data tabs + discussion forum.

## Hard facts

| | |
|---|---|
| Metric | Macro-averaged AUC-ROC over 12 binary labels |
| Test set | ~1,300 studies, hidden |
| Dataset size | **569.76 GB**, 819,640 files (DICOM + 5 CSVs) |
| Submission | Kaggle Notebook, ≤9 h, **no internet**, `submission.csv` |
| Final submission | 2026-10-22 (entry/merge 2026-10-15) |
| Prizes | $77k — 10 LB prizes ($5k–9k) + 3 efficiency prizes |
| Field (day 3) | 7,504 entrants / 640 teams |

## The three facts that decide the strategy

**1. `Report` does NOT exist at test time.**
Host (Po-Hao Chen) confirmed: "The test set will not have `Report` available, whether before or
after notebook submission." So this is *not* a multimodal inference problem. The reports are a
**training-time label source only**. Inference is pure imaging.

**2. Only ~58 training studies carry real per-condition labels.**
Everything else must be labelled by parsing the free-text report. Reports are multilingual
(~9 languages). **Label quality is the competition.** Whoever extracts the cleanest labels from
the reports wins — the vision model is almost secondary.

**3. There is no metadata shortcut.**
A forum probe (Oleksii Zhukov) found DICOM headers alone reach 0.6516 macro AUC on random folds
but only 0.5981 under scanner-grouped folds — the 0.053 gap is pure site memorization. Series
composition alone (the 4 columns already in `train_series.csv`) gives 0.5954. So public LB scores
of 0.8–0.9 reflect genuine image reading, not leakage.

**Corollary on validation**: use **GroupKFold on a scanner/site fingerprint**
(`Manufacturer + ManufacturerModelName + SoftwareVersions + ImagingFrequency + ReceiveCoilName`
→ 265 distinct fingerprints, top 20 cover 45.5% of studies). Random folds are optimistically
biased by ~0.05 AUC.

## Data layout

- `train.csv` — one row per study: `StudyInstanceUID`, `PatientSex` (documented but reportedly
  *missing* from the actual file — see forum), `Report`, + 12 binary labels (mostly blank).
- `train_series.csv` — one row per series: `SeriesInstanceUID`, `Fluid_Sensitive` (0/1),
  `Fat_Suppression` (0/1), `Anatomical_Plane` (Sagittal/Coronal/Axial).
- `train_series/<StudyUID>/<SeriesUID>/<SOPUID>.dcm` — 20–45 slices/series (median 30), long tail
  to a few hundred. Mixed transfer syntaxes (Explicit VR LE, JPEG Lossless, JPEG 2000, Implicit VR
  LE). 86 allowlisted metadata tags. Intensities/orientations/resolutions vary widely.
- Prevalence is **not** guaranteed constant across train / public LB / private LB.

## Labels (12)

`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`,
`Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`

Anatomy notes: sagittal is the workhorse plane for ACL/meniscus; coronal for MCL and
medial/lateral compartment OA; axial for PF OA and synovitis. Fluid-sensitive sequences
(T2/PD/STIR) carry effusion, contusion, Baker's cyst, synovitis.

## Rules watch-outs

- **Unresolved**: whether sending report text to a commercial LLM API (OpenAI/Anthropic/Google)
  to derive labels is permitted. The data-security rule forbids providing competition data to
  anyone who hasn't accepted the rules; at least one participant reads that as prohibiting it.
  **No host ruling yet.** Safe path: run an **open-weights multilingual LLM locally or in a Kaggle
  notebook** (e.g. Qwen) so text never leaves the environment. Assume commercial APIs are off
  limits until the host says otherwise.
- External public knee-MRI datasets (MRNet, fastMRI+, OAI, SKM-TEA) — all free but gated behind a
  click-through research agreement. Eligibility under "equally accessible at no cost" also awaiting
  a host ruling.
- Winners must open-source code + weights (CC-BY-NC 4.0), record a short video, and publish the
  model publicly.

## Efficiency track

`Efficiency = (AUC_benchmark − max_AUC) + RuntimeSeconds / 32400`, minimised. Eligible if ranked
above the `sample_submission.csv` benchmark on the private LB. A submission can win both tracks.
Given a crowded main LB, this is the more winnable track for a small team.

## Environment constraints (this machine)

- Free disk: **181.7 GB** — the 570 GB dataset **cannot** be downloaded locally.
- **No NVIDIA GPU** (`nvidia-smi` not found).
- ⇒ All DICOM work and training must happen **on Kaggle** (free T4×2 / P100, ~30 GPU-h per week)
  or another cloud GPU. Locally we only handle the CSVs (`train.csv`, `train_series.csv`,
  `sample_submission.csv`) — reports are text and small.
