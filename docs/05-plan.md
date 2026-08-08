# Plan of attack

Deadline 2026-10-22 — about **10 weeks**. Constraints and intel in
[competition-notes.md](competition-notes.md).

## Strategic read

The competition is won at the **label-extraction** step, not the model step. ~58 studies have real
labels; the rest must be derived from multilingual free text. Everyone gets the same images, so the
differentiator is (a) label quality and (b) a validation scheme that doesn't lie to you.

Target: a solid single-model solution with honest site-grouped CV, then push the **Efficiency
track**, which is more winnable for a small team than fighting 640 teams' ensembles on raw AUC.

## Phase 0 — Access (blocking, do first)

1. Join the competition on Kaggle and accept the rules.
2. Create a Kaggle API token → `C:\Users\lovilocal.adm\.kaggle\kaggle.json`.
3. Pull **only the CSVs** locally (~small); leave the 570 GB of DICOMs on Kaggle.
4. Post a forum reply asking the host to rule on the commercial-LLM-API question. Until answered,
   build the label extractor on open weights only.

## Phase 1 — Labels from reports (weeks 1–3) — *the main event*

1. EDA on `train.csv`: language distribution, report length, section structure, how many studies.
2. Build a **rule/regex extractor** per label per language as the baseline — fast, auditable, free.
   Start from the public extractor referenced on the forum, then improve it.
3. Upgrade to an **open-weights multilingual LLM** (Qwen-class) run in a Kaggle notebook, prompted
   per label with a strict schema. Handle negation ("no evidence of ACL tear"), uncertainty
   ("possible"), and laterality (medial vs lateral) — these are where naive regex dies.
4. **Validate the extractor against the ~58 gold-labelled studies.** This is the only ground truth
   there is; guard it, don't train on it.
5. Ship a versioned `labels_v1.csv` as a Kaggle Dataset so notebooks can consume it offline.

Exit criterion: extractor agreement with the 58 gold studies is high and per-label prevalence looks
clinically sane.

## Phase 2 — Validation harness + trivial baselines (week 3)

1. Build the **scanner fingerprint** and a `GroupKFold` split on it. Never report a random-fold
   number again.
2. Baselines to beat, in order: constant 0.5 → per-label prevalence → series-composition GBDT
   (~0.595) → full DICOM-metadata GBDT (~0.598 grouped). Anything image-based must clear ~0.60.
3. Wire up the submission notebook end to end with the trivial baseline so the pipeline is proven
   before the model exists.

## Phase 3 — Imaging model (weeks 4–8)

1. **Preprocessing**: decode DICOM (handle all 4 transfer syntaxes), per-series intensity
   normalization, resample to a fixed volume. Cache as a Kaggle Dataset of `.npy`/`.npz` so
   training doesn't re-decode 800k files every run. This step is a big chunk of the work.
2. **Series routing**: use `Anatomical_Plane` / `Fluid_Sensitive` / `Fat_Suppression` to pick the
   right series per label group rather than dumping every series into one model:
   - Sagittal → ACL, menisci
   - Coronal → MCL, Medial/Lateral OA
   - Axial → PF OA, synovitis
   - Fluid-sensitive → Effusion, Contusion, Baker's
3. **Architecture**: start 2.5D (per-slice 2D CNN backbone + attention pooling across slices) —
   cheap, strong, and efficiency-friendly. Compare against a 3D backbone / DINOv3-style pretrained
   features (a forum thread is exploring this) only if 2.5D plateaus.
4. Multi-label head with BCE, per-label positive weighting for rare findings (Fracture,
   Synovitis).
5. Track per-label AUC — the macro metric means a single weak rare label costs as much as ACL.

## Phase 4 — Efficiency track + finish (weeks 8–10)

1. Once accuracy plateaus, produce a lean variant: fewer slices, smaller backbone, half precision,
   no TTA, no ensemble. Measure runtime; the score is `(bench − max_AUC) + secs/32400`.
2. Submit both a max-accuracy and a max-efficiency selection.
3. Prep winners' obligations early *if* in contention: training code, short video, public weights.

## Risk register

| Risk | Mitigation |
|---|---|
| 570 GB can't be handled locally, no local GPU | Do everything in Kaggle notebooks; budget the ~30 GPU-h/week |
| Noisy report-derived labels cap the ceiling | Invest in Phase 1; validate on the 58 gold studies |
| Random-fold CV overstates by ~0.05 | Site-grouped CV from day one |
| Prevalence shift train → private LB | AUC is ranking-based so it's fairly robust; avoid threshold tuning |
| Commercial-LLM ruling goes against it | Build on open weights from the start |
| Oct 1–2 Mintek Hackathon final collides | Front-load Phase 1–3; treat late Sept as a freeze |
| 9 h runtime limit on ~1,300 test studies | Measure inference cost per study early, not in week 10 |
