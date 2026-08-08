# 04 — Method: how we actually solve this

The reasoning behind every technical choice. [05-plan.md](05-plan.md) is the schedule; this is the
*why*.

---

## 1. Understand the metric first

**Macro-averaged AUC-ROC over the twelve labels.**

Three properties of this metric drive everything:

**AUC is a ranking metric.** It asks: take a random positive study and a random negative one — how
often does the model score the positive higher? Consequently:
- **Absolute calibration is irrelevant.** Predicting 0.001 vs 0.999 scores identically to 0.4 vs
  0.6, as long as the order is the same. Don't waste time tuning thresholds or calibrating.
- **It's robust to prevalence shift**, which matters because the organisers explicitly warn that
  prevalence differs across train / public LB / private LB.
- A constant prediction scores exactly **0.5**. That's the benchmark to beat.

**Macro-averaging means every label counts equally.** Fracture — rare, hard, scored 0.519 in the
forum's metadata probe — is worth exactly as much as ACL. **One label stuck at 0.55 costs you ~0.04
of final score.** That's usually the gap between 5th and 50th place. Improving your worst label is
almost always worth more than improving your best.

**Per-label AUC is the only diagnostic that matters.** Never look at just the macro number during
development; always print the twelve.

---

## 2. The central problem: labels

Only ~58 of the training studies carry real per-condition labels. Everything else must be derived
from free-text reports in ~9 languages. So:

> **The quality of your label extractor sets the ceiling on your model. The vision architecture
> mostly determines how close to that ceiling you get.**

Everyone has the same images. Not everyone will have the same labels. That asymmetry is where the
competition is won.

### How to build the extractor

**Step 1 — rules baseline.** Per-label, per-language keyword patterns with explicit negation
handling. Fast, free, fully auditable, and it gives you something to diff against later. A public
extractor already exists (referenced from the forum's metadata-probe thread) — start there rather
than from zero, then improve it.

**Step 2 — open-weights LLM.** A multilingual instruct model (Qwen-class) run **locally or in a
Kaggle notebook**, prompted per study to emit a strict JSON schema of twelve labels. This handles
the linguistic variety that regex can't.

> **Rules constraint:** whether report text may be sent to a *commercial* LLM API (OpenAI,
> Anthropic, Google) is **an open question on the forum with no host ruling**. The data-security
> rule forbids providing competition data to people who haven't accepted the rules. We assume it is
> **not permitted** and build on open weights, which is safe under either ruling.

**Step 3 — the three linguistic traps.** Explicitly test each:
- *Negation*: "no evidence of ACL tear" must not become ACL=1. Most naive extractors fail here.
- *Uncertainty*: "possible", "cannot exclude", "suspicious for". Pick a policy — probably treat
  hedged findings as positive, since AUC rewards ranking and a hedged mention genuinely raises the
  probability — and apply it consistently.
- *Laterality*: six labels depend on medial vs lateral. Resolve it within the sentence, and beware
  reports that establish laterality in one sentence and describe findings in the next.

**Step 4 — validate against the 58 gold studies.** They are your **only** ground truth. Use them
purely to *measure* extractor quality; do not train on them and do not tune so hard against 58
samples that you overfit them. Also sanity-check per-label prevalence against clinical expectation:
if your extractor says 40% of knees have fractures, it's broken.

**Step 5 — consider soft labels.** Instead of a hard 0/1, emit a confidence. Since the metric is
ranking-based, training on soft targets preserves information that thresholding throws away.

---

## 3. Validation: the thing most teams will get wrong

**Use `GroupKFold` on the scanner fingerprint. Never random folds.**

The forum probe measured this directly: DICOM metadata alone reaches **0.6516** macro AUC under
random folds but **0.5981** under scanner-grouped folds. That 0.053 gap is the model memorising
sites — studies from the same scanner leak between train and validation, and the model learns "site
12 has a lot of arthritis" rather than how to read a knee.

Fingerprint = `Manufacturer + ManufacturerModelName + SoftwareVersions + ImagingFrequency +
ReceiveCoilName` → 265 distinct groups, top 20 covering 45.5% of studies.

If your CV and the leaderboard disagree, **trust site-grouped CV** — with ~1,300 test studies split
into public and private, the public LB is a small and noisy sample.

### Baselines to beat, in order

| Baseline | Expected macro AUC | Purpose |
|---|---|---|
| Constant 0.5 | 0.500 | Proves the pipeline works |
| Per-label training prevalence | ~0.500 | Confirms AUC ignores calibration |
| Series-composition GBDT (the 4 `train_series.csv` columns) | ~0.595 | Free signal from protocol choice |
| Full DICOM-metadata GBDT | ~0.598 grouped | The no-pixels ceiling |
| **Any image model** | **must clear ~0.60** | Or it isn't reading the images at all |

That third row is important and cheap: it needs no pixel decoding whatsoever and gets you most of
the way to 0.60. Build it in Phase 2 as a sanity floor.

---

## 4. Architecture

### Why 2.5D first

The natural instinct is a 3D CNN on the volume. Start with **2.5D** instead:

> Run a standard 2D CNN backbone (pretrained on ImageNet) over each slice independently, then pool
> the per-slice features across the stack with attention, then classify.

Reasons:
- **Pretrained weights.** 2D ImageNet backbones are vastly better initialised than any 3D
  alternative. With noisy labels and limited compute, initialisation matters more than capacity.
- **Cheap.** Fits the 9-hour runtime cap and the ~30 GPU-h/week Kaggle quota.
- **Attention pooling gives interpretability for free** — you can see which slices the model used,
  which is a fast way to catch "it's looking at the wrong series".
- **Efficiency-track friendly**, and the efficiency track is our realistic win.

Escalate to a 3D backbone or self-supervised volumetric features (a forum thread is exploring
DINOv3-style pretraining) **only if 2.5D plateaus** — not before.

### Series routing

Don't pour every series into one model. Route by plane and sequence flags — the mapping from
[02-domain-primer.md](02-domain-primer.md):

| Label group | Plane | Sequence |
|---|---|---|
| ACL, Medial/Lateral Meniscus | Sagittal | Fluid-sensitive |
| MCL, Medial OA, Lateral OA | Coronal | Fluid-sensitive |
| PF OA, Synovitis | Axial | Fluid-sensitive |
| Effusion, Baker's | Axial/Sagittal | Fluid-sensitive |
| Contusion, Fracture | Any | Fluid-sensitive **+ fat-suppressed** |

Every study won't have every plane, so define a fallback ordering. A multi-label head over a shared
backbone with routed inputs is a reasonable middle ground between one giant model and twelve
separate ones.

### Loss and imbalance

Multi-label BCE with **per-label positive weighting** for the rare findings (Fracture, Synovitis).
Because macro-AUC weights all labels equally, the loss should not be dominated by the common ones.

### Resolution trade-off

A meniscal tear may span three slices out of thirty and a few dozen pixels. Effusion survives
brutal downsampling. **This tension is the whole Efficiency track**: find the smallest input
representation that doesn't destroy the hard, small-lesion labels.

---

## 5. Efficiency track

```
Efficiency = (AUC_benchmark − max_AUC_of_any_submission) + RuntimeSeconds / 32400
```

Minimised. The first term is identical for everyone (it's a constant derived from the benchmark and
the best submission overall)… **but your own AUC does not appear in it.** Read carefully: eligibility
requires only ranking above the `sample_submission.csv` benchmark on the private LB. Beyond that
gate, the ranking is driven by **runtime**.

*(This reading should be double-checked against the host's clarification thread — a forum thread is
open asking exactly this. Confirm before optimising for it.)*

Strategic implication: a **fast, decent** model may beat a **slow, excellent** one here. Against 640
teams stacking ensembles for the main leaderboard, this is the more winnable prize for a small team.
Practically: fewer slices, a smaller backbone, half precision, no TTA, no ensembling — and measure
inference cost per study **early**, not in week 10.

---

## 6. Things that will go wrong (pre-mortem)

| Failure | Prevention |
|---|---|
| Noisy labels cap performance and you can't tell | Validate the extractor on the 58 gold studies before training anything |
| CV looks great, LB doesn't | Site-grouped folds from day one |
| Model learns scanner, not pathology | Same; plus per-series intensity normalization |
| JPEG2000 / JPEG-Lossless decode failures found late | Test all four transfer syntaxes in week 1 |
| Re-decoding DICOMs each run burns the GPU quota | Cache preprocessed arrays as a Kaggle Dataset once |
| Inference exceeds 9 h on ~1,300 studies | Measure per-study cost early; extrapolate |
| Macro AUC dragged down by one dead label | Always print all twelve per-label AUCs |
| Overfitting to 58 gold studies | Use them to measure, never to train |
| Mintek Hackathon final (Oct 1–2) collides with crunch | Front-load; treat late September as frozen |
