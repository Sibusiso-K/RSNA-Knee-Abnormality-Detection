# 00 — Project state (READ THIS FIRST)

> **This is the living file.** It is the single source of truth for *where the project is right now*.
> Every other doc explains something stable; this one changes constantly.
> **Update it at the end of every working session.** If it's stale, everything else is a trap.

**Last updated:** 2026-08-08
**Days to final submission (2026-10-22):** ~75

---

## Where we are right now

**Phase 0 — Access: still blocked** (needs Sibusiso; ~2 minutes of clicking).
**Phase 1 — Label extraction: scaffolding built and tested against synthetic reports.**

The rule-based extractor is written, unit-tested (29 passing), and runs end to end on invented
reports in five languages. It has **never seen a real report** — every pattern in
`src/extract/patterns.py` is a hypothesis until audited against `train.csv`.

We still have **no data** and have **not joined the competition**. No imaging model exists. No
submission has been made.

### What runs today, with no data

```bash
python scripts/extract_labels.py --demo
```

```bash
python -m pytest tests/ -q
```

## The next three actions, in order

1. **Section-aware extraction.** Propagate anatomical context from template headers
   (`Medial Meniscus:`, `Lateral Compartment:`, `Cruciate Ligaments:`) into the sentences beneath
   them, and **exclude** indication/history/referral-question sections. Expected to be the largest
   single gain available — see the lead above.
2. **Re-measure**, then work the worst labels: Effusion (0.525), Medial OA (0.563),
   Contusion (0.581). Use `--disagree` to read actual conflicts rather than guessing.
3. **Then** the open-weights LLM extractor (Phase 1 step 2 in [05-plan.md](05-plan.md)), with the
   rule extractor as the baseline it has to beat.

Running in parallel, once labels stabilise: Phase 2's site-grouped CV harness needs DICOM headers,
which means a Kaggle notebook — nothing about it depends on the label work finishing.

## Measured facts (2026-08-08, from the real CSVs)

| | |
|---|---|
| Training studies | **4,407** |
| Gold-labelled studies | **58** (forum figure confirmed exactly) |
| Gold balance | Healthy — positives per label range 9 (MCL) to 35 (Effusion). Usable for evaluation, still far too few to train on |
| Series | 24,371, mean 5.5/study. Sagittal 9,864 · Coronal 8,609 · Axial 5,898 |
| `PatientSex` | **Absent** — forum report confirmed. Do not plan around it |
| `Fluid_Sensitive` vs `Fat_Suppression` | **Identical in all 24,371 rows.** One flag, not two — see [03-data-guide.md](03-data-guide.md) |
| Reports | 4,407, none empty. Median 977 chars, p90 2,118, max 4,743. Clean UTF-8 |
| Languages (crude probe) | EN ~1,746 · unclassified ~1,173 · ES 574 · TR 406 · DE 264 · NL 150 · FR 76 · IT 18 |

### Extractor baseline: **macro AUC 0.685** vs the 58 gold studies

Per-label AUC: ACL 0.822 · MCL 0.803 · Baker's 0.812 · Fracture 0.761 · Lateral Meniscus 0.735 ·
Lateral OA 0.699 · PF OA 0.683 · Synovitis 0.622 · Medial Meniscus 0.618 · Contusion 0.581 ·
Medial OA 0.563 · **Effusion 0.525**

> **This number is not a leaderboard estimate.** There are no reports at test time. It measures
> *label quality* — how good the training targets are that we hand the imaging model.

## The biggest open lead: reports are often *structured*

Header frequencies across all 4,407 reports show many are templated by anatomy, not free prose:

```
795 findings   687 conclusion   680 impresion   663 impression   604 technique
465 medial meniscus     454 lateral meniscus     406 indication
380 medial compartment  380 lateral compartment  363 patellofemoral compartment
354 cruciate ligaments  346 medial compartment cartilage   345 osseous structures
```

**The extractor currently throws this away.** In `Medial Meniscus: Tear of the posterior horn.`
the sentence splitter breaks at the colon, so the clause carrying the injury contains no meniscus
anchor and no laterality — no mention is created at all. That is very likely why Medial Meniscus
recall is 0.500 and Medial OA recall is 0.200 despite obvious positives in the text.

**Fix**: propagate an *anatomical section context* (structure + laterality) from the header down to
the sentences beneath it, instead of requiring every sentence to be self-contained. Highest-value
next change by a wide margin.

Second lead: the Dutch sample contains `Diagnostische vraagstelling: Meniscusscheur/mediaal?` — a
*referral question*, not a finding. Indication/history/question sections need to be **excluded**, or
they manufacture false positives. Likely explains Contusion precision 0.393.

## Blockers

| Blocker | Status | Notes |
|---|---|---|
| Not joined / no Kaggle API token | **Resolved 2026-08-08** | Token lives in `~/.kaggle/access_token`, not `kaggle.json` |
| 570 GB dataset vs 182 GB free disk | **Permanent constraint** | Never download DICOMs locally. Work in Kaggle notebooks. |
| No local NVIDIA GPU | **Permanent constraint** | All training on Kaggle (~30 GPU-h/week) or other cloud |
| Commercial LLM API on report text — allowed? | **Open, no host ruling** | Assume **not** allowed. Build on open weights. |
| External datasets (MRNet, OAI, fastMRI+) eligible? | **Open, no host ruling** | Don't design around them yet |

## Decision log

Decisions made and why, so we don't relitigate them. Append, don't rewrite.

| Date | Decision | Reasoning |
|---|---|---|
| 2026-08-08 | All heavy compute happens on Kaggle, not locally | 570 GB data + no local GPU makes local training impossible |
| 2026-08-08 | Treat this as an **imaging** problem, not multimodal inference | Host confirmed `Report` is absent from the test set; text is a label source only |
| 2026-08-08 | Label extraction is the primary investment | Only ~58 studies have real labels; everything else is report-derived |
| 2026-08-08 | Site-grouped CV from day one, never random folds | Forum evidence: random folds overstate macro AUC by ~0.053 |
| 2026-08-08 | Build the label extractor on open weights only | Commercial-LLM-API rules question is unresolved; open weights is safe either way |
| 2026-08-08 | Target the Efficiency track as the realistic win | 640 teams; efficiency is less crowded and rewards a lean single model |
| 2026-08-08 | Extractor emits **soft** scores (0..1), not binary | Metric is rank-based AUC; a hedged "possible tear" belongs between absent and present, and thresholding throws that away |
| 2026-08-08 | Pattern sets are **unioned across languages**, not dispatched by detected language | Radiology vocabulary rarely collides across languages, and a union beats a language detector that is wrong 5% of the time |
| 2026-08-08 | Extract **10 concepts**, not 12 labels; laterality splits meniscus and tibiofemoral OA | Reports describe one structure qualified by side, so this matches how the text is actually written |
| 2026-08-08 | Unresolved laterality **drops** the mention (configurable) | Conservative default. `--unresolved-laterality both` is the alternative; which is better is an empirical question for the gold set |

## Session log

Newest first. One short entry per session: what changed, what was learned.

### 2026-08-08 — Session 3
- Kaggle auth solved: the new `KGAT_`-style token does **not** go in `kaggle.json` (that is the
  legacy `username`+`key` scheme). It belongs in `~/.kaggle/access_token` as raw text, or use
  `python -m kaggle auth login`. Also `kaggle` is not on PATH in Git Bash — use `python -m kaggle`.
- Downloaded the five CSVs (8.8 MB total).
- Ran the extractor on real reports for the first time: **macro AUC 0.685** vs gold.
- Fixed a `NameError` in `evaluate.format_report` (`_fmt` was used but never defined).
- Recorded measured facts above; found the structured-report lead and the
  `Fluid_Sensitive == Fat_Suppression` data quirk.

### 2026-08-08 — Session 2
- Built the Phase 1 rule extractor: `src/extract/` (patterns, negation/uncertainty context,
  laterality resolution, aggregation) plus `scripts/extract_labels.py` and 29 unit tests.
- Design: 10 concepts → 12 labels; soft scores; languages unioned; NegEx-style clause-bounded
  negation.
- Three real bugs found by running it rather than by reading it: German adjective declension
  (`vorderen Kreuzband` missed), French `interne` unmatched as medial, and French elision
  (`pas d'epanchement`) failing to negate — that last one scored a *denied* effusion as present.
- **Every pattern is still unvalidated against real reports.**

### 2026-08-08 — Session 1
- Created repo `Sibusiso-K/RSNA-Knee-Abnormality-Detection`, scaffolded project skeleton.
- Researched the competition end to end (Overview, Data, forum). Three findings reshaped the
  approach: no `Report` at test time, only ~58 gold-labelled studies, no DICOM metadata shortcut.
- Discovered the hard environment constraints (570 GB data, 182 GB free, no GPU).
- Wrote the full documentation set.
- **Did not** download data — blocked on joining the competition.

---

## How to update this file

At the end of a session, edit three things: *Where we are right now*, *The next three actions*, and
add a **Session log** entry at the top of that list. Add to the decision log only when a real choice
was made. Change *Last updated*. Commit and push — that's what makes it available from anywhere.
