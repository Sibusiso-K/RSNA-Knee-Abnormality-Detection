# 00 — Project state (READ THIS FIRST)

> **This is the living file.** It is the single source of truth for *where the project is right now*.
> Every other doc explains something stable; this one changes constantly.
> **Update it at the end of every working session.** If it's stale, everything else is a trap.

**Last updated:** 2026-08-08
**Days to final submission (2026-10-22):** ~75

---

## Where we are right now

**Phase 0 — Access: done.** Data is downloaded and the extractor has run against real reports.
**Phase 1 — Label extraction: first honest number in hand — 0.688 macro AUC vs the gold studies.**

The rule-based extractor now understands report *structure*, not just sentences: it parses
templated section headers (`Medial Meniscus:`, `Medial Compartment:`), inherits concept and
laterality from them when a sentence is bare ("Tear of the posterior horn."), and excludes
referral-question / technique sections that would otherwise manufacture false positives. 34 unit
tests pass. No imaging model exists yet. No submission has been made.

### What runs today

```bash
python scripts/extract_labels.py --evaluate       # score against the 58 gold studies
python scripts/extract_labels.py --audit 30        # read real extractions
python scripts/extract_labels.py --disagree        # dump conflicts with report text
python scripts/extract_labels.py --demo             # synthetic sanity check, no data needed
python -m pytest tests/ -q                          # 34 tests
```

## The next three actions, in order

1. **Work the worst labels using `--disagree`, not guesswork.** Current per-label AUC: Effusion
   0.533, Contusion 0.570, Medial OA 0.605, PF OA 0.656, Lateral OA 0.684, Medial Meniscus 0.620 are
   the tail. Effusion in particular is suspicious — it's supposed to be the *easiest* label
   (see [02-domain-primer.md](02-domain-primer.md)) and is instead the worst. Read actual
   disagreements before changing patterns again.
2. **Investigate Contusion's precision (0.385, worst of all twelve).** 16 false positives vs 10
   true positives suggests the bone-context section fallback (`ctx.bone_context`) is over-firing —
   possibly matching generic "oedema" under an `Osseous Structures:` header that's actually
   describing something else in the same section.
3. **Then** the open-weights LLM extractor (Phase 1 step 2 in [05-plan.md](05-plan.md)), with this
   rule extractor — now at 0.688 — as the baseline it has to beat.

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

### Extractor progress vs the 58 gold studies

| | Session 3a (sentence-only) | Session 3b (section-aware) |
|---|---|---|
| Macro AUC | 0.685 | **0.688** |

Roughly flat in aggregate, but that number hides the real work. Per-label AUC now:
ACL 0.825 · Baker's 0.823 · MCL 0.806 · Fracture 0.762 · Lateral Meniscus 0.739 · Lateral OA 0.684 ·
PF OA 0.656 · Synovitis 0.630 · Medial Meniscus 0.620 · Medial OA 0.605 · Contusion 0.570 ·
**Effusion 0.533**

> **This number is not a leaderboard estimate.** There are no reports at test time. It measures
> *label quality* — how good the training targets are that we hand the imaging model.

## What building section-awareness actually found

Reports are often templated by anatomy, not free prose — header frequencies across all 4,407:

```
795 findings   687 conclusion   680 impresion   663 impression   604 technique
465 medial meniscus     454 lateral meniscus     406 indication
380 medial compartment  380 lateral compartment  363 patellofemoral compartment
```

Built `src/extract/sections.py`: parses these headers, propagates concept + laterality down to
bare sentences beneath them ("Tear of the posterior horn." under `Medial Meniscus:`), and excludes
indication/technique/history sections so referral questions ("possible meniscal tear?") stop
manufacturing false positives.

**Two structural bugs turned up only by running it on real reports, not by reading the code:**

1. The header regex anchored on line-start (`^`) only. Many reports run every section on **one
   line separated by periods**, not newlines (`Técnica: ... Resultados: Rotura del LCA. ...`).
   Only the first header matched; everything after it — including the actual findings — fell
   inside that one section, and it happened to classify as *excluded*. This silently deleted
   real findings across an unknown number of reports and measurably regressed Effusion
   (0.525 → 0.458 mid-fix) before being caught and fixed.
2. A second, narrower case: a header can immediately follow *another* header's colon with zero
   sentence content between them (`FINDINGS: Medial Meniscus: Tear...`). The boundary set didn't
   include `:`, so nested headers like this weren't split either. Fixed alongside the first.

Also expanded OA vocabulary — `cartilage fissuring`, `marginal spurring`, `chondrosis` were missing
entirely, catching real positives (Medial OA AUC 0.563 → 0.605) at a small cost to Lateral OA / PF
OA precision that needs a closer look.

**Effusion regressed slightly overall (0.525 → 0.533, but was worse mid-flight) despite none of
this work targeting it directly — worth an `--disagree` pass before touching anything else,** since
it's supposed to be the easiest label per the domain primer and is instead the current worst.
**Contusion's precision (0.385) is the single worst number in the table** — 16 false positives — and
is the prime suspect for the `bone_context` section fallback over-firing.

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
- Downloaded the five CSVs (8.8 MB total). Ran the extractor on real reports for the first time:
  0.685 macro AUC vs gold. Fixed a `NameError` in `evaluate.format_report` (`_fmt` undefined).
  Recorded measured facts; found the structured-report lead and the
  `Fluid_Sensitive == Fat_Suppression` data quirk.
- Built `src/extract/sections.py`: header parsing, concept/laterality inheritance, section
  exclusion. Found and fixed two real header-regex bugs by running against real reports (see
  above) — one of which was silently deleting findings, not just missing them. Expanded OA
  vocabulary (fissuring/spurring/chondrosis). Added 5 regression tests (34 total, all passing).
  Net: **0.688 macro AUC**. Effusion and Contusion flagged as next targets, not yet fixed.

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
