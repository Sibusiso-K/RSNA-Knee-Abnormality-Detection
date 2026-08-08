# 00 — Project state (READ THIS FIRST)

> **This is the living file.** It is the single source of truth for *where the project is right now*.
> Every other doc explains something stable; this one changes constantly.
> **Update it at the end of every working session.** If it's stale, everything else is a trap.

**Last updated:** 2026-08-08
**Days to final submission (2026-10-22):** ~75

---

## Where we are right now

**Phase 0 — Access: done.** Data is downloaded and the extractor has run against real reports.
**Phase 1 — Label extraction: 0.710 macro AUC vs the gold studies**, up from 0.685 at first contact.

The rule-based extractor now understands report *structure*, not just sentences: it parses
templated section headers (`Medial Meniscus:`, `Medial Compartment:`), inherits concept and
laterality from them when a sentence is bare ("Tear of the posterior horn."), and excludes
referral-question / technique sections that would otherwise manufacture false positives. It also
covers **Greek**, ~7% of the corpus and previously invisible entirely. 34 unit tests pass. No
imaging model exists yet. No submission has been made.

### What runs today

```bash
python scripts/extract_labels.py --evaluate       # score against the 58 gold studies
python scripts/extract_labels.py --audit 30        # read real extractions
python scripts/extract_labels.py --disagree        # dump conflicts with report text
python scripts/extract_labels.py --demo             # synthetic sanity check, no data needed
python -m pytest tests/ -q                          # 34 tests
```

## The next three actions, in order

1. **Work the new tail using `--disagree`.** Current per-label AUC: Medial Meniscus 0.637, Medial OA
   0.658, PF OA 0.656, Lateral OA 0.684, Contusion 0.640 are now the weakest. Medial Meniscus in
   particular is worth attention — it's a common, clinically important finding and sits well below
   Lateral Meniscus (0.739) despite using the same pattern logic; check whether medial-side reports
   have a phrasing quirk lateral doesn't.
2. **Audit Contusion's remaining false positives (precision 0.615, up from 0.385 but still the
   second-worst).** The explicit-word fix (below) closed most of the gap; the residual 5 FPs are
   worth a `--disagree` look before assuming they're noise.
3. **Then** the open-weights LLM extractor (Phase 1 step 2 in [05-plan.md](05-plan.md)), with this
   rule extractor — now at 0.710 — as the baseline it has to beat.

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

| | 3a: sentence-only | 3b: section-aware | 4: +Greek, +effusion synonyms, +contusion precision fix |
|---|---|---|---|
| Macro AUC | 0.685 | 0.688 | **0.710** |

Per-label AUC now: MCL 0.859 · ACL 0.846 · Baker's 0.823 · Fracture 0.747 · Lateral Meniscus 0.739 ·
Lateral OA 0.684 · PF OA 0.656 · Medial OA 0.658 · Contusion 0.640 · Medial Meniscus 0.637 ·
Effusion 0.596 · Synovitis 0.630

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

## Session 4: chasing Effusion and Contusion specifically

### Effusion: 0.533 → 0.596

Dumped every gold-positive Effusion study the extractor missed and read the reports directly.
Two categories of miss, both fixed:

- **A whole missing language.** `[Ͱ-Ͽ]` (Greek script) appears in **321/4,407 reports
  (7.3%)** — not a rounding error, a real chunk of the corpus the extractor had zero coverage for.
  Also found and fixed an upstream data quirk: every Greek **mu** (μ, U+03BC) in the corpus has
  been silently replaced with the **micro sign** (µ, U+00B5) — almost certainly a font/OCR
  substitution bug in however these reports were digitized. Folded it in `text.normalize()` so
  every Greek pattern works against the actual bytes in the data, then added Greek vocabulary
  for all ten concepts, negation, uncertainty, and laterality (not just effusion).
- **Circumlocution the direct-term list didn't cover**, per language:
  - Dutch: `hydrops` (a real medical term for joint effusion, easy to miss because in English
    "hydrops" usually means something unrelated), and `opzetting van de suprapatellaire recessus`
    (distension of the suprapatellar recess — named by anatomy, not a "fluid" word at all).
  - Turkish: reports here favour `sıvı artışı` / `sıvı artmış` (fluid increase) over a dedicated
    effusion word. Added a loosely-bound pattern requiring a joint/bursa word nearby, so it
    doesn't fire on unrelated fluid increases elsewhere in the same report.
  - Greek: `υγρό ενδαρθρικά` (intra-articular fluid), `συλλογή υγρού` (fluid collection).

### Contusion: 0.570 → 0.640 (precision 0.385 → 0.615)

The false-positive dump was unambiguous: almost every false positive was bare **"edema" + a nearby
bone word** (`subchondral`, `marrow`, `osseous`) — matching subchondral bone marrow edema from
osteoarthritis, an adjacent osteochondral lesion, or a reactive change near an unrelated ligament
tear. Gold labels reserve `Contusion` for the traumatic bone-bruise pattern specifically; bone
marrow edema alone doesn't distinguish it (see [02-domain-primer.md](02-domain-primer.md) —
contusion, fracture, and OA subchondral change all produce marrow edema, and only the explicit
wording or trauma context tells them apart from free text alone).

**Fix**: require the explicit word (`contusion`/`bruise`/`kontuzyon`/etc.) rather than bare
"edema" + bone. This trades some recall (misses a bone bruise described only as "marrow edema
pattern") for a lot of precision, and measured better on the gold set. A test that encoded the old
(wrong) assumption was rewritten rather than deleted, so the reasoning survives in the test file.

**Not yet fixed**: the equivalent ambiguity may also affect Fracture, which shares the same marrow-
edema vocabulary. Worth checking with `--disagree` before assuming it's clean.

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

### 2026-08-08 — Session 4
- Chased Effusion and Contusion specifically, via `--disagree`-style dumps of actual gold-label
  false negatives/positives rather than guessing at patterns.
- Effusion 0.533 → 0.596: found Greek is 7.3% of the corpus (321/4,407 reports) with zero prior
  coverage; found and fixed a corpus-wide data quirk where Greek mu is encoded as the micro sign;
  added Greek vocabulary for all ten concepts; added Dutch (`hydrops`, suprapatellar-recess
  phrasing) and Turkish (`sıvı artışı`) effusion synonyms the direct-term list was missing.
- Contusion 0.570 → 0.640, precision 0.385 → 0.615: bare "edema" + a nearby bone word was firing on
  OA/reactive subchondral edema, not just true contusions. Now requires the explicit
  contusion/bruise word. One test's assumption was disproven by the data and rewritten in place.
- **Net: 0.688 → 0.710 macro AUC vs gold.** 34 tests still passing.
- Flagged but not investigated: Fracture may share the same marrow-edema ambiguity that hit
  Contusion. Medial Meniscus (0.637) now trails Lateral Meniscus (0.739) by a wide margin for no
  obvious reason yet.

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
