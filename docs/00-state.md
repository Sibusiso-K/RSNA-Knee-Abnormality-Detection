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

1. **Join the competition** — sign in at
   [kaggle.com/competitions/rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
   and click *Join Competition* to accept the rules. Nothing downloads until this is done.
2. **Create an API token** — Kaggle → your avatar → Settings → API → *Create New Token*. It
   downloads `kaggle.json`. Put it at `C:\Users\lovilocal.adm\.kaggle\kaggle.json`.
   (See [07-environment.md](07-environment.md) for other machines.)
3. **Pull the CSVs and audit the extractor against reality** —
   `bash scripts/download_data.sh`, then:
   - `python scripts/extract_labels.py --audit 30` — read the mentions it finds and, more
     importantly, notice what it *misses*. Recall failures are silent.
   - `python scripts/extract_labels.py --evaluate` — score against the ~58 gold studies.
   - `python scripts/extract_labels.py --disagree` — dump conflicts with the report text attached.

Then tune patterns, and only afterwards move to the LLM extractor (Phase 1 step 2 in
[05-plan.md](05-plan.md)).

## Open questions the real data will answer

- Which languages actually appear, and in what proportion? Patterns currently cover EN/ES/PT/FR/
  DE/IT/NL/TR by guesswork.
- Do reports have section headers at all? `text.py` weights impression above findings, which does
  nothing if the reports are unstructured prose.
- Is `PatientSex` really missing, as a forum thread claims?
- How often is meniscal laterality actually stated? This decides `unresolved_laterality`
  (`drop` vs `both`) — currently `drop`, untested.
- How many studies genuinely carry gold labels? "~58" comes from a forum post, not from us.

## Blockers

| Blocker | Status | Notes |
|---|---|---|
| Not joined / no Kaggle API token | **Open** | Needs Sibusiso; 2 min of clicking |
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
