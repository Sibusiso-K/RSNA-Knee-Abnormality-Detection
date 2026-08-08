# 00 — Project state (READ THIS FIRST)

> **This is the living file.** It is the single source of truth for *where the project is right now*.
> Every other doc explains something stable; this one changes constantly.
> **Update it at the end of every working session.** If it's stale, everything else is a trap.

**Last updated:** 2026-08-08
**Days to final submission (2026-10-22):** ~75

---

## Where we are right now

**Phase 0 — Access. Not started.** Nothing is unblocked until the Kaggle steps below are done.

We have a repo, a research write-up, and a plan. We have **no data** and have **not joined the
competition**. No model exists. No submission has been made.

## The next three actions, in order

1. **Join the competition** — sign in at
   [kaggle.com/competitions/rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
   and click *Join Competition* to accept the rules. Nothing downloads until this is done.
2. **Create an API token** — Kaggle → your avatar → Settings → API → *Create New Token*. It
   downloads `kaggle.json`. Put it at `C:\Users\lovilocal.adm\.kaggle\kaggle.json`.
   (See [07-environment.md](07-environment.md) for other machines.)
3. **Pull the CSVs only** — `bash scripts/download_data.sh` (it fetches only the small text files;
   see [03-data-guide.md](03-data-guide.md) for why we never pull the DICOMs).

Then Phase 1 (report label extraction) can begin — see [05-plan.md](05-plan.md).

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

## Session log

Newest first. One short entry per session: what changed, what was learned.

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
