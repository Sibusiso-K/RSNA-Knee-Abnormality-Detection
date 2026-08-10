# 09 — Handover

**Written 2026-08-10.** For picking this project up cold, on a different machine or account.
Read this, then [00-state.md](00-state.md). Everything else is reference.

Repo: `https://github.com/Sibusiso-K/RSNA-Knee-Abnormality-Detection` — **everything is committed
and pushed.** Nothing important lives only on this laptop.

---

## 1. Read this first if you are on a different account

| Asset | Owner | Portable? |
|---|---|---|
| Git repo | `Sibusiso-K` (GitHub) | ✅ Clone it |
| Kaggle datasets `knee-src`, `knee-labels`, `knee-model-v1` | `sibusisokhumalo11` | ⚠️ **Private** |
| Kaggle kernels `knee-smoke/train/train-8ep/submit/llm-labels` | `sibusisokhumalo11` | ⚠️ **Private** |
| Competition entry + submission slots | `sibusisokhumalo11` | ❌ Tied to that account |

**A different *Kaggle* account cannot see any of the above and would have to re-upload the
datasets and re-run the kernels from the repo — the code is all here, so that works, but the
trained checkpoint would have to be re-created or shared.** A different *Claude* account is fine:
just clone the repo.

**Competition rules matter here.** Do not share competition data or derived work product
(including `labels_*.csv`, which is derived from report text) outside the registered team. All
Kaggle datasets were created **private** for exactly this reason. Keep them private.

---

## 2. Where the project actually stands

| Phase | Status |
|---|---|
| 0 — Access | ✅ Data downloaded, API working |
| 1 — Labels from reports | ✅ **Ensemble (rule + LLM) 0.8234** vs gold, from 0.685 at first contact |
| 2 — Site-grouped CV | ✅ Built and **verified honest** — 151 groups / 4,349 studies |
| 3 — Imaging model | ✅ **Fold 0: 0.7746** grouped-CV macro AUC |
| 4 — Submission | ✅ Pipeline produces real predictions. **Never submitted to the leaderboard** |

### Numbers that are real (measured, not estimated)

**Label extraction vs the 58 gold studies:**

| Extractor | Macro AUC |
|---|---|
| Rule (regex, 11 languages) | 0.7565 |
| LLM (Qwen2.5-Instruct, open weights) | 0.7806 |
| **Mean of both** | **0.8234** |

The LLM alone (+0.024 on n=58) is **inside the noise band and is not a win**. The ensemble
(+0.067) is. Errors are complementary: LLM wins Effusion by **+0.178** (the label regex could not
fix); rule wins MCL by **+0.204** (LLM called 33 MCL positives against 9 actual).

**Imaging model, fold 0, 4 epochs, site-grouped CV:**

```
epoch 0  0.6818     epoch 2  0.7590
epoch 1  0.7304     epoch 3  0.7746   <- still improving, stopped early not converged
```

Per-label: Contusion 0.865 · Fracture 0.858 · Baker's 0.844 · Synovitis 0.836 · ACL 0.775 ·
MCL 0.771 · PF OA 0.754 · Lateral OA 0.744 · Effusion 0.738 · Medial OA 0.714 ·
Medial Meniscus 0.702 · Lateral Meniscus 0.695

**Floors any model must clear:** 0.500 constant · 0.595 series-composition · **0.598
DICOM-metadata-only under site-grouped folds**. Below ~0.60 the model is not reading images.

### The single most important insight

**Label noise, not model capacity, is the binding constraint.** Evidence: Synovitis scores
**0.836 from imaging** while its *text labels* score only **0.630** — the imaging model learns the
finding better than the labels teaching it. That is why the ensemble-label retrain (below) is the
highest-value pending work, ahead of any architecture change.

---

## 3. Environment, and the traps that cost real time

### Kaggle CLI

```bash
kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4
kaggle kernels status sibusisokhumalo11/<name>
kaggle kernels output sibusisokhumalo11/<name> -p out/
```

**Every one of these was hit for real. Do not rediscover them.**

1. **`--accelerator` is case-sensitive and silently ignores invalid values.** `nvidiaTeslaT4` is
   accepted without error and falls back to a **P100**, which Kaggle's PyTorch (sm_70+) *cannot
   run at all* — every CUDA op dies with `cudaErrorNoKernelImageForDevice`. Use
   **`NvidiaTeslaT4`**. Omitting the flag also gives a P100.
2. **Kaggle mounts inputs as a nested tree** — `/kaggle/input/competitions/<slug>/` and
   `/kaggle/input/datasets/<owner>/<slug>/` — *but not always*: `knee-model-v1` mounted **flat** at
   `/kaggle/input/knee-model-v1` in the same run. **Resolve every path by searching for marker
   files.** Hardcoding, or scoping the search to one subtree, silently breaks.
3. **Kaggle strips the top-level folder of an uploaded dataset.** A dataset built from `src/`
   arrives as the *contents* of `src/`. The notebooks rebuild the package under `/kaggle/working`.
4. **Kernel logs are sometimes JSON, sometimes plain text, and sometimes 0 bytes.** A JSON-only
   parser fails silently and looks like an empty run. Parse defensively.
5. **Kaggle returns logs only after a kernel completes** — there is no streaming API. For live
   progress you must watch the notebook in a browser.
6. **`.mcp.json` is read from the directory Claude Code was launched in**, not the repo you happen
   to be editing.
7. **`bitsandbytes` is not in the Kaggle image.** Install it, and keep a non-quantised fallback.

### Local machine

- Python 3.13, no GPU, ~180 GB free. **Never download the DICOMs** (569.76 GB / 819,640 files).
- Kaggle credentials: the new-style `KGAT_` token goes in **`~/.kaggle/access_token`** as raw
  text, *not* in `kaggle.json` (that is the legacy `username`+`key` scheme). `kaggle` is not on
  PATH in Git Bash — use `python -m kaggle`.

---

## 4. TWO KERNELS WERE STILL RUNNING AT HANDOVER — check these first

Both were `RUNNING` at 13:55 UTC on 2026-08-10 and their results are **not in this document**.
**Step one for whoever picks this up is to collect them.**

```bash
kaggle kernels status sibusisokhumalo11/knee-train-8ep
kaggle kernels status sibusisokhumalo11/knee-llm-labels
kaggle kernels output sibusisokhumalo11/<name> -p out/
```

| Kernel | What | Started | Watch for |
|---|---|---|---|
| `knee-train-8ep` | 8 epochs, fold 0, **rule-only** labels | ~05:30 UTC | Should beat 0.7746 — fold 0 was still improving at epoch 3. Checkpoint saves on every improvement, so **a timeout still leaves the best epoch** |
| `knee-llm-labels` v3 | LLM-labels all 4,407 studies → `labels_ensemble_v1.csv` | ~08:50 UTC | ⚠️ Writes output **only after all 4,407 finish** — a timeout loses the entire run |

Kaggle's kernel ceiling is **12 h**. `knee-train-8ep` was ~8.3 h in and past its ~7.8 h estimate
(queue time and validation passes were not in that figure).

**If `knee-llm-labels` timed out**, rebuild it with chunked writes and resume rather than retrying
the same all-or-nothing pattern. That exact failure mode already cost a local run 14/58 studies of
work, and repeating it here would cost ~6 GPU-hours.

**If it succeeded**, download `labels_ensemble_v1.csv`, publish it as a **private** Kaggle Dataset,
and go straight to next action #1 — `kaggle_02_train.py` already prefers it automatically.

### Also unfinished

- **Nothing has ever been submitted to the leaderboard.** Zero submission slots used.
- The Kaggle MCP server (`.mcp.json`, tokenless OAuth) was configured but **never loaded** — it
  needs a Claude Code restart from the directory containing `.mcp.json`, then its `authorize` tool.
- The LLM-vs-rule comparison used a model whose identity is **unrecorded** (that run's Kaggle log
  came back 0 bytes). The scores are real and were re-verified locally against gold; the model
  behind them is not known. `llm_run_info.txt` now prevents a repeat.

---

## 5. Next actions, in priority order

1. **Retrain on ensemble labels.** `notebooks/kaggle_02_train.py` already prefers
   `labels_ensemble_v1.csv` automatically and prints which file it used. Publish the ensemble CSV
   as a private Kaggle Dataset, attach it, run. This is the highest-value change: labels improved
   +0.067, and label noise is the binding constraint.
   *Temper expectations:* label quality does not translate 1:1 into imaging AUC — the model was
   already partly seeing through the noise.
2. **Submit.** Nothing has ever been submitted. This is a **code competition**: submission goes
   through the notebook's *Submit to Competition* button (or the Kaggle MCP upload tools), **not**
   by uploading a CSV. The `submission.csv` on disk holds only the 3 placeholder studies; the real
   ~1,300-study file is produced when Kaggle re-runs the kernel during scoring.
   A leaderboard score is the only way to learn whether our grouped CV translates to the hidden
   test set — genuinely worth a slot even on the weaker model.
3. **Train remaining folds** (1–4) once epoch count and labels are settled. 5 folds × ~4–8 h is
   most of a weekly quota, so don't spend it on a configuration still in flux.
4. **Efficiency track.** `Efficiency = (AUC_benchmark − max_AUC) + RuntimeSeconds/32400`,
   minimised. Inference measured at **~0.2 h against a 9 h cap** — enormous headroom, and the
   track is far less crowded than the main leaderboard. Verify the formula reading against the
   host's clarification thread first (flagged unresolved in
   [08-model-and-rules.md](08-model-and-rules.md)).
5. **Meniscus labels are the weakest imaging pair** (0.702 / 0.695). The domain primer predicted
   this: a tear spanning ~3 slices of 16 is the hardest thing in the dataset. More slices
   (`N_SLICES`) or higher resolution is the obvious lever, at a runtime cost.

---

## 6. Discipline that produced these numbers — please keep it

- **GroupKFold on the scanner fingerprint. Never random KFold.** Random folds overstate macro AUC
  by ~0.053 via site memorisation. `check_grouping()` is a tripwire that shouts if the key becomes
  near-unique — it exists because `ImagingFrequency` drift once produced **3,229 groups over 4,349
  studies**, silently reducing GroupKFold to random KFold while still looking rigorous.
- **The 58 gold studies are never trained on.** They are the only real ground truth. Training on
  them and then reporting agreement with them is circular.
- **n=58 is small. Treat ±0.03 as noise.** The LLM's +0.024 was deliberately *not* reported as a
  win. Per-label winner selection on 58 studies would be textbook overfitting; the ensemble uses a
  plain unweighted mean with zero fitted parameters, which is why it is trustworthy.
- **Diagnose by dumping actual state, not by guessing a second fix.** One diagnostic run found the
  inconsistent mount layout after two rounds of speculation had failed.
- **A silent degradation is worse than a crash.** The submission notebook once wrote a valid,
  scoreable, constant-0.5 file and exited 0. It now prints a loud banner when it falls back.
- **Record provenance.** Checkpoints carry `labels`/`epochs`/`n_groups`; the LLM run writes
  `llm_run_info.txt`. One run's log came back empty and left the model identity unknowable —
  the numbers were real but unreproducible.

---

## 7. Repo map

```
docs/00-state.md          living state — read first
docs/01-competition.md    rules, data spec, forum intel
docs/02-domain-primer.md  knee anatomy, MRI, all 12 findings
docs/03-data-guide.md     DICOM, CSVs, the 570 GB problem
docs/04-method.md         why every technical choice
docs/05-plan.md           phased schedule
docs/06-glossary.md       every term, plain language
docs/07-environment.md    setup + Kaggle CLI workflow
docs/08-model-and-rules.md model choice, licensing, rules position
docs/09-handover.md       this file

src/extract/    report -> labels (rule extractor, 11 languages + LLM)
src/data/       DICOM -> volumes  (shared by train AND inference)
src/model/      2.5D net, scanner fingerprint, GroupKFold, macro-AUC
notebooks/      kaggle_01_smoke / 02_train / 03_submit / 04_llm_labels
tests/          60 tests — `python -m pytest tests/ -q`
```

**Timeline:** entry deadline **2026-10-15**, final submission **2026-10-22**.
Note the Mintek Hackathon final falls **Oct 1–2**, ten days before the entry deadline.
