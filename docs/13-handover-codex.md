# 13 — Handover to Codex, as of 2026-09-06

Paste everything below the line into a fresh Codex session, or hand Codex this file directly.

---

You are picking up an RSNA Kaggle competition mid-flight from a Claude Code session. Your job:
**improve the leaderboard score.** Full context below; skim once, then work.

## Get the repo

**Local checkout already exists** on this machine at:
`C:\Users\lovilocal.adm\Desktop\RSNA-Knee-Abnormality-Detection`

If you are running on a different machine, clone it:
```
git clone https://github.com/Sibusiso-K/RSNA-Knee-Abnormality-Detection.git
```
Branch `main`, everything below is merged and pushed as of this handover. **Commit and push after
every session, on every branch** — another session has previously reconstructed hand-lost work
from a Kaggle dataset because unpushed git commits sat unseen on a branch. Don't repeat that.

- **Kaggle:** account `sibusisokhumalo11`, already authenticated on this machine via
  `~/.kaggle/access_token`. Use `python -m kaggle` — the bare `kaggle` binary is not on PATH.
- **Deadline:** final submission 2026-10-22.

## Use a real browser to check Kaggle, not just the CLI

**This is not optional — it caught a bug the CLI actively hid.** `kaggle competitions submissions`
returns an EMPTY `publicScore` for two completely different situations: (1) the submission is
still being scored, and (2) the submission **silently failed** ("Notebook Timeout" or similar) and
will never get a score. The CLI gives you no way to tell these apart — both just show a blank
field. The only way to know which one you're looking at is the Kaggle web UI's submissions page
(`https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/submissions`), which shows a
clear status badge (`check_circle Succeeded` vs `warning Notebook Timeout`) next to each entry.

This cost real time in the last session: a submission sat with a blank score for 19+ hours before
the web UI was checked and revealed it had failed almost immediately. **Check the web UI, with a
real Chrome browser (Claude in Chrome, or whatever equivalent browser tool you have), any time a
submission's score doesn't show up within an hour or so of pushing it** — don't just keep polling
the CLI and assuming "still scoring."

## Where the score actually stands

| Submission | LB score | What it was |
|---|---|---|
| **`knee-submit-6slice-24ep`** | **0.887** | **Best on record.** 5-fold DINOv2-small, xattn head, 6 slices/slot, EPOCHS 10→24 |
| `knee-submit-mix16-24ep` | 0.875 | 16-member blend: the 0.887 family + three OLD 10-epoch families (base, seed2, gattn). **Worse than the 0.887 family alone** — the stale members dragged it down |
| `knee-submit-pseudo-sel` | 0.865 | 5-fold selective-blend pseudo-labels (text-only on 4 contested targets). Essentially flat vs the 0.864 baseline |
| `knee-submit-ours16` | 0.864 | Old standing baseline: same 16-member idea, all at the original 10 epochs |
| `knee-submit-base-24ep` | **pending — relaunched on GPU**, see below | DINOv2-base, same 24-epoch fix. CPU version (v1) hit `Notebook Timeout`; v2 is on T4 GPU now |

**Read this right:** 0.887 is the number to beat, not 0.864. `mix16-24ep`'s result means **do not
blend a freshly-improved family with stale, unretrained members** — it measurably hurts. The
untested, promising next move is an ensemble built ONLY from freshly-retrained (24-epoch) members
across encoder sizes — see "What's queued" below.

## The one lever that has paid off twice: EPOCHS 10 → 24

Every model in this project was undertrained — checkpoints kept peaking at the last epoch of a
10-epoch schedule. Retraining at 24 epochs:

- Small encoder: CV rose only 0.8185 → 0.8261 (+0.0076, **below** the project's own +0.02
  "real gain" noise bar) — but LB rose 0.864 → **0.887** (+0.023). A real, large gain that the CV
  number badly undersold.
- Base encoder: CV rose 0.8120 → 0.8308 (+0.0188, just under the bar again). LB result pending
  (see above) — expect it to also outperform what CV suggests, based on the pattern.

**Lesson: do not use the CV noise bar to decide whether to submit.** It's a good filter against
illusory gains, but near the label-noise ceiling (labels cap out around 0.893 against real gold —
see `docs/00-state.md`) it systematically undersells real ones. Submit and read the real number.

## What's queued / should happen next

1. **Get `knee-submit-base-24ep`'s real score** (relaunched on GPU as v2 after the CPU timeout).
   Check the web UI, not just the CLI.
2. **Build and submit a from-scratch ensemble of ONLY 24-epoch-retrained members** — small (5
   folds, done, `knee-slot-6slice-24ep-v1` dataset) + base (5 folds, done,
   `knee-slot-base-24ep-v1` dataset) — with NO old 10-epoch members mixed in. `mix16-24ep`
   demonstrated that mixing stale members with a stronger family hurts; a same-generation blend of
   two encoder sizes might behave differently (cross-family diversity helped at the 10-epoch level
   too — see `knee-submit-slots` v5 = 0.856 from a 6-member cross-family blend). This is the most
   promising untested move right now.
3. **Frontier-LLM label extraction** (see `docs/08-model-and-rules.md`, `src/extract/api.py`):
   scaffolding is built and tested, but needs an `ANTHROPIC_API_KEY` and the user's explicit
   go-ahead before spending real money (even a few dollars). Validate on the 58 gold studies
   first (`python scripts/extract_labels_llm.py --api-evaluate --confirm-spend`) before scaling to
   the full 4,407-report corpus. This is the largest unexploited lever in the project
   (label ceiling is ~0.893 vs the current pipeline's ~0.82 CV) but is a real spend decision.
4. **gattn at 24 epochs** (`kaggle/train-gattn-32ep`) was built but never actually completed a run
   before this handover — worth finishing if there's TPU budget left this week.

## Operational facts, all learned the hard way (again)

- **Kaggle session time caps: TPU 9h, GPU 12h.** A run that exceeds this is hard-killed
  (`exit 137`/SIGKILL) and its log does NOT survive — `kaggle kernels output` returns a genuinely
  empty log file even though the run produced real output. Use `torch.load(path)['score']` on the
  checkpoint (and the `progress_fold{N}.txt` marker file, see below) to find out what actually
  happened, and check the web UI's Logs tab (it shows the cancellation reason even when the CLI's
  downloaded log is empty).
- **One concurrent TPU session.** A second `kernels push` for a TPU kernel while one is already
  running/queued is flatly REJECTED ("Maximum batch TPU session count of 1 reached"), not queued.
  Wait for the first to finish before pushing another.
- **Push GPU kernels with `--accelerator NvidiaTeslaT4` explicitly**, or via
  `scripts/launch_kaggle.sh` which does this for you. Without it, Kaggle may hand back a P100
  (confirmed 2026-08-31) or leave a kernel CPU-only, and this PyTorch build only supports sm_70+
  — P100 is sm_60 and every CUDA op fails immediately.
- **CPU inference times out for DINOv2-base members, but is fine for DINOv2-small.** Confirmed
  twice now (`knee-submit-10` historically, `knee-submit-base-24ep` v1 this session). Any
  submission kernel with base-encoder members needs `enable_gpu: true`.
- **A stall watchdog now exists** in both `notebooks/kaggle_06_train_slots.py` (the shared script
  behind every `train-*-Nep` kernel) and the standalone `notebooks/kaggle/knee-train-pseudo-sel/`
  script: a background thread aborts the run if no training step completes within
  `STEP_TIMEOUT_S` (default 120s) instead of silently burning the full session cap. This caught
  several genuine stalls this session — some at fold-boundary DINOv2 weight reloads, seemingly a
  Kaggle infra flakiness rather than a code bug. **When it fires, retry just the missing fold(s)**
  (`FOLDS` env var / the sed patch in the relevant `kaggle/<name>/build.sh`) rather than the whole
  run — this repeatedly worked cleanly (see `train-6slice-base-24ep-a` through `-e`, each a
  smaller fold subset of the same family).
- A cheap `progress_fold{N}.txt` marker, overwritten every epoch, is now written alongside the
  ~90-350 MB checkpoint — check it first when diagnosing a run that didn't finish; it's much
  faster to pull than the full checkpoint.
- **The Kaggle TPU image cannot decode DICOM.** Train on TPU, submit on GPU or CPU (CPU only for
  small-encoder members, see above).
- `kaggle datasets version` / `datasets create` are **asynchronous** — wait for `datasets status`
  to report ready (or just re-download and verify) before a kernel that depends on it launches,
  or it may mount stale data.
- **The public test set is ~3 studies**; the real ~1,300 come at the private re-run. A fast, tiny
  `submission.csv` is normal, not a symptom.

## How to work

Evidence first. State a threshold before a run — but per the lesson above, **don't skip a cheap LB
test just because a CV number sits inside the noise band near the label ceiling.** Change one
variable at a time. Report nulls plainly. When a result looks surprising, check the inputs before
believing it — five times out of five in this project's history, a surprising number meant a
broken input rather than a discovery.

0.887 against a 0.936-ish top-10 cutoff (check the current leaderboard — it has been climbing) is
real progress and still a meaningful gap. Say so honestly rather than declaring victory early.
