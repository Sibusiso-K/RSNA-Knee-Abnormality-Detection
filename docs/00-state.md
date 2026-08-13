# 00 — Project state (READ THIS FIRST)

> **This is the living file.** It is the single source of truth for *where the project is right now*.
> Every other doc explains something stable; this one changes constantly.
> **Update it at the end of every working session.** If it's stale, everything else is a trap.

**Last updated:** 2026-08-11, 21:30 UTC (session 15 — measured the field, rebuilt the input)

> ## Session 15: we are 713th of 1,185, and below a notebook anyone can fork
>
> **Measured, not estimated** (Kaggle API, 2026-08-11):
>
> | | |
> |---|---|
> | Us | **0.783 — rank 713 / 1,185** |
> | Top 10 cutoff | **0.930** |
> | Rank 100 | 0.901 · Rank 50 0.912 · Rank 1 0.943 |
> | Teams ≥ 0.80 | **693** |
> | Free public notebook | **0.899** (`aadigupta7686/0-899-let-me-cook`, a fork of the 280-vote `pilkwang/rsna-knee-baseline-v1`) |
>
> **The gap is not mysterious and it is not capacity.** Reading our own source against
> the four preprocessing defects published by a top team (discussion 734105 §6), we have
> two of them outright, plus three more of our own. Every one is fixed in code as of this
> session; none has been trained yet.
>
> | Defect | Us before | Now |
> |---|---|---|
> | **Laterality normalisation** | ❌ absent entirely | ✅ from image centre in patient space |
> | **Effective resolution** | ❌ 160 mm @ 224 px = 0.714 mm/px | ✅ 130 mm @ 336 px = 0.387 mm/px |
> | **Sequence routing** | ❌ 3 planes, fluid-only, `iloc[0]` | ✅ 6 slots, weighting from DICOM headers |
> | **Vertical-flip augmentation** | ❌ destroyed where findings sit | ✅ removed (rigid jitter only) |
> | **DICOM decode per epoch** | ❌ ~3,600 s/epoch, ~1.3 M decodes/run | ✅ cached once on free CPU |
> | Slice ordering by geometry | ✅ already correct | unchanged |
> | 2.5D triplet into encoder | ✅ already correct | unchanged |
>
> **Laterality is the big one.** Five of twelve targets are side-specific (Medial/Lateral
> Meniscus, Medial/Lateral OA, MCL). Left and right knees are mirror images, so "medial"
> was landing on opposite sides of the frame at random across **42% of the macro metric**.
> Measured on 40 studies: side resolves for **100%** of them (L=21, R=19, none unknown).
>
> **Resolution explains the dead 160 mm experiment.** At 0.714 mm/px a 1–3 mm meniscal
> tear is 1–4 pixels. The crop was the right idea at the wrong scale — which is why it
> returned 0.002 rather than nothing at all.
>
> ⛔ **GPU quota exhausted: 34:15 / 30 hrs.** Resets weekly.
> ✅ **Kaggle TPU untouched: 00:00 / 20 hrs.**
> ✅ **DICOM decoding is CPU work and CPU notebooks do not draw on the GPU quota.**
> That is the unblock: `kaggle_05_cache.py` builds the whole input on free CPU.
>
> **The cache is the strategic move, not just an optimisation.** `kaggle_02_train.py`
> decoded every DICOM inside `__getitem__`, so ~90% of each GPU hour went to decoding
> while the GPU idled. Decoding once converts the 30 h/week from ~1/8 useful to nearly
> all useful — roughly **8x more effective training throughput on the same free-tier
> allowance**. It also makes training portable: the cache is ~9 GB, so it runs on
> Lightning or anywhere else, while the 570 GB of DICOMs never leave Kaggle.
>
> **Labels: ours are superseded, and that closes a blocked workstream.**
> `scripts/compare_labels.py` re-scores every candidate on the same 58 gold studies:
>
> | Source | macro vs gold |
> |---|---|
> | `stevenleehans/llm_labels_v4_blend` | **0.8927** |
> | cross-author rank-mean blend (steven + pilkwang + sol56) | **0.8941** |
> | `pilkwang/report_labels_v2` | 0.8700 |
> | **ours, `labels_v1.csv` (rule-only)** | **0.7565** |
> | ours + the public sets blended in | 0.8849 — *worse*, ours drags it down |
>
> ⚠️ On 58 studies anything under ~0.02 is unresolvable, so 0.8941 and 0.8927 are the
> same number. The honest reading is "the public sets are far better than ours, and
> which of them is best is undecided".
>
> **Consequence: stop work on LLM shard 2.** It was blocked on GPU quota and aimed at
> `labels_ensemble_v1.csv` (projected ~0.8234). The public sets already beat that
> projection by ~0.07 without spending an hour of quota. That workstream is closed.
>
> ⚠️ **Two guards worth keeping.** `compare_labels.py` refuses candidates that score a
> perfect 1.0000: `barun2104`'s fold files carry `train.csv`'s own label columns —
> 696/696 cells identical on gold, NaN everywhere else. Ranking a globbed directory by
> score would have crowned the answer key and let it define the label strategy. And the
> cache build now audits geometry-derived laterality against the `Laterality` tag,
> because a mirror applied to the wrong half of the corpus is strictly worse than none.
>
> ### ✅ `knee-cache` built — 4,407 studies, 8.96 GB, 1h58m, zero GPU quota
>
> Measured on the **full corpus**, not a sample:
>
> | | |
> |---|---|
> | Studies cached | 4,407 / 4,407 |
> | Laterality unresolved | **13 (0.3%)** |
> | Tag vs geometry agreement | **11,503 / 11,715 (98.2%)** |
> | Series geometry resolves that have **no** tag | **12,229** |
> | Physical crop applied | 19,710 / 19,751 (**100%**) |
> | Slot fill | 0.747 overall |
>
> Per slot: SAG_FLUID_FS 0.932 · COR_FLUID_FS 0.956 · AX_FLUID_FS 0.904 ·
> SAG_FLUID_NOFS 0.579 · COR_T1 0.637 · SAG_T1 0.475.
>
> **L=2,067 R=2,327 — 47% of the corpus was being fed to the encoder mirrored.** And
> geometry resolves *more* series (12,229) than carry a `Laterality` tag at all (11,909),
> so it more than doubled coverage rather than merely reproducing the tag. Where the two
> disagree (212 series, 1.8%) the tag wins, which is the safe default.
>
> The three structural slots carry 47–64% coverage. Those series existed all along and
> `pick_series` discarded every one of them.
>
> ### ✅ Cache audited on free CPU — every check green
>
> | Check | Result |
> |---|---|
> | Studies with zero usable slots | **0** |
> | Slots per study | mean 4.48, min **2**, max 6 |
> | Scanner fingerprint groups | **152 / 4,407 (ratio 0.034)**, 42 singletons |
> | Largest groups | 353, 258, 237, 217, 214 — top-20 cover 65.1% |
> | Gold studies present | **58 / 58** |
> | Label merge (`labels_blend_v1`) | 4,406 / 4,407 |
> | Training pool after gold holdout | **4,349** |
> | Fold sizes | 870/870/870/870/869 — ratio **1.00** |
>
> **152 groups, not 3,229.** That is the check worth having run: reusing `_clean_tag`'s
> whole-MHz rounding of `ImagingFrequency` kept the grouping coarse. A near-unique
> fingerprint degrades GroupKFold to random KFold and returns the 0.053 of site leakage
> as free-looking score, and it would be invisible — the run completes and prints a
> better number. 4,349 also matches the historical grouped-CV pool exactly.
>
> ### Training is written, dry-run, and blocked only on quota
>
> The whole chain was exercised locally on CPU against the 40-study smoke cache with
> DINOv2-small: cache → label alignment → grouped folds → train → checkpoint → reload
> `strict=True` → forward with a partially masked study. 11.0M trainable parameters.
> **The AUC from that run is meaningless and is not recorded** — 11 validation studies,
> one epoch. It proves the plumbing, nothing else.
>
> ⛔ `kaggle kernels push` returns **"Maximum weekly GPU quota of 30.00 hours reached"**
> and creates no kernel version, so retrying the push is a clean, spam-free way to poll
> for the reset. Reset timing is genuinely unclear — public sources disagree between
> Saturday and Sunday 00:00 UTC, Kaggle has a "floating" quota feature, and the settings
> tooltip does not say. Do not write a date here until one is observed.
>
> ⛔ **Lightning AI is not an option on the free tier: 0 credits** across org, unallocated
> and total, with no personal account holding a separate allowance. Confirmed 2026-08-12.
>
> **Not yet known:** whether any of this moves the leaderboard. Nothing has been trained
> on the new representation. The numbers above are input-quality measurements, and this
> file has three times recorded a label improvement that did not transfer.
>
> ### ✅ Run 1 (TPU): CV 0.7948 / gold58 0.8105 — and 20 epochs does NOT help
>
> | Schedule | best CV | gold58 | wall |
> |---|---|---|---|
> | 10 epochs | **0.7948** | **0.8105** | 30 min |
> | 20 epochs | 0.7964 | 0.8090 | 57 min |
>
> +0.0016 for double the training: inside noise. The 20-epoch curve plateaus at
> epoch ~5 (0.7885) and oscillates 0.786–0.793 for fourteen more epochs while loss
> falls 0.550 → 0.436. **Same overfitting signature the old pipeline had.** Training
> length is saturated and is not the binding constraint. Do not revisit.
>
> Per-label at 10 epochs: Baker's 0.845 · Medial OA 0.832 · Effusion 0.831 ·
> Synovitis 0.825 · Fracture 0.815 · Contusion 0.813 · Medial Meniscus 0.812 ·
> ACL 0.782 · Lateral OA 0.771 · Lateral Meniscus 0.754 · MCL 0.747 · **PF OA 0.711**.
>
> ⚠️ **0.7948 is NOT comparable to the old 0.7746.** Different labels (blend 0.8930 vs
> rule-only 0.7565) and a different validation target (undecided cells dropped).
>
> ⚠️ **Do not reuse the +0.008 CV→LB offset.** It was measured on the old pipeline with
> old labels against a different target. What CV 0.7948 maps to on the leaderboard is
> genuinely unknown until a submission lands. **`gold58` is the better LB predictor** —
> it is scored against real annotations, as the leaderboard is — but on 58 studies it
> carries ±0.05 easily.
>
> ### ✅ LB 0.850 — and the CV→LB offset is +0.055, not +0.008
>
> Submission `55475708`, 5-fold rank-mean, 2026-08-13.
>
> | | CV | LB | offset |
> |---|---|---|---|
> | old pipeline | 0.7746 | 0.783 | +0.008 |
> | **new pipeline** | **0.7949** | **0.850** | **+0.055** |
>
> **Rank 627 / 1341**, up from 713 / 1185. The field moved too: top-10 cutoff is now
> **0.935** (was 0.930), rank 100 = 0.907, rank 200 = 0.899.
>
> The offset changed because the CV *metric* changed — blend labels, undecided cells
> dropped — so the two CV columns are not the same measurement. **Site-grouped CV is
> substantially harsher than this leaderboard.** Use +0.055 as the working anchor at
> this operating point, and re-measure it rather than extrapolating far from it.
>
> **Still +0.085 short of top 10.** The rebuild bought the input, not the field.
>
> ⚠️ **More slices per slot is NOT the gap.** The public 0.899 runs `N_GROUP_MAX = 1`,
> so it gets no inference-time group averaging either. A cache rebuild for more slices
> would buy a difference that recipe does not contain. Checked before spending ~4 h on it.
>
> ⚠️ **Detection floor: ±0.011 on one fold** (measured across the 5-fold spread
> 0.7817–0.8041). Any single-fold experiment worth less than ~0.02 is unmeasurable
> without repeating folds.
>
> ### Four levers, four nothings — the model is saturated on this representation
>
> | Lever | fold-0 CV | vs 0.7990 | Verdict |
> |---|---|---|---|
> | 8 vs 4 epochs (old pipeline) | — | — | nothing |
> | 20 vs 10 epochs | 0.7964 | +0.0016 | noise |
> | DINOv2-base (43.1M vs 11.0M) | 0.7985 | −0.0005 | nothing |
> | **448 px (0.290 mm/px vs 0.387)** | **0.7976** | **−0.0014** | **nothing** |
> | 2-model rank-mean | 0.8077 | +0.0086 | real, under the 0.010 bar |
>
> The 448 run was valid, not a silent no-op: the log shows `(4407, 6, 3, 448, 448)`
> and step time rose 0.35 s → 0.56 s, matching 1025 tokens against 577.
>
> **Neither capacity nor pixel density moves this.** Everything sits at ~0.798 CV /
> ~0.82 gold58. Do not spend further runs on either axis.
>
> ⚠️ **A retracted inference.** This file previously argued the model "does not reach
> its teacher" (0.82 vs labels' 0.8930) and concluded information is lost in the
> representation. That does not follow: a perfect *image* model need not match
> *report-derived* labels, because reports carry context no pixel contains. The sound
> version of the argument is external — **teams score 0.935 on the same hidden test
> set**, so the headroom is real even though this particular diagnostic did not
> establish where it lives.
>
> ### Three capacity levers, three nothings — and where the loss actually is
>
> | Lever | Result | Verdict |
> |---|---|---|
> | 8 vs 4 epochs (old pipeline) | — | nothing |
> | 20 vs 10 epochs | +0.0016 | noise |
> | **DINOv2-base vs small** (43.1M vs 11.0M trainable) | **−0.0005** | **nothing** |
>
> **Capacity and training budget are not the constraint.** Stop proposing experiments
> that assume they are.
>
> **The diagnostic that should drive everything from here:** the labels score **0.8930**
> against the 58 gold studies; the model scores **~0.82** against the same studies.
> The model does not reach its own teacher, so label noise is not the binding ceiling
> either — roughly 0.07 of headroom sits unused. What remains is the **image
> representation**: resolution, slice coverage, or how the slots are formed.
>
> ### Diversity pays a little, but not enough to buy
>
> Measured on fold 0 (`knee-diversity`), both members on the same validation split:
>
> | | fold-0 CV |
> |---|---|
> | base alone | 0.7988 |
> | small alone | 0.7992 |
> | rank-mean of both | **0.8077 (+0.0086)** |
> | per-label Spearman | mean 0.853, min 0.751, max 0.925 |
>
> Real but below the **pre-committed 0.010 bar**, so base × 5 folds was NOT bought —
> it would have cost ~3.5 h of a 20 h budget to average harder over the same
> bottleneck. **base fold 0 already exists and joins the final ensemble free.**
>
> The pre-registration is the point: the threshold was written into the script before
> the run, and honouring it at +0.0086 is what makes the next one worth anything.
>
> ### ⚠️ Operational limits, all measured the hard way
>
> | Limit | Consequence |
> |---|---|
> | **Kaggle TPU image cannot decode our DICOMs** | Every study fails; submission becomes a constant 0.5 |
> | **Only ONE concurrent TPU session** | Training and a TPU submission cannot overlap |
> | TPU billing is wall-clock, incl. the 9 GB mount | Run all folds in ONE kernel, not five |
> | `find_dir` over `train_series/` costs ~1,100 s/call | `SKIP_DIRS` is not optional |
> | GPU/CPU image DOES decode | Use GPU for submissions once quota refreshes |
>
> **The TPU decode failure is the important one.** The same code decodes fine on the
> CPU image (L=1 R=2, crop 11/11) and fails on all 3 studies on TPU. Submissions run
> with internet OFF, so nothing can be pip-installed to fix it at scoring time.
> **Submit on GPU or CPU, never TPU.** Train on TPU.
>
> It was caught only by the constant-submission tripwire in `kaggle_07_submit_slots.py`
> — six members loaded correctly, the rank-mean ran, and it wrote an all-0.5 file that
> would have scored ~0.5 and burned a slot. Keep that tripwire.
>
> ### Cost model on TPU (measured)
>
> | | |
> |---|---|
> | Startup (mount + load 8.34 GB + folds) | ~98 s |
> | Step (batch 8 = 48 slot images) | ~0.35 s |
> | Epoch (434 steps + validation) | ~160 s |
> | 10-epoch fold | **~30 min** |
>
> A fold costs ~2.5% of the weekly TPU budget, so experiments are cheap now. The
> binding cost is no longer compute.
>
> ### Experiment queue, in priority order
>
> **Run 1 is a measurement, not an attempt to win.** Its job is to produce one honest
> grouped-CV number on the new representation so everything after it has a baseline to
> be compared against. Do not stack changes onto it.
>
> | # | Change | Cost | Why it is where it is |
> |---|---|---|---|
> | 1 | Fold 0, DINOv2-small, 10 ep | 1 fold | The baseline. Nothing else is interpretable without it. |
> | 2 | 5-fold ensemble | 5 folds | The public 0.899 is a *single* model. Folds are the most reliable gain in the list. |
> | 3 | Label-correlation post-processing | **free** | Macro AUC is per-label and rank-based, so no calibration is needed — but predicted Effusion may rank Synovitis better than predicted Synovitis does (0.7115 vs 0.6780 *on labels*). Pure post-processing, testable on existing predictions. |
> | 4 | Cache v2: more slices/slot (N_GROUP 2–3) | **free CPU** + folds | The public baseline is stuck at 3 slices by Kaggle RAM. We are not: the cache is built offline. Buys stack augmentation in training and averaging at inference. |
> | 5 | DINOv2-base | folds | 2x feature width, ~4x activation memory. May need BATCH 4. |
> | 6 | Higher resolution (448 = 14x32) | **free CPU** + folds | 0.29 mm/px. Only after 4 shows more input helps. |
>
> **Do not reorder 3 and 4 ahead of 2.** Both are attractive because they are cheap, and
> both change the input or the output rather than the amount of training — which is
> exactly the class of change this project has repeatedly measured at ~0.002.
>
> **A note on calibration, so nobody spends time on it:** the metric is macro AUC over
> twelve *independent* per-label AUCs. AUC reads order only. So per-label calibration,
> temperature scaling and threshold tuning are all worth exactly zero here. Only the
> within-label ranking matters.

**Days to final submission (2026-10-22):** ~72

---

## Where we are right now

| Phase | Status |
|---|---|
| 0 — Access | ✅ Done |
| 1 — Labels from reports | ✅ **Ensemble (rule + LLM) 0.8234** vs gold, up from 0.757 rule-only |
| 2 — Site-grouped CV | ✅ Done and **verified honest** (151 groups / 4,349 studies) |
| 3 — Imaging model | ✅ **Fold 0 trained: 0.7746 grouped-CV macro AUC** |
| 4 — Submission | ✅ **Submitted 2026-08-10: public LB 0.783** (CV 0.7746 — CV transfers) |

**Trained model exists and works.** `knee-model-v1` (fold 0, EfficientNetV2-S 2.5D +
attention-MIL) scores **0.7746** under honest site-grouped CV — well clear of the 0.598
metadata-only floor, so it is reading images, not memorising scanners.

### `knee-train-8ep` landed: 8 epochs is **not** better than 4

Completed 2026-08-10, 29,974 s (8.3 h) on T4×2, ~3,600 s/epoch.

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| AUC | 0.6707 | 0.7249 | 0.7487 | 0.7388 | 0.7600 | 0.7642 | **0.7697** | 0.7664 |
| loss | 1.0620 | 1.0168 | 0.9878 | 0.8940 | 0.7646 | 0.6231 | 0.4880 | 0.4135 |

**Best 0.7697 (epoch 6) vs the 4-epoch run's 0.7746 — doubling the epochs did not help.** The −0.005
is within noise; the honest statement is "no improvement", not "a regression".

**Do not compare these two runs epoch-by-epoch.** `OneCycleLR` sets
`total_steps = EPOCHS * steps_per_epoch`, so the LR schedule stretches to whatever `EPOCHS` is. At
epoch 3 the 4-epoch run had fully annealed (0.7746 = converged) while this run was still mid-
schedule (0.7388). Only the end-of-schedule numbers are comparable. The handover's expectation —
"should beat 0.7746, fold 0 was still improving at epoch 3" — was based on exactly that invalid
comparison.

**The last two epochs are an overfitting signature:** loss fell 1.06 → 0.41 (−61%) while AUC
plateaued and then *declined* at epoch 7, with LR annealing to zero. More epochs are memorising the
training set, not learning the findings. **This is direct evidence for the central thesis: capacity
and training budget are not the binding constraint — label noise is.** Do not extend to 12+ epochs;
spend the GPU hours on ensemble labels instead.

⚠️ The checkpoint carries only `backbone` / `fold` / `score` — **not** the `labels` / `epochs` /
`n_groups` provenance the handover describes. That change (a1c6fd7) landed after this kernel was
pushed, so `knee-src` predates it. We know from the run config it was rule-only `labels_v1`, 8
epochs, but the file itself does not say so. Provenance is not retroactive.

### ✅ RESOLVED: our CV transfers, and the gap to the field is real

**First submission, 2026-08-10: public LB `0.783`** (submission `55411501`, fold-0 `knee-model-v1`,
rule-only labels). Site-grouped CV for the same checkpoint was **0.7746**.

### ⚠️ Two calibration points now, and they disagree — do not trust a fixed offset

| Submission | CV | LB | offset |
|---|---|---|---|
| v1 fold 0, 16/224 | 0.7746 | **0.783** | +0.0084 |
| v2 fold 0 + 160 mm crop | 0.7767 | **0.781** | +0.0043 |

**CV went up (+0.0021) while LB went down (−0.002).** The offset is not a constant, and the
ordering does not even survive: the model with the better CV scored worse on the leaderboard.

What this does and does not license:

- ✅ **CV predicts LB at coarse resolution.** Both land at ~0.78. Grouped CV is not lying to us,
  and it is still the right instrument for deciding whether something works.
- ❌ **CV cannot resolve differences of ~0.005 or less.** Neither can a single LB score. Anything
  in that band is noise in *both* instruments, so "CV improved by 0.002" means nothing.
- ❌ **Do not convert CV to an expected LB rank using a fixed offset.** An earlier version of this
  file did exactly that off n=1; with n=2 the offsets differ by a factor of two.

**Planning consequence:** only pursue changes plausibly worth **≫0.01**. At 30 h GPU/week, a 4-epoch
run costs ~1/8 of the budget, and a change worth 0.002 is indistinguishable from doing nothing —
which is precisely what the 160 mm crop turned out to be. It is now **confirmed dead**: better CV,
worse LB, no effect.

**The LB came in +0.008 above CV** on the first submission. That settled a question this file
previously listed as open.
The earlier reasoning went: our CV might badly *understate* us, because (a) it scores against noisy
report-derived labels, and (b) it holds out entire scanner sites while the hidden test may not.
Either effect would have put the LB well above 0.7746. **Neither did. That hypothesis is dead.**

Two consequences, and the second is unwelcome:

1. **Site-grouped CV predicts the leaderboard almost exactly.** We can iterate against CV and trust
   it, without spending submission slots to find out. This is worth a lot — 5 slots/day, and the
   grouping discipline in §"Discipline" is what earned it.
2. **We are genuinely ~0.16 behind.** Top 0.942, 20th 0.919, us 0.783 (1,018 teams). This is a
   capability deficit, not a measurement artifact. Do not soften this.

Read alongside the `knee-train-8ep` result above — more epochs bought nothing and began overfitting
— both findings point away from the training loop and at the labels. That is the project's existing
thesis, now supported by evidence rather than inference.

⚠️ **But calibrate expectations about labels too.** Ensemble labels are +0.067 *on label quality vs
gold*, and label quality does not translate 1:1 into imaging AUC (the model already partly sees
through the noise). **Better labels alone will not close 0.16.** Something beyond the current
recipe — more slices/resolution, multi-fold ensembling, a stronger backbone, or an approach we have
not tried — is needed to be competitive. Treat the ensemble retrain as the next step, not the plan.

Entry deadline 2026-10-15 is the real clock. Note only **2 submissions count** toward the final
leaderboard score (Kaggle auto-selects the best if fewer are chosen).

**Best available labels are now the ENSEMBLE at 0.8234** (rule + LLM mean), not the 0.757 rule-only
`labels_v1.csv` that `knee-model-v1` was trained on. Retraining against ensemble labels is the
highest-value pending change — session 11 showed label noise, not capacity, is the binding
constraint.

### What runs today

```bash
python scripts/extract_labels.py --evaluate        # rule extractor vs the 58 gold studies
python -m pytest tests/ -q                          # 60 tests
kaggle kernels push -p . --accelerator NvidiaTeslaT4
```

## Older next-actions (Phase 1 text extraction — superseded by the imaging work above)

1. **Effusion and Synovitis were chased in session 7 — no safe fix found, and that's the right
   outcome, not a stall.** Verified one Effusion false positive directly against the raw CSV: the
   report's own Impression says "Mild synovitis of right knee joint" while gold Synovitis=0 — a
   genuine report/label mismatch, independently confirming the forum thread on this. A "not
   repeated in Impression" heuristic looked promising on a few examples but breaks down because
   some languages in this corpus (Spanish reports especially) mechanically duplicate Findings into
   Impression regardless of significance — the signal isn't reliable enough to encode as a rule.
   For Synovitis: only 3/27 gold positives even have a thickening/hypertrophy fallback word: most
   gold-positive reports name no synovitis-related term in any language. This is the label the
   domain primer predicted would be hardest to extract by keyword alone (usually diagnosed with
   contrast these studies lack) — it needs the LLM extractor, not more regex. **Move to Phase 1
   step 2 rather than continuing to chase these two by pattern.**
2. **Consider whether Fracture and Contusion still share the marrow-edema ambiguity noted in
   session 4.** Not yet resolved — needs the host's detailed label-criteria post, not more
   pattern-guessing on 2-3 examples.
3. **LLM extractor (Phase 1 step 2) is now under construction** — `src/extract/llm.py` +
   `scripts/extract_labels_llm.py`, built session 8. Runs against a local Ollama server, open
   weights only, so report text never leaves the machine. Same `StudyExtraction`/`extract_frame`
   interface as `RuleExtractor`, so it plugs into the existing `evaluate.py` harness directly.
   - `--demo` (10 synthetic reports, no data needed) already ran against both locally-available
     models: **`llama3.2` hallucinates badly** — invented meniscus tears and synovitis findings
     with zero textual basis, on 5 of 10 cases. **`qwen2.5-coder:7b` is much cleaner** (7/10 clean)
     but still has real negation misses: scored "No joint effusion" as 0.4 instead of 0 once, and
     missed German "Kein Gelenkerguss" (no effusion) entirely, scoring Effusion=1.0. The rule
     extractor gets both of those right via explicit negation detection — a genuine data point,
     not a foregone conclusion that LLM > regex.
   - **First `--compare` attempt stalled for 50+ min with zero output** — not a fundamental
     problem, a real diagnostic gap: `extract_frame` printed nothing per-study, and Python fully
     buffers stdout when redirected to a file, so a genuine hang and normal-but-slow progress were
     indistinguishable from outside. Fixed: `extract_frame` now prints a `flush=True` line per
     study (id, char count, elapsed time); script default timeout dropped 180s→90s; added
     `--limit` for smoke-testing a small fast subset before a full run.
   - **Verified the fix works**: a 6-study smoke test ran cleanly, ~40s/study, no stalls. LLM
     scored 0.855 vs. the rule extractor's 0.765 **on that same tiny 6-study slice** — promising
     but nowhere near a real number (n=6).
   - **Second attempt (full 58 studies) confirmed the fix works** — reached 14/58 with clean,
     visible per-study timing (60-130s/study, no hang) before being stopped intentionally to end
     the session. Rule-extractor half of that run reconfirmed **0.757** exactly, as expected.
     **The LLM-vs-rule number on the full 58 studies still does not exist — nothing partial was
     saved (the script only writes output after all 58 complete), so this is a full rerun next
     session, not a resume.** At the observed pace (~75s/study average), a full run is
     ~50-70 minutes on this machine's CPU. Don't report an LLM macro AUC until this actually
     finishes — the only real numbers so far are the 6-study slice (0.855, too small to trust) and
     the qualitative demo findings above (cleaner than llama3.2, but still misses negation the
     rule extractor catches).
   - Explicitly staying on local CPU (Ollama) for now per user direction — only move to a Kaggle
     GPU notebook if local hardware genuinely blocks the work, and say so first before switching.
4. Worth a quick pass: check whether other negation words share the "izlenme" false-friend
   structure in other languages (a wildcarded root that's also a substring of the positive form).
   Found three Turkish instances in one sitting; unclear if it's isolated to Turkish.

## Session 9: Phases 2-3 built, ready for Kaggle

**RULES CHANGE — commercial LLMs are now permitted** (host announcement, 2026-08-08). This
reverses the assumption Phases 1's sessions 4-7 were built on. Full detail and what it changes:
[08-model-and-rules.md](08-model-and-rules.md). Practical effect: label extraction can now use a
hosted LLM, but **open-weights-on-Kaggle stays the preferred route** — free, offline, and it
sidesteps the "minimal cost / reasonably accessible" test the host reserved the right to judge
after the fact.

Built and unit-tested this session (50 tests passing, up from 40):

| File | What it is |
|---|---|
| `src/data/dicom.py` | DICOM → normalized (3, S, H, W) volumes. **Shared by train and inference on purpose** — preprocessing skew is how you score well in CV and badly on the LB |
| `src/model/net.py` | 2.5D EfficientNetV2-S + gated attention-MIL, 12-way multi-label head |
| `src/model/validation.py` | Scanner fingerprinting + GroupKFold + macro-AUC |
| `notebooks/kaggle_01_smoke.py` | 15-min pre-flight: transfer-syntax decode, load timing, plane coverage, one GPU step |
| `notebooks/kaggle_02_train.py` | Site-grouped CV training, gold studies held out |
| `notebooks/kaggle_03_submit.py` | Inference → `submission.csv`, degrades to 0.5 rather than crashing |

`data/labels_v1.csv` — all **4,407** studies labelled by the rule extractor. Prevalences pass the
clinical sanity check (ACL 13.6%, Fracture 6.9%, Effusion 54.9%, Medial Meniscus 40.0%).

**Verified locally, not assumed**: 2.5D triplet construction (correct shape, edges replicate rather
than wrap, centre channel preserved), attention pooling normalisation, `pos_weight` favouring rare
labels while staying capped, full forward pass (2,3,16,64,64)→(2,12), **GroupKFold splitting no
group across folds**, and macro-AUC skipping single-class labels.

### Session 10: Kaggle live — smoke passed, fold 0 training

**Everything runs on Kaggle now.** Datasets `knee-src` + `knee-labels` (both **private** —
`labels_v1.csv` is derived from competition report text, so publishing it would be prohibited
sharing of competition-derived work product). Kernels: `knee-smoke`, `knee-train`, `knee-submit`.

Driven entirely through the **Kaggle CLI**, not browser automation: `kernels push` is
reproducible and version-controlled, and file-picker automation is where these setups break.
(The Chrome profile on this machine is not signed in to Kaggle, so the UI route wasn't available
anyway.)

#### Smoke test results (T4) — all green

| Check | Result |
|---|---|
| Path resolution | src + competition both found |
| Data | 4,407 studies / 24,371 series |
| DICOM decode | Explicit VR Little Endian OK |
| Load speed | **1.3 s/study** → epoch ~0.8 h, inference ~0.2 h vs the **9 h cap** |
| Plane coverage | 3/3 planes on 8/8 studies |
| Volume sanity | per-plane std 0.18–0.32 (real signal, not blanks) |
| GPU fwd+bwd | logits (2,12), **peak 7.7 GB of ~15 GB** |

**Consequence: no `.npy` preprocessing cache needed.** An epoch is under an hour, so the caching
stage in [03-data-guide.md](03-data-guide.md) step 6 is unnecessary — skip it.

#### Five bugs the pre-flight caught (none reached a real training run)

1. **`.gitignore` ate `src/data/`.** `data/` with no leading slash matches a `data` dir at *any*
   depth, so `dicom.py` — imported by every notebook — was never committed. Anchored to `/data/`.
2. **Kaggle flattens uploaded dataset folders**, and `knee-src` isn't a legal package name, so
   `from src.… import` failed. Notebooks rebuild a real package under `/kaggle/working`.
3. **Kaggle mounts inputs as a nested tree** (`/kaggle/input/{competitions,datasets}/…`), not the
   flat `/kaggle/input/<slug>/` that public example code assumes. **Every hardcoded path was
   wrong, including `COMP`.** All paths now resolve by walking `/kaggle/input` for marker files.
4. **Dead `find_dir_ckpt()` call + `os.path.isdir(None)`** — `py_compile` passes both since it only
   checks syntax. Added an AST undefined-name check that caught them.
5. **P100 vs PyTorch.** Kaggle defaulted to a Tesla P100 (sm_60); Kaggle's PyTorch supports sm_70+
   only, so every CUDA op died with `cudaErrorNoKernelImageForDevice`. The CLI *does* select an
   accelerator, but **the value is case-sensitive and invalid strings are accepted silently** —
   `nvidiaTeslaT4` was ignored and fell back to P100 twice. Proved it by pushing a garbage value
   (`ZZINVALID`, accepted without error), then used the documented **`NvidiaTeslaT4`**.

> **Always pass `--accelerator NvidiaTeslaT4` when pushing a GPU kernel.** Omitting it silently
> gives a P100, which cannot run our model at all.

#### Batch size, forced by the memory measurement

7.7 GB at batch 2 means batch 4 would sit on the T4 ceiling and OOM mid-epoch. So `BATCH=2` with
`ACCUM=4` (effective batch 8, unchanged). Loss is divided by `ACCUM` so accumulated gradients
average rather than sum, and `OneCycleLR.total_steps` counts **optimiser** steps, not batches —
otherwise the schedule runs off its end mid-training and raises.

Verified numerically rather than by inspection: batch-2×4 accumulation produces parameters
**identical** to a true batch-8 step, and the optimiser-step count (1652) fits the schedule
budget (1656).

### Session 11: FIRST TRAINED MODEL — fold 0 grouped-CV macro AUC **0.7746**

The first real imaging result, and it clears both floors decisively.

| Epoch | Grouped-CV macro AUC | Loss | Time |
|---|---|---|---|
| 0 | 0.6818 | 1.078 | 3601 s |
| 1 | 0.7304 | 1.065 | 3450 s |
| 2 | 0.7590 | 0.911 | 3529 s |
| **3** | **0.7746** | 0.741 | 3521 s |

Floors: 0.500 constant · 0.595 series-composition · **0.598 DICOM-metadata-only (site-grouped)**.
At 0.7746 the model is genuinely reading images, not memorising scanners.

Per-label at epoch 3:
`Contusion 0.865 · Fracture 0.858 · Baker's 0.844 · Synovitis 0.836 · ACL 0.775 · MCL 0.771 ·
PF OA 0.754 · Lateral OA 0.744 · Effusion 0.738 · Medial OA 0.714 · Medial Meniscus 0.702 ·
Lateral Meniscus 0.695`

#### The fingerprint fix worked

```
151 distinct fingerprints  (was 3,229)
grouping: 151 groups / 4349 studies (ratio 0.03), singletons 43,
          largest [350, 256, 233, 214, 211]
```

Ratio **0.74 → 0.03**; largest groups hold 200–350 studies. This is real site-grouped CV, and 151
is the right order of magnitude against the forum's reported 265. **The 0.7746 is honest.**

#### Two findings worth acting on

- **Synovitis 0.836 from imaging, vs 0.630 from text.** It was the *worst* text label — session 7
  concluded regex fundamentally could not extract it (only 3/27 gold positives even carry a
  thickening word). The imaging model learns it far better than its labels suggest, which says the
  architecture is sound and that **label noise, not capacity, is the current ceiling.**
- **Menisci are now the weak spot (0.702 / 0.695).** Exactly as [02-domain-primer.md](02-domain-primer.md)
  predicted: a tear spanning ~3 slices of 16 is the hardest thing in this dataset, and the one most
  punished by aggressive slice downsampling.

#### Not converged

Loss fell 1.08 → 0.74 and AUC rose every epoch — training was **stopped early, not finished**. More
epochs is the cheapest available gain before touching architecture.

Checkpoint `knee_fold0.pth` (82 MB) verified loadable — correct backbone, head `[12, 3840]`,
score recorded — and published as private dataset `knee-model-v1`.

### Session 12: submission pipeline produces real predictions

`knee-submit` v5 runs real inference from `knee-model-v1`:

```
ckpt: /kaggle/input/knee-model-v1   checkpoints found: 1
loaded 1 model(s) on cuda
WROTE submission.csv - shape (3, 13), load failures: 0
```

Schema matches `sample_submission.csv` exactly, no NaN, all in [0,1], **zero constant-0.5 rows**,
mean per-label spread 0.251 - genuinely differentiated output, not a degenerate constant.

#### The bug worth remembering: a silent constant-0.5 submission

v3 ran clean, exited 0, wrote a valid file - and never loaded the model. It would have scored
0.500 while looking perfectly healthy, wasting a daily submission slot.

Cause was a regression introduced one commit earlier: scoping the checkpoint search to
`/kaggle/input/datasets` with an early `return None`, to avoid crawling the 819,640-file
competition tree. **Kaggle mounts datasets inconsistently within the same run** - measured:

```
/kaggle/input/datasets/sibusisokhumalo11/knee-src   (nested)
/kaggle/input/knee-model-v1                          (flat)
```

so the early return gave up before seeing the model. Now walks all of `/kaggle/input` and prunes
only `competitions/`. Any fallback now prints a loud banner - a degraded run that still writes a
scoreable file is worse than a crash precisely because it is easy to miss. 4 regression tests
cover both mount layouts and the prune.

> **Lesson: diagnose by dumping actual state, not by guessing a second fix.** One diagnostic run
> gave the answer that two rounds of speculation would not have.

### Session 13: ENSEMBLE LABELS — 0.757 -> 0.8234 vs gold

The LLM extractor finally ran (on Kaggle GPU, Qwen2.5-Instruct, open weights).
Headline numbers on the 58 gold studies:

| Extractor | Macro AUC |
|---|---|
| Rule | 0.7565 |
| LLM | 0.7806 |
| **Plain mean of both** | **0.8234** |

**The LLM alone (+0.024) is inside the n=58 noise band and is NOT a clear win.** The ensemble is,
at +0.067 over the rule extractor. It is also a legitimate combination rather than fitting: an
unweighted mean has no parameters tuned against the gold set. Per-label *selection* would be
overfitting the only ground truth we have, and is deliberately not done.

#### Why it works: the errors are complementary

| Label | Rule | LLM | Delta |
|---|---|---|---|
| **Effusion** | 0.612 | **0.790** | **+0.178** |
| Medial OA | 0.778 | 0.877 | +0.099 |
| Lateral OA | 0.757 | 0.844 | +0.087 |
| Medial Meniscus | 0.759 | 0.845 | +0.086 |
| Synovitis | 0.630 | 0.679 | +0.049 |
| Contusion | 0.677 | 0.726 | +0.049 |
| **MCL** | **0.859** | 0.655 | **-0.204** |
| ACL | 0.906 | 0.876 | -0.030 |

The LLM fixes exactly what regex could not — **Effusion**, the label session 7 gave up on — while
the rule extractor is far better on **MCL**, where the LLM predicted 33 positives against 9 actual
(precision 0.212). Neither is dominant; averaging captures both.

#### Reproducibility gap, now fixed

That run's Kaggle log came back **0 bytes**, so there is no record of whether the 4-bit 7B path or
the 3B fp16 fallback produced the scores. The numbers are real (the scores CSV survived and was
re-scored locally against gold), but the model behind them is unknown. The notebook now writes
`llm_run_info.txt` alongside its output so this cannot recur.

Also: Kaggle kernel logs are **sometimes JSON, sometimes plain text**. A JSON-only parser fails
silently and looks like an empty run.

### Kaggle run order

1. Upload `src/` as a Kaggle Dataset (`knee-src`) and `data/labels_v1.csv` (`knee-labels`).
2. Run `kaggle_01_smoke.py` — **read its numbers before anything else.** If a study takes >2s to
   load, cache to `.npy` and publish as a Dataset before training, or every epoch re-decodes 800k
   DICOMs and burns the weekly GPU quota on I/O.
3. Run `kaggle_02_train.py` on fold 0 only. Confirm grouped CV clears **0.598** (the
   metadata-only floor) — under that, the model isn't reading the images.
4. Publish `/kaggle/working` as `knee-model-v1`, run `kaggle_03_submit.py`, submit.

`kaggle_03_submit.py` writes a valid constant-0.5 file when no checkpoint is attached, so the
submission path can be proven end to end **before** the model exists. Worth doing first — it
converts "does our notebook even produce a scoreable file" from an unknown into a settled question.

## Immediate next steps (after session 11)

1. **Train longer.** 4 epochs was not convergence — loss and AUC were both still improving. 8–10
   epochs on fold 0 is the cheapest gain available and needs no code change beyond `EPOCHS`.
2. **Run the remaining folds** only once epoch count is settled; 5 folds × 4 h is most of a weekly
   quota, so don't spend it on a configuration still being tuned.
3. **Submit.** `knee-model-v1` is published and `kaggle_03_submit.py` already resolves checkpoints
   from it. The submission path is proven end to end (session 10), so this is now low-risk.
4. **Then** improve labels, not the model. Synovitis scoring 0.836 from images while its text
   labels sit at 0.630 is direct evidence that **label noise is the binding constraint** — which is
   what the LLM extractor (Phase 1 step 2) exists to fix, and what the rules change now permits
   more options for.

## Older: immediate next step (session 9 starting point)

Just run this — the reliability fix is in, verified working, nothing else to set up:

```bash
python scripts/extract_labels_llm.py --compare --model qwen2.5-coder:7b --timeout 90
```

Expect ~50-70 minutes on this machine's CPU with `ollama serve` running. Watch the per-study
progress lines; if one study visibly stalls past ~2-3 minutes, that's worth investigating (which
study, how long its report is) rather than just killing and re-running blind. Report the real
macro AUC once it finishes — do not reuse the 6-study (0.855) or 14/58-partial numbers from
session 8, neither is a trustworthy estimate of the full-58 result.

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

| | 3a | 3b | 4 | 5a | 5b | 6: +Croatian, bare "OA", Dutch kraakbeen |
|---|---|---|---|---|---|---|
| Macro AUC | 0.685 | 0.688 | 0.710 | 0.728 | 0.734 | **0.757** |

Per-label AUC now: ACL 0.906 · Fracture 0.823 · Baker's 0.823 · MCL 0.859 · Medial OA 0.778 ·
Lateral OA 0.757 · Lateral Meniscus 0.773 · Medial Meniscus 0.759 · PF OA 0.680 · Contusion 0.677 ·
Synovitis 0.630 · Effusion 0.612

## Session 6: chasing the three OA labels (Medial/Lateral/PF)

Same method as before — dumped actual gold disagreements per label rather than guessing. Found
three gaps, all confirmed by checking real prevalence before investing:

- **Bare "OA" abbreviation.** 270/4,407 reports say "OA" rather than spelling out
  "osteoarthritis" (e.g. "Incipient OA of all three compartmens") — none of the full-word patterns
  caught it. Added `\boa\b`, plus "all three compartments" to the both-sides laterality list (covers
  medial+lateral; PF OA is a separate label so this still slightly undercounts, cheaply accepted).
- **Croatian, ~5% of the corpus** (214/4,407 reports contain the marker word "promjene"),
  previously uncovered. Added OA vocabulary (`artroza`, `gonartroza`, `hondromalacija` — a
  different root spelling from the Romance-language chondromalacia pattern), effusion
  (`izljev`), negation (`bez`/`nema`), and the medial/lateral terms (only `medijalni` needed a new
  pattern — `lateralni` already matched the existing `\blateral\w*` prefix).
- **Dutch "kraakbeen"** (cartilage) is a different root from German "knorpel", so
  "kraakbeenverlies/-lijden/-schade" (cartilage loss/disease/damage) was invisible despite Dutch
  otherwise being covered.

**Net this session: 0.734 → 0.757.** All three OA labels moved (Medial 0.658→0.778, Lateral
0.676→0.757, PF 0.656→0.680); Meniscus labels also picked up spillover gains from the Croatian and
bare-OA vocabulary. 40 tests still passing, no new regression tests added this session (token-
conservative pass — the fixes are narrow vocabulary additions of the same shape already covered by
existing multilingual tests, judged lower-risk than the structural bugs fixed in session 5).

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

## Session 5: chasing Fracture and Medial Meniscus

### Fracture: 0.747 → 0.795

Dumped Fracture false positives and negatives. Found two things, one fixable and one genuinely
ambiguous — kept them separate rather than guessing at a single fix:

- **Genuinely ambiguous, not fixed**: two Greek false positives both said
  "μικροδοκιδώδες κάταγμα ανεπάρκειας" (microtrabecular **insufficiency** fracture, synonym SONK)
  — gold=0 both times. But a Spanish false-negative dump for a *different* study, checked directly
  against its gold label, showed "subchondral insufficiency fracture" wording scoring gold=**1**.
  So "insufficiency fracture" phrasing does not reliably predict the gold label either way — this
  looks like real clinical judgment-call variance in how the 58 gold studies were labelled, not a
  pattern bug. Documented rather than patched; fixing it from 2-3 conflicting examples would be
  overfitting the exact thing the evaluation protocol warns against.
- **A real, fixable, and much bigger bug**: a Turkish false negative used "subkondral **kırığı**"
  (subchondral fracture) — the word was already in the vocabulary (`kirik\w*`), but spelled with
  Turkish dotless **ı** (U+0131), a distinct Unicode letter from ASCII "i" (U+0069), not an accent
  variant of it. Every Turkish pattern written with ASCII "i" — `kirik` (fracture), `yirtik`
  (tear), `sivi` (fluid), `ic yan` (medial) — was silently failing against real Turkish text using
  "kırık", "yırtık", "sıvı". Folded ı→i in `text.normalize()`, the same layer that already handles
  the Greek mu substitution. This is a **corpus-wide fix, not a fracture-specific one** — it
  touches every Turkish pattern across all twelve labels.

### Medial Meniscus: 0.637 → 0.724 (Lateral Meniscus was 0.739, now 0.730 — the two are now close)

This was the flagged mystery from session 4 — Medial trailing Lateral by a wide margin on
identical pattern logic. Traced it rather than assuming the Turkish-ı fix explained it: checked
which newly-detected true positives contain Turkish dotless-ı or Cyrillic script. **5 of them do.**
So the asymmetry wasn't a Medial-specific bug at all — it was the Turkish-ı fix (Fracture's
byproduct) plus new Bulgarian meniscus/injury vocabulary (`мениск\w*`, `руптур\w*`, `разкъсв\w*`,
~5% of the corpus, checked prevalence directly: 220/4,407 reports contain Cyrillic script) landing
disproportionately on medial-side mentions in this particular 58-study sample. Added Bulgarian
coverage across all ten concepts, mirroring the Greek session-4 approach, once the 5% prevalence
number justified the investment.

### A second, bigger Turkish bug found while writing the regression test

Writing a test for the "kırığı" fix ("subkondral kırığı izlenmektedir" — subchondral fracture **is
observed**) failed even after the ı-fold, tracing to two more issues:

- **Turkish consonant softening**: "kırık" (fracture) → "kırığı" (its fracture) softens the final
  k to ğ before a vowel suffix — ordinary Turkish morphology, not a typo. Same for "yırtık" (tear)
  → "yırtığı". Fixed both `kirik\w*` → `kiri[kg]\w*` and `yirtik\w*` → `yirti[kg]\w*`.
- **A backwards negation pattern, and a bigger one than the fracture case alone.**
  `izlenme\w*` was meant to catch "not observed," but Turkish "izlenmektedir" (**is** observed —
  positive) and "izlenmemektedir" (is **not** observed — negative) both start with the substring
  "izlenme," because "-mekte(dir)" is a present-tense suffix that happens to start with "me"
  regardless of polarity — it isn't the negation morpheme itself. The wildcard therefore matched
  the affirmative form too and **negated real positive findings**. `saptanma\w*` had the identical
  structural bug. `gorulume\w*` had it *and* a spelling typo (extra "u" — "görülme" normalizes to
  "gorulme," not "gorulume," so this one likely never matched its intended word at all in any
  report). Replaced all three with explicit negative-suffix enumeration
  (`izlen(?:medi|memis|memekte)\w*` and parallel forms) rather than wildcarding past the ambiguous
  root. This is corpus-wide, not fracture-specific — it touches every Turkish positive finding that
  used one of these three very common radiology-report verbs.

**Net this session: 0.710 → 0.734 macro AUC** (0.728 after the ı/Bulgarian pass, 0.734 after the
negation-verb fix on top). 40 tests passing (6 new this session, all currently green — the
Turkish-fracture test failed once mid-session and stayed red until the consonant-softening +
negation fixes actually resolved it, rather than being adjusted to match wrong behavior).

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

### 2026-08-08 — Session 8 (LLM extractor built, first full run interrupted)
- Built `src/extract/llm.py` + `scripts/extract_labels_llm.py`: local-Ollama LLM extractor,
  same interface as `RuleExtractor`, drops into the existing `evaluate.py` harness.
- Demo (10 synthetic reports, no data needed): `llama3.2` hallucinates badly (invented findings
  with zero textual basis on 5/10 cases); `qwen2.5-coder:7b` much cleaner (7/10 clean) but still
  misses real negations the rule extractor catches (a denied German "Kein Gelenkerguss" scored
  as present).
- First full 58-study comparison attempt stalled 50+ min with no output — diagnosed as a real
  visibility bug (no per-study printing + Python's full stdout buffering when piped to a file),
  not a fundamental blocker. Fixed: live flushed progress output, shorter default timeout,
  `--limit` flag for smoke-testing. Verified the fix on a 6-study run (clean, ~40s/study) and
  a second full-58 attempt (reached 14/58 cleanly, rule-extractor half reconfirmed 0.757) before
  stopping it intentionally to end the session — see "Immediate next step" above.
- **No LLM-vs-rule number on the full 58 studies yet.** Next session starts by just running the
  command in "Immediate next step" above.

### 2026-08-08 — Session 7 (Effusion/Synovitis, no code change)
- Chased Effusion and Synovitis per the pattern of prior sessions, but this one ended in a
  documented non-fix rather than a score bump — see the next-actions note above for the reasoning.
  No commit to `src/` this session; `docs/00-state.md` only.
- Confirms the label set has genuine noise (at least one verified report/label mismatch) and a
  hard ceiling for keyword-only Synovitis extraction. Both are useful things to know before
  building the LLM extractor, which is the next real lever for these two labels.

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
