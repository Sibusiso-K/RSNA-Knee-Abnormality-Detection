# 00 — Project state (READ THIS FIRST)

> **This is the living file.** It is the single source of truth for *where the project is right now*.
> Every other doc explains something stable; this one changes constantly.
> **Update it at the end of every working session.** If it's stale, everything else is a trap.

**Last updated:** 2026-08-10, 17:00 UTC (post-handover session)

> ✅ **`knee-train-8ep` collected** — 8 epochs is not better than 4, see below.
> ✅ **First leaderboard submission made — 0.783**, and CV transfers.
> ❌ **`knee-llm-labels` v3 FAILED** after ~7.5 h — status `ERROR`, and its Kaggle log came
> back **0 bytes**, so there is no stack trace. No `labels_llm_v1.csv`, no
> `labels_ensemble_v1.csv`: the full-corpus stage produced nothing. **We still have no
> ensemble labels for the corpus.** Relaunch with the sharded rewrite.
**Days to final submission (2026-10-22):** ~73

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

**The LB came in +0.008 above CV.** That settles a question this file previously listed as open.
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
