# Handover — RSNA Knee, as of 2026-08-21

Paste everything below the line into a fresh Claude Code window.

---

You are picking up an RSNA Kaggle competition mid-flight.

- **Repo:** `C:\Users\lovilocal.adm\Desktop\RSNA-Knee-Abnormality-Detection`, branch **`main`** (everything merged and pushed as of 2026-08-21).
- **Kaggle:** account `sibusisokhumalo11`, already authenticated. Use `python -m kaggle` — the bare `kaggle` binary is not on PATH.
- **Deadline:** final submission 2026-10-22.

**Two Claude sessions share this repo.** Commit and push after every session, on every branch. This has already cost real work: one session reconstructed `GroupAttnHead` and the member fingerprint gate by hand from the Kaggle dataset while the originals sat unpushed in git on a branch nobody else could see.

## Where the score is

| | |
|---|---|
| **Best LB** | **0.864** — a tie between `55503594` (5 members) and `55619346` (16 members) |
| Rank context | top-10 cutoff ~0.936; rank 100 ~0.901 |
| History | 0.783 → 0.850 → 0.856 → **0.864**, flat since 2026-08-14 |

## Submission does NOT need a browser

```
python -m kaggle competitions submit rsna-knee-abnormality-detection \
  -k sibusisokhumalo11/<kernel> -v <version> -f submission.csv -m "<description>"
```

This was assumed to be browser-only for eleven sessions and never was.

## The model in one paragraph

Multi-label image classifier, 12 binary targets, macro-AUC. DINOv2-small (384) mostly frozen — last 6 blocks + final LayerNorm trainable — with a small attention head on top. A study becomes a fixed `(6 slots × 6 slices × 336 × 336)` uint8 tensor: the 6 slots are plane × sequence-weighting bins (`SAG_FLUID_FS, COR_FLUID_FS, AX_FLUID_FS, SAG_FLUID_NOFS, COR_T1, SAG_T1`) with a presence mask; slices enter as 2.5D RGB triplets; a 130 mm physical crop at 336 px gives 0.387 mm/px, constant across scanners; left knees are mirrored to look like right ones (5 of 12 targets are side-specific). `XAttnHead` has 12 target queries cross-attending the patch tokens. Members combine by **per-column rank mean** (AUC reads order only) — the opposite of the right rule for combining *labels*, where BCE needs calibrated targets.

## What is NOT worth retrying — nine measured nulls

Fold-0 baseline is **0.8207** (xattn, 6 slices, 336 px). Fold noise is **±0.011**; the standing bar for a real gain is **≥ +0.02**.

| Lever | Result | Verdict |
|---|---|---|
| Epochs 8v4, 20v10 (old 3-slice config) | — | null |
| DINOv2-base | 0.8120 vs 0.8185 | null |
| 448 px with the **pooled** head | −0.0014 | null |
| **448 px with the spatial head** | 0.8240 | null (+0.0033) |
| 12 slices/slot | 0.8161 | null (saturated) |
| `gattn` — slice axis inside the model | 0.8171 | null |
| Fat-suppression routing | 0.8217 | null |
| **Ensemble breadth** | 5→0.864, 16→0.864, 35→0.861 | **exhausted** |
| **Public pre-trained weights** | 20 members → 0.839, 25 → 0.844 | worse than ours |

On the last one: those weights carry a widely-quoted ~0.899 reputation. Run through this pipeline they score 0.839 — their self-reported holdout of 0.8381 transferred about 1:1. **Do not spend a slot on them.**

The one lever that ever paid: **slice coverage 3 → 6, +0.0236 across all five folds**, concentrated on the *focal* findings exactly as the spacing hypothesis predicted (Fracture +0.068, PF OA +0.049) while diffuse ones barely moved (Effusion +0.007).

## What IS live — three threads, in priority order

**1. Every model in the project is undertrained.** 8 of 10 checkpoints across the two most recent 5-fold runs peaked at epoch 9 of 10 — the *last* epoch — across both label sets and both encoder sizes. `knee-train-6slice-24ep` is built and queued: the exact 0.864 recipe with `EPOCHS 10 → 24`, nothing else changed. **Not yet run — both weekly quotas were exhausted.** Run this first when quota resets. The "epochs are null" rows above were measured on the older 3-slice config and do not cover this.

**2. Frontier-LLM label extraction.** Permitted by competition rules since 2026-08-08, needs **no Kaggle quota**, never started. See the ceiling arithmetic below.

**3. Selective-blend pseudo-labels** (`knee-train-pseudo-sel`). Uniform pseudo-labels gained +0.0111 CV but *lost* on the board (0.864 → 0.857). Diagnosis: the round-1 model is weakest exactly on MCL / menisci / OA — the labels it is being asked to relabel — so it reinforces its own errors. The follow-up applies text-only labels to the 4 disagreeing targets.

## The label ceiling, measured

Labels are report-derived; only 58 of 4,407 studies have real annotations. `labels_blend_v1.csv` scores **0.8930** against those 58.

There is no *better* label set to test (v2 is 0.8935 vs v1's 0.8930 — nothing on 58 studies), so the slope was measured downward instead:

```
labels 0.8930 -> fold-0 CV 0.8207
labels 0.7565 -> fold-0 CV 0.7872
slope = 0.0335 / 0.137 = 0.245
```

Three times the noise band, so the model **does** track its targets — but it recovers only a quarter of any label change. Perfect labels would be worth about **+0.026 CV**. A realistic improvement to ~0.93 buys about +0.009, **under the noise bar**.

Read that honestly in both directions: it is the largest identified lever, *and* it is not obviously enough to reach 0.936 alone. It also explains the pseudo-label regression — with transmission that weak, a round that degrades the contested labels loses more than the confident ones gain.

## Operational facts, all learned the hard way

- **The Kaggle TPU image cannot decode DICOM.** Train on TPU, **submit on GPU or CPU**. A TPU submission writes an all-0.5 file that looks successful.
- **The T4 image bundles `torch_xla`**, so the XLA auto-detect hands back a fake CPU-XLA device instead of the GPU. Force XLA off for GPU kernels. This killed `knee-train-pseudo-sel` v2.
- **One concurrent TPU session.** Quotas: TPU 20 h/week, GPU 30 h/week, **separate budgets**. CPU kernels are free, so cache builds cost nothing.
- `kaggle datasets version` is **asynchronous**. Wait for `datasets status` to report ready or the kernel runs old code. `scripts/launch_kaggle.sh --push-src` already polls.
- Kernels load `src/` from the **`knee-src` dataset**, not from git. Editing a file locally changes nothing on Kaggle until you re-publish.
- **The public test set is ~3 studies**; the real ~1,300 come at the private re-run. A fast, tiny `submission.csv` is normal, not a symptom.

## Verifying a run when the log is empty

`kernels output` returns a zero-byte log often enough that you need a fallback, and "the log was empty so I submitted anyway" is not one.

- **Fold scores live inside the checkpoint:** `torch.load(path)['score']`, along with `head`, `size`, `slices_per_slot`, `labels`, `fold`, `epoch`.
- **Every row exactly 0.5** → the script hit `write_and_exit`; nothing was scored.
- **Member count and weights are recoverable exactly.** Percentile ranks over `n` scored studies take values `k/n`, so a weighted rank-mean lands on exact multiples of `1/(n·W)`, where `W` is the total blend weight. Scan `W` upward until every value in the file is a multiple. `W = 16` confirmed all sixteen members loaded; `W = 50` confirmed 15-at-weight-2 plus 20-at-weight-1.
- **A cache build can be verified from file size alone:** shard 0 of the 448 build is 15,924,658,304 bytes = exactly `2204 × 6 × 6 × 448²` plus a 128-byte npy header. No download needed.

## Five bugs that produced clean runs with wrong inputs

Every one of these finished, saved a checkpoint, and printed a believable number.

1. **Half a sharded cache.** `find_dir` returns ONE directory; a 2-shard cache mounts one per shard. The 12-slice run trained on 2,176 studies instead of 4,349 and scored 0.7956 — read at the time as "more slices hurt". Fixed by `shard_paths()` plus a length assert.
2. **Members fed the wrong anatomy.** The `champ` public members declare 224 px / 9 slices / band 0.35–0.65; they were fed 336 px / 12 / 0.2–0.8. Nothing raised — DINOv2 interpolates its position embeddings — so 5 of 25 members in a scored submission read the wrong scale. `src/model/members.py` now gates on the full fingerprint.
3. **`np.concatenate` on multi-shard caches** loads every shard *and* allocates the result, so peak RAM is twice the cache. Killed a 448 run with a bare `Killed` and no traceback. `src/data/shards.py` memmaps instead.
4. **Stale `pkg/src`** shipped in cache-kernel output shadowed the live `knee-src`. `SKIP_DIRS` excludes `pkg` and the builder deletes it.
5. **Missing `xm.mark_step()`** on XLA: 12.2 s/step and an HBM OOM.

**The slice grids do not nest.** `band_indices` is a `linspace`, so a 6-slice grid sits at fractions of 1/5 and a 12-slice one at 1/11 — they share only their endpoints. A 6-slice model cannot be fed slices carved from a 12-slice cache; it gets the right shape and the wrong anatomy. `build_study_multi` decodes the union of both grids from one header pass and proves itself byte-identical to `build_study` on three real studies before any member is scored.

## How to work

Evidence first. **State a threshold before a run and honour it when the number lands just under** — 0.8240 against a 0.841 bar is a null, not "promising". Change one variable at a time. Report nulls plainly: most of what is known here came from experiments that returned nothing, and calling noise a gain would have sent the project down a dead end.

When a result looks surprising, check the inputs before believing it. Five times out of five, a surprising number meant a broken input rather than a discovery.

Do not trust a reputation number you have not measured yourself. The ~0.899 public weights scored 0.839 here, and a day was spent finding that out.

0.864 against a 0.936 cutoff is real progress and still a long way short. Say so.
