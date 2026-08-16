# Handover prompt — verify new runs, then submit

Paste everything below the line into a fresh Claude Code window.

---

You are picking up an RSNA Kaggle competition mid-flight. Repo:
`C:\Users\lovilocal.adm\Desktop\RSNA-Knee-Abnormality-Detection`, branch
`handover-continuation`. Kaggle account `sibusisokhumalo11`, already
authenticated — use `python -m kaggle` (the bare `kaggle` binary is not on PATH).
Use **Claude in Chrome** for anything Kaggle's API can't reach: kernel logs don't
stream for script kernels and competition submission is a browser-only action.

**Your job:** some training runs were done on a different account. Verify them
properly, and if they're better than what we have, submit. Do not submit
anything you haven't verified.

## Where we stand

| | |
|---|---|
| Best scored LB | **0.864** (submission `55503594`) — 5 folds, 6 slices/slot, CV 0.8185 |
| Previous | 0.856 (6-member cross-family mix) |
| Rank | ~654 / 1383 · top-10 cutoff **0.936** |
| Best config | DINOv2-small + `head="xattn"` + **6 slices/slot** + 336px + `labels_blend_v1` |

Submission history: 0.783 (old pipeline) → 0.850 → 0.856 → **0.864**.

## The one thing that worked

Ten levers were measured. **Only slice coverage paid**: 3 → 6 slices/slot gave
**+0.0236 across all five folds**, and the gain landed on the *focal* findings
exactly as predicted (Fracture +0.068, PF OA +0.049, Medial Meniscus +0.049)
while diffuse ones barely moved (Effusion +0.007). Three samples spaced 6–14
slices apart were letting evidence fall between them.

Everything else returned noise: epochs (×3 attempts), DINOv2-base (−0.0005),
448px (−0.0014), 12 slices (−0.005, saturated). **Do not re-test these.**
Capacity and pixel density are not the constraint.

## How to judge a number

- **Fold noise is ±0.011.** Single-fold differences under ~0.02 are unmeasurable.
  Say so rather than reporting them as gains.
- **CV → LB offset is roughly +0.045 to +0.055.** Two measured points: CV 0.7949
  → LB 0.850, and CV 0.8185 → LB 0.864. It drifts; re-measure rather than
  extrapolating far from it.
- `gold58` scores against real annotations but n=58; a 0.02 spread there is noise.
  Three readings from that ruler have already been overturned.

## Verify inputs before believing any score

Four bugs in this project produced clean runs with wrong inputs and believable
numbers. Check these in every log before trusting a result:

1. **Shard count** — `cache (4407, 6, N, 336, 336)` and `training pool 4349`.
   A sharded cache mounts one directory per shard and a half-load once scored
   0.7956 where the truth was 0.8161. There is now an assert; confirm it ran.
2. **Encoder width** — checkpoint `cls_token` width must match the mounted
   DINOv2 (384 = small, 768 = base). Mismatch previously killed a submission.
3. **Head type** — checkpoints record `head`; older ones don't and it is inferred
   from `head.cross_attn` keys. A slot-head default silently fails to load xattn.
4. **Stale `src`** — cache kernels used to ship `pkg/src` in their output and it
   shadowed the live `knee-src`. `SKIP_DIRS` now excludes `pkg`.

Also: `kaggle datasets version` is **asynchronous**. After `--push-src`, wait for
`datasets status` to report ready before launching, or the kernel runs old code.

## Operational limits, all learned the hard way

- **The Kaggle TPU image cannot decode DICOM.** Train on TPU, **submit on GPU or
  CPU**. A TPU submission produces an all-0.5 file that looks successful.
- **One concurrent TPU session.** Training blocks a TPU submission.
- TPU billing is wall-clock including the cache mount — run all folds in **one**
  kernel, not five.
- CPU inference costs ~2.3 s per study per member; ~1,300 test studies means
  6 members ≈ 5.5 h against the 9 h cap. Size the ensemble to fit.

## What to do

1. Find the new runs (`python -m kaggle kernels list --user <account>`, or ask
   for the kernel/dataset names). Read each log in Chrome and report the fold
   scores **and** the input-sanity lines above.
2. Compare against 0.8185 (current best 5-fold mean) using the ±0.011 / ≥0.02
   rules. Say plainly if they aren't better.
3. If they are better: publish the checkpoints as a Kaggle dataset, point a
   submission kernel at them, run on **CPU or GPU**.
4. **Before submitting, check the log for:** `combined N member(s)`, `decode
   failures 0`, and the absence of `still the 0.5 default`. That tripwire has
   already caught one all-0.5 submission that would have scored ~0.5.
5. Submit via Chrome (Output tab → Submit to Competition) with a description
   recording config and CV. Confirm it registers, then report the score.

**Critical:** if the new models were trained at a different slices-per-slot than
the test cache the submission builds, the tensors will be the right shape and
the wrong anatomy — it will score badly and look like a bad model. The
submission notebook must rebind `N_SLICE` to match training and average logits
over groups. See `kaggle/submit-6slice/build.sh`.

## How to work

Evidence first. State a threshold **before** a run and honour it when the number
lands just under. Change one variable at a time. Report nulls plainly — most of
what's been learned here came from experiments that returned nothing, and
mislabelling noise as progress would have sent the whole project down a dead
end. When a result looks surprising, check the inputs before believing it; four
times out of four so far, a surprising number meant a broken input rather than a
discovery. Don't narrate optimism you can't support: 0.864 against a 0.936
cutoff is real progress and still a long way short.
