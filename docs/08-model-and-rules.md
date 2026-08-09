# 08 — Model choice & rules position

## 1. RULES CHANGE, 2026-08-08: commercial LLMs are now permitted

Host announcement ([discussion/733965](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965),
Po-Hao "Howard" Chen), quoting the operative sentence:

> "Use of commercially hosted LLMs and other external inference services is permitted, provided
> that the service and method of use otherwise comply with the Competition Rules, including
> requirements that external data, models, software, and associated tools be reasonably accessible
> to all participants and of minimal cost."

Explicitly: sending report text to an external LLM for label extraction is **not** prohibited
private sharing (Rules §2.6.b).

**This reverses the assumption we built Phase 1 on.** Sessions 4–7 deliberately avoided commercial
APIs because the question was unresolved and the conservative reading was "not allowed". That was
the right call *at the time* — but it is now obsolete, and any doc still saying "assume commercial
APIs are not permitted" is stale.

Two constraints remain, and they are not trivial:
- **"minimal cost" and "reasonably accessible to all participants."** A pipeline that needs $500 of
  frontier-model inference is arguably neither. A cheap model over 4,407 short reports is fine.
- The host **reserves the right to rule a service prohibitively costly or unfair after the fact.**

**Still unresolved**: whether external MRI datasets under research-only licences (MRNet, OAI,
fastMRI+, SKM-TEA) are usable given the cash prizes. A participant raised the commercial-use
conflict directly under the announcement; no host answer yet. **Do not build a solution that
depends on them** until that is answered.

### What this changes for us

Label extraction is a **training-time, one-off** job over 4,407 reports — it is *not* in the
inference path (there are no reports at test time). So the options are now:

| Option | Cost | Compliance | Verdict |
|---|---|---|---|
| Rule extractor (current, 0.757) | free | certain | Keep as the baseline and the fallback |
| Open-weights LLM on a Kaggle GPU | free | certain | **Preferred** — free, offline, no cost-fairness question at all |
| Commercial API | $ | now permitted, but "minimal cost" is a judgement call | Only if it clearly beats the above |

Preference for the open-weights-on-Kaggle route is not caution for its own sake: it is free, it
sidesteps a rule whose boundary the host explicitly reserved the right to redraw, and the local
experiment (session 8) showed a 7B open model already handles most of these reports competently.

---

## 2. Architecture: 2.5D CNN + attention-MIL

Not a novel design — deliberately. The same shape won the last **three consecutive RSNA
competitions**:

| Competition | Backbone | Slice fusion |
|---|---|---|
| RSNA 2022 cervical spine (1st) | EfficientNetV2-S / ConvNeXt | LSTM |
| RSNA 2023 abdominal trauma (1st) | CoaT Lite / EfficientNetV2-S | GRU |
| RSNA 2024 lumbar spine (1st) | ConvNeXt-S / EfficientNetV2-S | BiLSTM + attention-MIL |

Ours (`src/model/net.py`): shared **EfficientNetV2-S** backbone over 2.5D adjacent-slice triplets,
**gated attention-MIL** pooling per plane, concat → 12-way multi-label head.

Deliberate choices, with reasons:

- **2D ImageNet backbone, not 3D.** With noisy report-derived labels and a free-tier GPU, weight
  initialisation matters more than capacity. Also keeps us inside the 9-hour inference cap that the
  Efficiency track rewards.
- **One shared backbone across all three planes, not three.** ~4,400 studies with noisy labels
  cannot supervise three separate backbones; that is a straight route to overfitting. Plane-specific
  reasoning lives in the cheap per-plane attention heads instead.
- **Attention pooling, not mean pooling.** A meniscal tear can occupy 3 slices out of 16. Mean
  pooling dilutes exactly the hardest label. The attention weights are also inspectable — the
  fastest way to catch "the model is reading the wrong plane."
- **Capped per-label `pos_weight`.** Macro-AUC weights Fracture (6.9% prevalence) the same as
  Effusion (54.9%); unweighted BCE would undertrain precisely the labels dragging the score down.
- **No horizontal flip augmentation.** Medial vs lateral are *different labels*. A left-right flip
  silently relabels the study. Only anterior-posterior flips are safe.

### Model licensing

`timm` EfficientNetV2 / ConvNeXt weights are Apache-2.0 — freely and publicly available, so they
satisfy the external-tools rule, and they are compatible with the winners' obligation to open-source
under CC-BY-NC 4.0. Avoid any backbone under a non-commercial or research-only licence: it would
collide with the prize terms exactly the way the external-dataset question does.

---

## 3. Anti-overfitting protocol

Non-negotiables, in priority order:

1. **GroupKFold on the scanner fingerprint. Never random KFold.** Random folds overstate macro AUC
   by ~0.053 through site memorisation. Enforced in `src/model/validation.py` and covered by a
   regression test (`test_grouped_folds_never_split_a_group`).
2. **The 58 gold studies are never trained on** — held out in `kaggle_02_train.py`. They are the
   only real ground truth; training on them then reporting agreement with them is circular.
3. **Trust grouped CV over the public LB.** ~1,300 test studies split public/private is a small,
   noisy sample; the organisers also warn prevalence differs across train/public/private.
4. **Always read all twelve per-label AUCs**, never the macro alone. One dead label costs ~0.04 of
   final score — the difference between 5th and 50th.
5. **Floors any real model must clear**: 0.500 constant · 0.595 series-composition · **0.598
   DICOM-metadata-only under site-grouped folds**. A model scoring under ~0.60 is not reading the
   images, whatever its loss curve says.

### External validation (candidate, blocked)

**MRNet** (Stanford, 1,370 knee MRIs, labelled ACL tear / meniscal tear / abnormality) overlaps 3 of
our 12 labels and has a held-out validation set — a genuine out-of-distribution check on the labels
we care most about. **Blocked on the same unresolved licence question as the other external
datasets.** Treat as a validation-only idea pending a host ruling; do not train on it.
