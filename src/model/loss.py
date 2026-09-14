"""BCE loss that can down-weight cells the report never committed on.

Experiment 2 of docs/15-claude-score-improvement-strategy.md. The
validation metric (`macro_auc` in the training scripts) already drops cells
within `UNDECIDED` of 0.5 - a soft label there means "the report did not
address this", not a weak positive or negative - but training's ordinary
`BCEWithLogitsLoss()` gives those same cells full weight. Every training
step pulls the model's logit toward 0 for a cell the metric never scores it
against, which is a plausible bottleneck: the model spends gradient budget
learning to reproduce report silence, exactly the behaviour the validation
metric was built to not reward.

This is deliberately a *weighting* scheme, not a masking one. Missingness in
these reports is not random (a report is more likely to omit a finding when
it is genuinely ambiguous, not uniformly across all studies), so an
`undecided_weight=0.0` that fully masks those cells is one candidate to
measure, not the assumed answer - see docs/15's warning that masking can
also harm generalization by letting the model ignore an entire mode of the
data.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def uncertainty_weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    undecided_weight: float = 1.0,
    undecided: float = 0.05,
) -> torch.Tensor:
    """Per-target-normalized BCE, with near-0.5 cells down-weighted.

    Same reduction path regardless of `undecided_weight`, INCLUDING the
    control (1.0): every weight is then equal and this collapses to a plain
    per-target mean, but going through the identical code path as the
    down-weighted candidates is the point - two differently-implemented
    reductions (a bare `.mean()` for the control, this function only for
    candidates) would let the reduction itself, not the weighting, explain
    any measured difference.

    1. Compute BCE unreduced (per sample, per target).
    2. Weight each cell: `undecided_weight` if `|target - 0.5| < undecided`,
       else 1.0; 0.0 if the target is not finite (drop it either way - an
       actual NaN is a data problem, not a modelling choice this function
       should make).
    3. Normalize PER TARGET by that target's valid weight sum in this batch,
       so a target with many undecided cells contributes one full "vote" to
       the loss like every other target, rather than being starved because
       most of its weight was zeroed out.
    4. Average over targets that had any support (nonzero weight sum) this
       batch. A target with zero support (every cell either non-finite or
       undecided-and-fully-masked) is excluded rather than contributing a
       spurious 0/0.

    Returns a scalar connected to `logits`'s graph even when NO target has
    support in this batch (returns `(logits * 0).sum()`, exactly zero but
    still differentiable) rather than raising - a batch this degenerate
    should not crash a multi-hour run over it.
    """
    finite = torch.isfinite(targets)
    # Replace non-finite targets with a neutral 0.5 before computing BCE so
    # a NaN never enters `raw` in the first place - NaN * 0 is NaN, not 0,
    # so masking after the fact would silently poison every downstream sum.
    safe_targets = torch.where(finite, targets, torch.full_like(targets, 0.5))
    raw = F.binary_cross_entropy_with_logits(logits, safe_targets, reduction="none")

    near_half = (targets - 0.5).abs() < undecided
    weight = torch.where(
        near_half, torch.full_like(targets, undecided_weight), torch.ones_like(targets)
    )
    weight = torch.where(finite, weight, torch.zeros_like(targets))

    weight_sum = weight.sum(dim=0)
    supported = weight_sum > 0
    if not torch.any(supported):
        return (logits * 0.0).sum()

    per_target = (raw * weight).sum(dim=0)[supported] / weight_sum[supported]
    return per_target.mean()
