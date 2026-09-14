"""uncertainty_weighted_bce: see src/model/loss.py for why this exists."""

import torch
import torch.nn.functional as F

from src.model.loss import uncertainty_weighted_bce


def test_full_weight_matches_plain_bce_mean():
    """undecided_weight=1.0 (the control) must reduce identically to
    BCEWithLogitsLoss()'s own mean - same numbers as the pipeline used
    before this function existed, not just "close"."""
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    targets = torch.rand(8, 5)
    got = uncertainty_weighted_bce(logits, targets, undecided_weight=1.0)
    want = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    assert torch.allclose(got, want, atol=1e-6)


def test_full_mask_drops_undecided_cells_entirely():
    """undecided_weight=0.0: a target's undecided cells must not move the
    loss at all, decided cells must have EXACTLY their usual weight."""
    logits = torch.tensor([[1.0], [1.0], [1.0]])
    targets = torch.tensor([[0.9], [0.5], [0.1]])  # decided, undecided, decided
    got = uncertainty_weighted_bce(logits, targets, undecided_weight=0.0, undecided=0.05)
    decided = torch.tensor([[1.0], [1.0]])
    decided_targets = torch.tensor([[0.9], [0.1]])
    want = F.binary_cross_entropy_with_logits(decided, decided_targets, reduction="mean")
    assert torch.allclose(got, want, atol=1e-6)


def test_partial_weight_matches_the_closed_form_weighted_average():
    """A single target, three cells (two decided, one undecided). The
    per-target reduction is just a weighted average of the three cells' own
    BCE values - check that directly rather than only checking direction."""
    logits = torch.tensor([[1.0], [1.0], [1.0]])
    targets = torch.tensor([[0.9], [0.5], [0.1]])
    per_cell = F.binary_cross_entropy_with_logits(
        logits.squeeze(-1), targets.squeeze(-1), reduction="none"
    )
    for w in (1.0, 0.25, 0.0):
        got = uncertainty_weighted_bce(logits, targets, undecided_weight=w)
        weights = torch.tensor([1.0, w, 1.0])
        want = (per_cell * weights).sum() / weights.sum()
        assert torch.allclose(got, want, atol=1e-6), w


def test_normalizes_per_target_not_globally():
    """Two targets, one with an undecided cell down-weighted and one
    without. Per-target normalization means the down-weighted target's
    contribution is the mean of its OWN remaining cells, not diluted by
    having fewer of them relative to the untouched target."""
    logits = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    targets = torch.tensor([[0.5, 0.9], [0.9, 0.9]])  # col0 has one undecided cell
    got = uncertainty_weighted_bce(logits, targets, undecided_weight=0.0, undecided=0.05)

    # col0 supported by row1 only (0.9); col1 supported by both rows (0.9, 0.9).
    col0 = F.binary_cross_entropy_with_logits(
        torch.tensor([2.0]), torch.tensor([0.9]), reduction="mean"
    )
    col1 = F.binary_cross_entropy_with_logits(
        torch.tensor([2.0, 2.0]), torch.tensor([0.9, 0.9]), reduction="mean"
    )
    want = (col0 + col1) / 2
    assert torch.allclose(got, want, atol=1e-6)


def test_nonfinite_targets_are_dropped_not_propagated_as_nan():
    logits = torch.tensor([[1.0], [1.0]])
    targets = torch.tensor([[0.9], [float("nan")]])
    got = uncertainty_weighted_bce(logits, targets, undecided_weight=1.0)
    assert torch.isfinite(got)
    want = F.binary_cross_entropy_with_logits(
        torch.tensor([1.0]), torch.tensor([0.9]), reduction="mean"
    )
    assert torch.allclose(got, want, atol=1e-6)


def test_zero_support_batch_returns_zero_not_nan_and_stays_differentiable():
    logits = torch.tensor([[1.0, 1.0]], requires_grad=True)
    targets = torch.tensor([[0.5, 0.5]])  # both undecided
    got = uncertainty_weighted_bce(logits, targets, undecided_weight=0.0)
    assert torch.isfinite(got)
    assert got.item() == 0.0
    got.backward()  # must not raise
    assert logits.grad is not None
