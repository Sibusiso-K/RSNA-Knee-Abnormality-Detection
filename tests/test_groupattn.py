"""The group-attention head — the slice axis, moved inside the model.

Tested without the encoder, like `test_slotnet.py`: DINOv2 weights only exist
on Kaggle, and everything worth pinning here is about masking and about which
tokens a query can actually reach.

The property that matters is the one the head exists for. `XAttnHead` sees one
slice group at a time and the harness averages the logits, so evidence in one
group is diluted by groups without it — measured as 3->6 slices +0.0236 but
6->12 slices -0.005. This head sees every group at once, so a query can select
rather than average. `test_a_single_group_can_dominate` is that claim.
"""

import torch

from src.labels import TARGETS
from src.model.slotnet import N_SLOT, GroupAttnHead

DIM, POOLED, HID, T = 32, 96, 16, 5


def _head(**kw):
    torch.manual_seed(0)
    return GroupAttnHead(dim=DIM, pooled_dim=POOLED, hidden=HID, heads=2,
                         dropout=0.0, **kw).eval()


def _inputs(b=2, g=2):
    torch.manual_seed(1)
    return (torch.randn(b, N_SLOT, g, T, DIM),
            torch.randn(b, N_SLOT, POOLED),
            torch.ones(b, N_SLOT))


def test_output_shape_is_one_logit_per_target():
    head = _head()
    tokens, pooled, mask = _inputs()
    assert head(tokens, pooled, mask).shape == (2, len(TARGETS))


def test_absent_slots_cannot_influence_the_output():
    """A masked slot is masked in EVERY one of its groups.

    The key-padding mask is built by expanding (B,S) over G and T, and getting
    that expand wrong is invisible: the run trains, and the model reads black
    images that the encoder has mapped to confident features.
    """
    head = _head()
    tokens, pooled, mask = _inputs()
    mask[0, 2] = 0.0
    with torch.no_grad():
        before = head(tokens, pooled, mask)
        tokens[0, 2] = torch.randn(2, T, DIM) * 100.0   # scribble every group
        after = head(tokens, pooled, mask)
    assert torch.allclose(before, after, atol=1e-5)


def test_an_unmasked_slot_does_influence_the_output():
    """Complement of the above, so it cannot pass on a dead forward pass."""
    head = _head()
    tokens, pooled, mask = _inputs()
    with torch.no_grad():
        before = head(tokens, pooled, mask)
        tokens[0, 3] = torch.randn(2, T, DIM) * 100.0
        after = head(tokens, pooled, mask)
    assert not torch.allclose(before, after, atol=1e-3)


def test_a_single_group_can_dominate():
    """Evidence in ONE group must be able to move the output on its own.

    This is the whole argument for the head. Under logit averaging over G
    groups, a finding present in one group and absent from the rest arrives
    attenuated by 1/G no matter how strong it is. Here the token is reachable
    by the query directly, so a large perturbation confined to group 1 must
    still change the answer.
    """
    head = _head()
    tokens, pooled, mask = _inputs(b=1, g=4)
    with torch.no_grad():
        before = head(tokens, pooled, mask)
        # A new PATTERN in group 1, not a constant offset. `token_proj` opens
        # with a LayerNorm, so adding the same number to every feature is
        # mean-centred straight back out and changes nothing — which is how
        # this test first "failed" against a head that was working correctly.
        torch.manual_seed(7)
        tokens[0, :, 1] = torch.randn(N_SLOT, T, DIM) * 20.0
        after = head(tokens, pooled, mask)
    assert (after - before).abs().max() > 1e-2


def test_group_identity_is_visible_to_the_head():
    """Which group a token came from must be information the head has.

    The encoder processes each (slot, group) triplet independently and never
    learns their order, so the group embedding is the only carrier. Without it
    the head sees an unordered bag and cannot express "the slice group above
    this one" - permuting the groups would be a no-op.
    """
    head = _head()
    tokens, pooled, mask = _inputs(b=1, g=3)
    with torch.no_grad():
        before = head(tokens, pooled, mask)
        after = head(tokens.flip(2), pooled, mask)      # reorder the groups
    assert not torch.allclose(before, after, atol=1e-4)


def test_more_groups_than_embeddings_raises():
    """Fails loudly rather than indexing off the end of the embedding table.

    Scoring a model at more groups than it was trained on is a real scenario -
    a 6-slice model against a 12-slice cache - and it must not be a silent
    wrap-around.
    """
    head = _head(max_groups=2)
    tokens, pooled, mask = _inputs(b=1, g=3)
    try:
        head(tokens, pooled, mask)
    except ValueError as exc:
        assert "max_groups" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


def test_trained_at_two_groups_scores_at_four():
    """The complement: within max_groups, group count is free at inference.

    A head that could only ever be scored at its training group count would
    make the 6- and 12-slice caches incompatible all over again.
    """
    head = _head(max_groups=8)
    for g in (1, 2, 4):
        tokens, pooled, mask = _inputs(b=1, g=g)
        assert head(tokens, pooled, mask).shape == (1, len(TARGETS))
