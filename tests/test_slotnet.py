"""Per-diagnosis slot attention.

`SlotHead` is tested without the encoder because the encoder needs weights that
only exist on Kaggle, and because every property worth pinning here is about the
attention and the mask rather than about features.

The mask is the one that matters most. Roughly 22% of slots are absent in this
corpus (measured: overall fill 0.783, and the T1 slots 0.55-0.63), so "what
happens to a missing slot" is not an edge case, it is a fifth of the input.
"""

import numpy as np
import torch

from src.labels import TARGETS
from src.model.slotnet import (
    N_SLOT,
    SLOT_PRIOR_STRENGTH,
    SLOT_PRIOR_TABLE,
    SlotHead,
)


def _head(**kw):
    torch.manual_seed(0)
    return SlotHead(dim=32, hidden=16, **kw)


def test_output_shape_is_one_logit_per_target():
    head = _head()
    x = torch.randn(4, N_SLOT, 32)
    mask = torch.ones(4, N_SLOT)
    assert head(x, mask).shape == (4, len(TARGETS))


def test_absent_slots_cannot_influence_the_output():
    """A masked slot must be ignored, not averaged in as a black image.

    This is the property the presence mask exists for: an absent acquisition is
    unknown, and a zero tensor is not unknown — the encoder maps it to a
    perfectly confident feature vector like any other input.
    """
    head = _head().eval()
    mask = torch.ones(1, N_SLOT)
    mask[0, 2] = 0.0

    x = torch.randn(1, N_SLOT, 32)
    with torch.no_grad():
        before = head(x, mask)
        x[0, 2] = torch.randn(32) * 100.0     # scribble on the masked slot
        after = head(x, mask)
    assert torch.allclose(before, after, atol=1e-5)


def test_an_unmasked_slot_does_influence_the_output():
    """The complement of the test above — otherwise it would pass on a no-op."""
    head = _head().eval()
    mask = torch.ones(1, N_SLOT)
    x = torch.randn(1, N_SLOT, 32)
    with torch.no_grad():
        before = head(x, mask)
        x[0, 2] = torch.randn(32) * 100.0
        after = head(x, mask)
    assert not torch.allclose(before, after, atol=1e-5)


def test_prior_tilts_attention_toward_the_named_slots():
    head = _head(prior=True)
    for target, slots in SLOT_PRIOR_TABLE.items():
        row = head.slot_prior[TARGETS.index(target)]
        for slot in slots:
            assert row[slot] == SLOT_PRIOR_STRENGTH
        others = [i for i in range(N_SLOT) if i not in slots]
        assert all(row[i] == 0.0 for i in others)


def test_prior_never_excludes_a_slot():
    """A tilt, not a mask. The model must be able to overrule the anatomy."""
    head = _head(prior=True).eval()
    x = torch.randn(2, N_SLOT, 32)
    mask = torch.ones(2, N_SLOT)
    h = head.proj(x) + head.slot_emb
    att = torch.einsum("bsh,oh->bos", h, head.query) / head.hidden ** 0.5
    att = (att + head.slot_prior.unsqueeze(0)).softmax(-1)
    assert (att > 0).all(), "every slot must keep non-zero attention weight"
    assert torch.allclose(att.sum(-1), torch.ones_like(att.sum(-1)), atol=1e-5)


def test_prior_can_be_switched_off_and_is_then_flat():
    head = _head(prior=False)
    assert torch.count_nonzero(head.slot_prior) == 0


def test_prior_travels_in_the_state_dict():
    """A buffer, not a constant.

    A checkpoint trained with the tilt and rebuilt without it loads cleanly with
    every shape matching and is a different model. Keeping it in the state dict
    is what makes that mismatch impossible rather than merely unlikely.
    """
    assert "slot_prior" in _head(prior=True).state_dict()


def test_every_target_has_a_prior_entry():
    """A target missing from the table would silently get a flat prior."""
    assert set(SLOT_PRIOR_TABLE) == set(TARGETS)


def test_prior_indices_are_in_range():
    for target, slots in SLOT_PRIOR_TABLE.items():
        assert slots, f"{target} has an empty slot tuple"
        assert all(0 <= s < N_SLOT for s in slots), target


def test_attention_is_per_diagnosis_not_shared():
    """Two diagnoses read on different slots must not pool identically."""
    head = _head(prior=True).eval()
    x = torch.randn(1, N_SLOT, 32)
    mask = torch.ones(1, N_SLOT)
    h = head.proj(x) + head.slot_emb
    att = torch.einsum("bsh,oh->bos", h, head.query) / head.hidden ** 0.5
    att = (att + head.slot_prior.unsqueeze(0)).softmax(-1)
    bakers = att[0, TARGETS.index("Baker's")]
    mcl = att[0, TARGETS.index("MCL")]
    assert not torch.allclose(bakers, mcl, atol=1e-3)
    # Baker's is a sagittal fluid-sensitive finding; MCL is coronal.
    assert bakers.argmax().item() == 0
    assert mcl.argmax().item() in (1, 4)


def test_gradients_reach_the_head():
    head = _head()
    out = head(torch.randn(2, N_SLOT, 32), torch.ones(2, N_SLOT))
    out.sum().backward()
    assert head.query.grad is not None
    assert np.isfinite(head.query.grad.numpy()).all()
