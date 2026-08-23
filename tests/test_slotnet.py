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


# --- cross-attention head -------------------------------------------------
#
# The head SlotHead's pooling was the bottleneck four experiments ran into:
# capacity (base -0.0005) and resolution (448 -0.0014) could not help while
# every slot was collapsed to one vector before any finding-specific reasoning.
# These pin the properties that make the replacement worth having.

from src.model.slotnet import (  # noqa: E402
    GROUP_NAMES,
    GROUP_OF_TARGET,
    TARGET_GROUPS,
    XAttnHead,
)


def _xhead(**kw):
    torch.manual_seed(0)
    return XAttnHead(dim=32, pooled_dim=96, hidden=16, heads=4, **kw)


def test_xattn_output_shape():
    head = _xhead()
    out = head(torch.randn(2, N_SLOT, 17, 32), torch.randn(2, N_SLOT, 96),
               torch.ones(2, N_SLOT))
    assert out.shape == (2, len(TARGETS))


def test_xattn_ignores_masked_slots_in_both_paths():
    """A masked slot must reach neither the tokens nor the pooled path.

    ~25% of slots are absent in this corpus, so this is a quarter of the input,
    not an edge case.
    """
    head = _xhead().eval()
    tokens = torch.randn(1, N_SLOT, 17, 32)
    pooled = torch.randn(1, N_SLOT, 96)
    mask = torch.ones(1, N_SLOT)
    mask[0, 2] = 0.0
    with torch.no_grad():
        before = head(tokens, pooled, mask)
        tokens[0, 2] = torch.randn(17, 32) * 100.0
        pooled[0, 2] = torch.randn(96) * 100.0
        after = head(tokens, pooled, mask)
    assert torch.allclose(before, after, atol=1e-5)


def test_xattn_sees_detail_a_mean_pool_cannot():
    """A change that preserves the slot MEAN must still change the output.

    This is the whole point of the head, stated precisely. Concentrating signal
    into one token while compensating across the others leaves every mean-based
    summary identical — so `SlotHead`, which only ever sees such summaries,
    cannot react. The cross-attention path reads the tokens and must.

    Note what this does NOT claim: attention over a key sequence is
    permutation-invariant, so shuffling tokens is correctly a no-op. Position
    is carried in the token VALUES (DINOv2 adds its positional embeddings
    inside the encoder), not in their order.
    """
    head = _xhead().eval()
    tokens = torch.randn(1, N_SLOT, 17, 32)
    pooled = torch.randn(1, N_SLOT, 96)
    mask = torch.ones(1, N_SLOT)

    spiked = tokens.clone()
    delta = torch.randn(32) * 5.0
    spiked[0, 0, 0] += delta                       # concentrate into one token
    spiked[0, 0, 1:] -= delta / (17 - 1)           # ... and hold the mean fixed

    assert torch.allclose(tokens[0, 0].mean(0), spiked[0, 0].mean(0), atol=1e-5)
    with torch.no_grad():
        before = head(tokens, pooled, mask)
        after = head(spiked, pooled, mask)
    assert not torch.allclose(before, after, atol=1e-5)


def test_every_target_belongs_to_exactly_one_group():
    seen = [t for group in TARGET_GROUPS.values() for t in group]
    assert sorted(seen) == sorted(TARGETS), "targets must partition into groups"
    assert len(seen) == len(set(seen)), "a target cannot be in two groups"


def test_group_ids_are_in_range_and_match_the_table():
    assert len(GROUP_OF_TARGET) == len(TARGETS)
    for target, gid in zip(TARGETS, GROUP_OF_TARGET):
        assert 0 <= gid < len(GROUP_NAMES)
        assert target in TARGET_GROUPS[GROUP_NAMES[gid]]


def test_related_findings_share_a_group():
    """Medial and lateral meniscus are read the same way; so are the OA labels."""
    gid = dict(zip(TARGETS, GROUP_OF_TARGET))
    assert gid["Medial Meniscus"] == gid["Lateral Meniscus"]
    assert gid["Medial OA"] == gid["Lateral OA"] == gid["PF OA"]
    assert gid["ACL"] == gid["MCL"]
    assert gid["Medial Meniscus"] != gid["ACL"]
