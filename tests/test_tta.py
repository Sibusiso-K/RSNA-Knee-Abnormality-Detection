"""Translation TTA: the transform must move pixels and nothing else.

Laterality, slot assignment and slice ordering are load-bearing in this
pipeline (five of twelve targets are side-specific; slots are plane x
sequence bins) so the tests assert what is NOT allowed to change as firmly as
what is.
"""

import torch

from src.model.tta import (
    TRAIN_AUG_SHIFT, TTA_SHIFT, TTA_VIEWS, TTA_WEIGHTS, translate_images,
)

H = W = 64


def _dot(row, col, value=200.0):
    img = torch.zeros(H, W)
    img[row, col] = value
    return img


def _peak(img):
    flat = img.flatten().argmax().item()
    return divmod(flat, W)  # (row, col)


def test_view_set_is_predeclared_and_within_training_range():
    assert len(TTA_VIEWS) == len(TTA_WEIGHTS) == 5
    assert TTA_VIEWS[0] == (0.0, 0.0), "identity must be first"
    assert abs(sum(TTA_WEIGHTS) - 1.0) < 1e-12
    assert all(abs(tx) <= TRAIN_AUG_SHIFT and abs(ty) <= TRAIN_AUG_SHIFT for tx, ty in TTA_VIEWS)
    assert TTA_SHIFT <= TRAIN_AUG_SHIFT


def test_identity_returns_the_same_tensor_untouched():
    x = torch.randint(0, 255, (2, 6, 3, H, W), dtype=torch.uint8)
    assert translate_images(x, 0.0, 0.0) is x


def test_shift_moves_pixels_by_the_expected_amount_along_one_axis_only():
    x = _dot(32, 32).expand(1, 3, H, W).contiguous()      # (S=1, C=3, H, W)
    expect = TTA_SHIFT * W / 2                             # ~0.96 px at 64 px
    # Use a larger shift than TTA_SHIFT so the move is measurable at 64 px.
    tx = 0.25
    out = translate_images(x, tx, 0.0)
    row, col = _peak(out[0, 0])
    assert row == 32                                       # no vertical motion
    assert abs(abs(col - 32) - tx * W / 2) <= 1            # 8 px horizontally
    out = translate_images(x, 0.0, tx)
    row, col = _peak(out[0, 0])
    assert col == 32
    assert abs(abs(row - 32) - tx * W / 2) <= 1
    assert expect < 1.5                                    # the real set is sub-pixel-scale at this size


def test_no_flip_an_off_centre_feature_stays_on_its_side():
    x = _dot(10, 50).expand(1, 3, H, W).contiguous()      # top-right feature
    for tx, ty in TTA_VIEWS:
        row, col = _peak(translate_images(x, tx, ty)[0, 0])
        assert row < H // 2 and col > W // 2, (tx, ty, row, col)


def test_slots_and_channels_are_shifted_identically_and_never_mixed():
    # (B=1, S=3 slots, C=3 slices): every (slot, slice) gets a distinct dot.
    x = torch.zeros(1, 3, 3, H, W)
    for s in range(3):
        for c in range(3):
            x[0, s, c, 20 + 8 * s, 20 + 8 * c] = 100.0 + 10 * (3 * s + c)
    tx, ty = 0.25, 0.25
    out = translate_images(x, tx, ty)
    assert out.shape == x.shape
    shifts = set()
    for s in range(3):
        for c in range(3):
            r0, c0 = 20 + 8 * s, 20 + 8 * c
            r1, c1 = _peak(out[0, s, c])
            shifts.add((r1 - r0, c1 - c0))
            # each output image contains only its own source's energy
            assert out[0, s, c].sum() > 0
            assert out[0, s, c].max() <= x[0, s, c].max() + 1e-4
    assert len(shifts) == 1, f"slots/slices shifted differently: {shifts}"


def test_output_is_float_in_pixel_range_and_border_padded_not_black():
    x = torch.full((1, 3, H, W), 180, dtype=torch.uint8)
    out = translate_images(x, TTA_SHIFT, -TTA_SHIFT)
    assert out.dtype == torch.float32
    assert out.min() >= 0 and out.max() <= 255
    assert torch.allclose(out, torch.full_like(out, 180.0), atol=1e-3)


def test_opposite_shifts_are_mirror_images_of_a_symmetric_input():
    x = torch.zeros(1, 3, H, W)
    x[..., 32, 24:40] = 200.0                              # symmetric about col 31.5
    a = translate_images(x, 0.25, 0.0)
    b = translate_images(x, -0.25, 0.0)
    assert torch.allclose(a.flip(-1), b, atol=1e-3)


# ---- tta_probs: the exact averaging order the submission must follow --------

import pytest

import src.model.tta as tta_module
from src.model.tta import tta_probs

_G = 32


class _FakeNet:
    """Position-sensitive stand-in for SlotNet: logits depend on WHERE things
    are, so a shifted view genuinely changes them, and large enough that
    sigmoid-before-averaging differs measurably from averaging logits first."""

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.k = torch.randn(12, _G, _G, generator=g) * 0.02
        self.calls = 0

    def __call__(self, x, mask):
        self.calls += 1
        img = x.float().mean(dim=(1, 2)) / 255.0                # (B, H, W)
        return (img[:, None] * self.k[None]).sum(dim=(-1, -2)) * 4.0 - 1.0


def _batch(seed=0, slices=6):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 255, (2, 3, slices, _G, _G), generator=g, dtype=torch.uint8)
    # a smooth ramp so 1-pixel shifts change the logits by a real amount
    x = (x.float() * 0.2 + torch.linspace(0, 200, _G)[None, None, None, None, :]).clamp(0, 255)
    return x.to(torch.uint8), torch.ones(2, 3)


def _old_path(net, xk, m, group=3):
    """What kaggle_08 did before TTA: average logits over groups, sigmoid."""
    n = xk.shape[2] // group
    acc = None
    for g in range(n):
        lg = net(xk[:, :, g * group:(g + 1) * group], m).float()
        acc = lg if acc is None else acc + lg
    return torch.sigmoid(acc / n)


def test_identity_only_weights_reproduce_the_pre_tta_submission_path_exactly():
    net, (xk, m) = _FakeNet(), _batch()
    got = tta_probs(net, xk, m, 3, views=TTA_VIEWS, weights=(1.0, 0.0, 0.0, 0.0, 0.0))
    assert torch.equal(got, _old_path(net, xk, m))


def test_identity_output_is_the_pre_tta_path_even_under_full_tta():
    net, (xk, m) = _FakeNet(), _batch(1)
    _, ident = tta_probs(net, xk, m, 3, return_identity=True)
    assert torch.equal(ident, _old_path(net, xk, m))


def test_order_is_group_mean_then_view_mean_of_logits_then_sigmoid():
    net, (xk, m) = _FakeNet(), _batch(2)
    big = ((0.0, 0.0), (0.3, 0.0), (-0.3, 0.0), (0.0, 0.3), (0.0, -0.3))
    got = tta_probs(net, xk, m, 3, views=big)
    per_view = []
    for tx, ty in big:
        vlog = []
        for g in range(2):
            xg = translate_images(xk[:, :, g * 3:(g + 1) * 3], tx, ty)
            vlog.append(net(xg, m).float())
        per_view.append(sum(vlog) / 2)
    want = torch.sigmoid(sum(w * v for w, v in zip(TTA_WEIGHTS, per_view)))
    assert torch.allclose(got, want, atol=1e-6)
    # ...and NOT the other order (probability-average the views), which would
    # silently give a different, un-evaluated prediction.
    # Guard the test itself: the toy net must actually be view-sensitive,
    # otherwise the two orders coincide and this assertion proves nothing.
    assert max((v - per_view[0]).abs().max() for v in per_view[1:]) > 0.3
    wrong = sum(w * torch.sigmoid(v) for w, v in zip(TTA_WEIGHTS, per_view))
    assert (got - wrong).abs().max() > 1e-3


def test_view_cache_translates_once_and_gives_identical_results(monkeypatch):
    net, (xk, m) = _FakeNet(), _batch(3)
    calls = {"n": 0}
    real = tta_module.translate_images

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(tta_module, "translate_images", counting)
    cache = {}
    first = tta_probs(net, xk, m, 3, view_cache=cache)
    n_first = calls["n"]
    second = tta_probs(net, xk, m, 3, view_cache=cache)      # a second "member"
    assert n_first == 2 * len(TTA_VIEWS)                     # 2 groups x 5 views
    assert calls["n"] == n_first                             # second member: no re-translation
    assert torch.equal(first, second)
    assert torch.equal(first, tta_probs(net, xk, m, 3))      # cache never changes the numbers


def test_forward_cost_is_one_pass_per_group_per_view():
    net, (xk, m) = _FakeNet(), _batch(4)
    tta_probs(net, xk, m, 3)
    assert net.calls == 2 * len(TTA_VIEWS)


def test_rejects_weights_that_do_not_sum_to_one_or_bad_group_size():
    net, (xk, m) = _FakeNet(), _batch(5)
    with pytest.raises(ValueError):
        tta_probs(net, xk, m, 3, weights=(0.3, 0.2, 0.2, 0.2, 0.2))
    with pytest.raises(ValueError):
        tta_probs(net, xk, m, 4)                             # 6 slices is not a multiple of 4


class _MutatingNet(_FakeNet):
    """Like SlotNet.forward: `.float().div_(255)` mutates a float32 input in place."""

    def __call__(self, x, mask):
        self.calls += 1
        img = x.float().div_(255.0).mean(dim=(1, 2))
        return (img[:, None] * self.k[None]).sum(dim=(-1, -2)) * 4.0 * 255.0 - 1.0


def test_members_sharing_a_view_cache_are_not_corrupted_by_in_place_forwards():
    (xk, m) = _batch(6)
    fresh = tta_probs(_MutatingNet(), xk, m, 3)
    cache = {}
    first = tta_probs(_MutatingNet(), xk, m, 3, view_cache=cache)
    later = tta_probs(_MutatingNet(), xk, m, 3, view_cache=cache)   # 2nd..10th member
    assert torch.equal(first, fresh)
    assert torch.equal(later, fresh)


def test_translation_ignores_the_callers_autocast_context():
    net, (xk, m) = _FakeNet(), _batch(7)
    plain = tta_probs(net, xk, m, 3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        inside = tta_probs(net, xk, m, 3)
    assert torch.allclose(plain, inside, atol=1e-6)
