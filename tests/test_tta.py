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
