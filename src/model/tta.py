"""Translation test-time augmentation, shared by evaluation and submission.

docs/15 step 4. One predeclared, modest view set - not a search space:
identity plus four axis-aligned shifts of TTA_SHIFT in `affine_grid`
normalised units (full image width = 2, so 0.03 is ~5 px at 336 px). That
is 60% of the +/-0.05 the training `augment()` already draws uniformly, so
every view is inside the distribution the models were trained on.

What is deliberately NOT here: no flips (laterality normalisation exists to
remove exactly that axis, and a vertical flip destroys anatomy position -
see `augment()`'s own docstring), no rotation, no zoom, no intensity gain.
Every image in a study is shifted by the SAME (tx, ty), so slot assignment
is untouched, and the channel axis (consecutive slices) is never permuted or
mixed - `grid_sample` is per-channel spatial resampling.

The evaluator and any submission must import this module rather than
re-implement the transform: "identical transformations in evaluation and
submission" is only checkable if there is one implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Mirrors AUG_SHIFT in notebooks/kaggle_06_train_slots.py (asserted in the
#: evaluator against the live training-script value, and in tests/test_tta.py
#: against this constant).
TRAIN_AUG_SHIFT = 0.05
TTA_SHIFT = 0.03

#: (tx, ty) per view. Identity first: predictions for the identity view must
#: be bit-identical to the plain, un-augmented predict() path.
TTA_VIEWS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (TTA_SHIFT, 0.0),
    (-TTA_SHIFT, 0.0),
    (0.0, TTA_SHIFT),
    (0.0, -TTA_SHIFT),
)
#: Fixed, predeclared, applied to per-view group-averaged LOGITS before the
#: sigmoid - the same order predict() already uses across slice groups.
TTA_WEIGHTS: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)


def translate_images(x: torch.Tensor, tx: float, ty: float) -> torch.Tensor:
    """Shift every image in `x` (..., C, H, W) by (tx, ty), border-padded.

    (0, 0) returns `x` itself, untouched (dtype included), so the identity
    view goes through exactly the code path the baseline uses. Any other
    shift returns float32 clamped to [0, 255], the same range and dtype the
    training augmentation feeds the model.
    """
    if tx == 0.0 and ty == 0.0:
        return x
    lead = x.shape[:-3]
    flat = x.reshape(-1, *x.shape[-3:]).float()
    n = flat.shape[0]
    theta = torch.zeros(n, 2, 3, device=flat.device, dtype=flat.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    # `border` padding, as in training: zooming/shifting must not fabricate a
    # black band where a popliteal fossa is looked for.
    out = F.grid_sample(flat, grid, mode="bilinear", padding_mode="border",
                        align_corners=False)
    return out.clamp(0, 255).reshape(*lead, *out.shape[-3:])
