"""Physical-scale cropping.

The corpus mixes scanners with different `PixelSpacing`, so a fixed-pixel
resize feeds the encoder the same anatomy at different scales — and the
variation tracks the site, which is the nuisance grouped CV exists to punish.
These tests pin the property that matters: one output pixel is the same number
of millimetres in every study.
"""

import numpy as np

from src.data.dicom import FOV_MM, physical_crop


class _FakeDs:
    """Minimal stand-in for a pydicom dataset: only PixelSpacing is read."""

    def __init__(self, spacing):
        if spacing is not None:
            self.PixelSpacing = spacing


def test_physical_crop_equalises_anatomical_scale():
    """Two scanners, same 160 mm of knee, different mm/px -> same physical extent."""
    fine = physical_crop(np.zeros((640, 640), np.float32), _FakeDs([0.25, 0.25]))
    coarse = physical_crop(np.zeros((320, 320), np.float32), _FakeDs([0.50, 0.50]))

    assert fine.shape == (int(FOV_MM / 0.25),) * 2
    assert coarse.shape == (int(FOV_MM / 0.50),) * 2
    # Different pixel counts, identical millimetres — so after the downstream
    # resize to a common size they agree on mm per output pixel.
    assert fine.shape[0] * 0.25 == coarse.shape[0] * 0.50 == FOV_MM


def test_physical_crop_is_centred():
    frame = np.zeros((100, 100), np.float32)
    frame[48:52, 48:52] = 1.0  # marker at the centre

    out = physical_crop(frame, _FakeDs([2.0, 2.0]))  # 160 mm / 2.0 = 80 px

    assert out.shape == (80, 80)
    assert out.sum() == 16.0  # nothing of the marker was cropped away
    assert out[38:42, 38:42].sum() == 16.0  # and it is still centred


def test_physical_crop_falls_back_rather_than_dropping_the_slice():
    """Degrading to the old behaviour beats losing the slice."""
    frame = np.ones((64, 64), np.float32)

    assert physical_crop(frame, _FakeDs(None)).shape == (64, 64)  # no spacing tag
    assert physical_crop(frame, _FakeDs([0.0, 0.0])).shape == (64, 64)  # zero spacing
    assert physical_crop(frame, _FakeDs(["x", "y"])).shape == (64, 64)  # unparseable
    assert physical_crop(frame, _FakeDs([0.1, 0.1])).shape == (64, 64)  # FOV > image


def test_anisotropic_spacing_crops_each_axis_independently():
    """Row and column spacing differ on some series; each axis uses its own."""
    out = physical_crop(np.zeros((400, 400), np.float32), _FakeDs([0.5, 0.8]))

    assert out.shape == (int(FOV_MM / 0.5), int(FOV_MM / 0.8))
