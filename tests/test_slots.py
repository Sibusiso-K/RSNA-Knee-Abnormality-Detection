"""Slot routing and laterality normalisation.

Laterality is the defect these tests exist for. Five of the twelve targets are
side-specific, and a mirror applied in the wrong direction — or on the wrong
axis — is worse than none at all: it produces confidently mislabelled anatomy
rather than merely inconsistent anatomy. The sign convention and the per-plane
axis choice are therefore pinned here rather than checked by eye on Kaggle.
"""

import numpy as np

from src.data.slots import (
    CROP_MM,
    GROUP,
    IMG,
    N_GROUP,
    SLOTS,
    SeriesInfo,
    band_indices,
    classify_weighting,
    normalise_intensity,
    normalise_laterality,
    physical_crop,
    pick_slot_series,
    side_from_geometry,
    study_side,
)


def _info(uid, plane, fluid, fatsat, n_files=30, side="R", pixel_mm=0.4):
    return SeriesInfo(uid, plane, fluid, fatsat, n_files, side, pixel_mm)


# --- weighting -----------------------------------------------------------


def test_t1_by_name_is_not_fluid_sensitive():
    fluid, fatsat = classify_weighting("AX T1 TSE", None, None)
    assert fluid is False and fatsat is False


def test_stir_is_fluid_sensitive_and_fat_suppressed():
    fluid, fatsat = classify_weighting("COR STIR", None, None)
    assert fluid is True and fatsat is True


def test_stir_beats_a_contradictory_t1_in_the_same_string():
    """A header naming both is naming the preparation, not the weighting."""
    fluid, _ = classify_weighting("T1 TIRM", None, None)
    assert fluid is True


def test_pd_fatsat_is_fluid_sensitive():
    fluid, fatsat = classify_weighting("SAG PD FS", None, None)
    assert fluid is True and fatsat is True


def test_tr_te_fallback_separates_t1_from_t2():
    assert classify_weighting("", 500.0, 12.0)[0] is False     # short TR/TE -> T1
    assert classify_weighting("", 3500.0, 80.0)[0] is True     # long  TR/TE -> T2
    assert classify_weighting("", 2500.0, 25.0)[0] is True     # long TR short TE -> PD


def test_unknown_weighting_defaults_to_fluid_sensitive():
    """The conservative error: an empty fluid slot loses the finding outright."""
    assert classify_weighting("", None, None)[0] is True


# --- laterality ----------------------------------------------------------


def test_side_from_geometry_uses_centre_not_corner():
    """A right knee whose FIRST VOXEL is on the +x side still reads as right.

    This is the whole reason the centre is computed. ImagePositionPatient is the
    corner; for a knee it sits ~half a field of view from the centre, which is
    more than enough to cross the midline and flip the sign.
    """
    # Row direction is -x over 320 px at 0.4 mm, so the image spans 128 mm
    # leftward from the corner. Corner x = +40 would read LEFT; the centre is
    # 40 - 64 = -24 mm, which correctly reads RIGHT and clears the dead zone.
    side = side_from_geometry(
        orientation=[-1, 0, 0, 0, 1, 0],
        position=[40.0, 0.0, 0.0],
        rows=320, cols=320, pixel_mm=0.4,
    )
    assert side == "R"


def test_side_from_geometry_positive_x_is_left():
    side = side_from_geometry(
        orientation=[1, 0, 0, 0, 1, 0],
        position=[40.0, 0.0, 0.0],
        rows=320, cols=320, pixel_mm=0.4,
    )
    assert side == "L"


def test_side_from_geometry_returns_none_near_the_midline():
    """Inside the dead zone the reading is not trustworthy; None beats a guess."""
    side = side_from_geometry([1, 0, 0, 0, 1, 0], [0.0, 0.0, 0.0], 10, 10, 0.4)
    assert side is None


def test_side_from_geometry_tolerates_missing_geometry():
    assert side_from_geometry(None, None, 0, 0, None) is None
    assert side_from_geometry([1, 0, 0], [0, 0, 0], 10, 10, 0.4) is None


def test_right_knee_is_never_touched():
    volume = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    for plane in ("Sagittal", "Coronal", "Axial"):
        assert np.array_equal(normalise_laterality(volume, plane, "R"), volume)


def test_unknown_side_is_never_touched():
    """A wrong mirror is misleading; an unmirrored study is merely inconsistent."""
    volume = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    assert np.array_equal(normalise_laterality(volume, "Coronal", None), volume)


def test_left_coronal_flips_the_image_horizontally():
    """Coronal/axial carry medial-lateral ACROSS the frame."""
    volume = np.arange(1 * 2 * 4, dtype=np.uint8).reshape(1, 2, 4)
    out = normalise_laterality(volume, "Coronal", "L")
    assert np.array_equal(out, volume[:, :, ::-1])
    # The stack order must NOT change for coronal.
    assert out.shape == volume.shape


def test_left_sagittal_reverses_slice_order_and_leaves_frames_alone():
    """Sagittal carries medial-lateral ALONG the stack, not across the frame.

    Flipping a sagittal frame horizontally would mirror anterior-posterior
    instead — relabelling nothing, but moving the patella to the back of the
    knee and the popliteal fossa to the front.
    """
    volume = np.arange(3 * 2 * 2, dtype=np.uint8).reshape(3, 2, 2)
    out = normalise_laterality(volume, "Sagittal", "L")
    assert np.array_equal(out, volume[::-1])
    for i in range(3):
        assert np.array_equal(out[i], volume[2 - i])


def test_study_side_is_a_majority_vote():
    infos = [_info("a", "Sagittal", True, True, side="R"),
             _info("b", "Coronal", True, True, side="R"),
             _info("c", "Axial", True, True, side="L")]
    assert study_side(infos) == "R"


def test_study_side_tie_returns_none():
    infos = [_info("a", "Sagittal", True, True, side="R"),
             _info("b", "Coronal", True, True, side="L")]
    assert study_side(infos) is None


def test_study_side_none_when_no_geometry_anywhere():
    assert study_side([_info("a", "Sagittal", True, True, side=None)]) is None


# --- slot routing --------------------------------------------------------


def test_slots_route_by_plane_and_weighting():
    infos = [
        _info("sag_fs", "Sagittal", True, True),
        _info("cor_fs", "Coronal", True, True),
        _info("cor_t1", "Coronal", False, False),
    ]
    picked = pick_slot_series(infos)
    names = [s[0] for s in SLOTS]
    assert picked[names.index("SAG_FLUID_FS")].uid == "sag_fs"
    assert picked[names.index("COR_FLUID_FS")].uid == "cor_fs"
    assert picked[names.index("COR_T1")].uid == "cor_t1"
    assert picked[names.index("AX_FLUID_FS")] is None


def test_slot_prefers_the_series_with_most_slices():
    """Slice count is the cheap proxy for 'covers the joint' vs 'localiser'."""
    infos = [
        _info("short", "Sagittal", True, True, n_files=3),
        _info("long", "Sagittal", True, True, n_files=32),
    ]
    picked = pick_slot_series(infos)
    assert picked[0].uid == "long"


def test_no_cross_slot_fallback():
    """A fluid coronal must not be used to fill COR_T1.

    Filling it would put a different tissue contrast behind the same slot
    embedding. The presence mask exists so absence can be declared.
    """
    picked = pick_slot_series([_info("cor_fs", "Coronal", True, True)])
    names = [s[0] for s in SLOTS]
    assert picked[names.index("COR_T1")] is None


# --- sampling and scale --------------------------------------------------


def test_band_indices_stay_inside_the_central_band():
    idx = band_indices(100, 3)
    assert idx.min() >= 19 and idx.max() <= 80
    assert len(idx) == 3


def test_band_indices_survive_a_very_short_series():
    idx = band_indices(4, 3)
    assert len(idx) == 3 and idx.min() >= 0 and idx.max() <= 3


def test_band_indices_handle_an_empty_series():
    assert len(band_indices(0, 3)) == 3


def test_physical_crop_equalises_anatomical_scale():
    fine, ok_fine = physical_crop(np.zeros((640, 640), np.float32), 0.25)
    coarse, ok_coarse = physical_crop(np.zeros((320, 320), np.float32), 0.50)
    assert ok_fine and ok_coarse
    assert fine.shape[0] == round(CROP_MM / 0.25)
    assert coarse.shape[0] == round(CROP_MM / 0.50)
    # Same physical extent, so after the resize one pixel means the same mm.
    assert fine.shape[0] * 0.25 == coarse.shape[0] * 0.50


def test_physical_crop_reports_rather_than_hides_a_fallback():
    """A crop that does not fit must be visible to the caller, not silent."""
    _frame, applied = physical_crop(np.zeros((64, 64), np.float32), 0.5)
    assert applied is False
    _frame, applied = physical_crop(np.zeros((64, 64), np.float32), None)
    assert applied is False


def test_normalise_intensity_returns_uint8_full_range():
    volume = np.linspace(0, 1000, 3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    out = normalise_intensity(volume)
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255


def test_normalise_intensity_handles_a_constant_volume():
    out = normalise_intensity(np.full((3, 4, 4), 7.0, dtype=np.float32))
    assert out.dtype == np.uint8 and out.max() == 0


def test_cache_geometry_is_dinov2_compatible():
    """336 = 14 x 24. A patch-14 ViT cannot take 256, and silently misaligns."""
    assert IMG % 14 == 0
    assert GROUP == 3, "three slices per encoder input = the RGB channels"
    assert N_GROUP >= 1
