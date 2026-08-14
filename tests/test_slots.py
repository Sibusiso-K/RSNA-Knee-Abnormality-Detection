"""Tests for the slot pipeline — the code that actually produced 0.856.

`src/data/slots.py` and `src/model/slotnet.py` had zero coverage until now,
which is uncomfortable given what they do: they are shared by cache building
*and* inference, so a change that shifts either one silently reintroduces
train/test preprocessing skew — score well in CV, badly on the leaderboard.

Three properties here are load-bearing enough to deserve naming:

1. **`SLOTS` ordering.** `SLOT_PRIOR_TABLE` indexes into it positionally and a
   published cache is written under it. Reordering the tuple silently
   reinterprets every cached array and every prior — no error, just wrong.
2. **The laterality mirror.** Sagittal reverses SLICE ORDER; coronal and axial
   flip the FRAME. Getting these the wrong way round mirrors
   anterior-posterior instead of medial-lateral, which relabels nothing but
   destroys patella position. A wrong mirror is worse than no mirror.
3. **Mask exclusion.** An absent slot must leave the logits untouched. If
   masking ever degrades to feeding zeros, the encoder maps a black image to a
   confident feature and the model learns from an acquisition that never
   happened.

No network and no GPU: the heads are plain torch, and `SlotNet` itself is not
constructed here because it would download DINOv2 weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.slots import (  # noqa: E402
    CROP_MM,
    GROUP,
    IMG,
    N_SLOT,
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
from src.labels import TARGETS  # noqa: E402
from src.model.slotnet import (  # noqa: E402
    GROUP_OF_TARGET,
    SLOT_PRIOR_STRENGTH,
    SLOT_PRIOR_TABLE,
    TARGET_GROUPS,
    SlotHead,
    XAttnHead,
)

# --------------------------------------------------------------------------
# 1. The frozen contract: ordering and geometry constants
# --------------------------------------------------------------------------


def test_slot_order_is_frozen():
    """Pinned deliberately. A cache written under one ordering cannot be read
    under another, and `SLOT_PRIOR_TABLE` indexes into this list positionally.
    If this test fails, every published cache and checkpoint is invalidated —
    that is the point of the test, not a reason to update the expected value.
    """
    assert [s[0] for s in SLOTS] == [
        "SAG_FLUID_FS",
        "COR_FLUID_FS",
        "AX_FLUID_FS",
        "SAG_FLUID_NOFS",
        "COR_T1",
        "SAG_T1",
    ]
    assert N_SLOT == 6


def test_slot_definitions_match_their_names():
    """The name is documentation; the tuple is what routing actually uses.
    They drift apart silently, so assert they agree."""
    for name, plane, fluid, fatsat in SLOTS:
        assert plane in ("Sagittal", "Coronal", "Axial")
        assert name.startswith({"Sagittal": "SAG", "Coronal": "COR", "Axial": "AX"}[plane])
        if "T1" in name:
            assert fluid is False
        if name.endswith("_FS"):
            assert fatsat is True
        if name.endswith("NOFS"):
            assert fatsat is False


def test_image_side_is_a_multiple_of_the_dinov2_patch():
    """336 = 14x24. The module docstring calls out 256 (18.29) as the trap."""
    assert IMG % 14 == 0
    assert 256 % 14 != 0, "sanity: 256 really is the invalid case the docs warn about"


def test_pixel_spacing_is_what_the_docs_claim():
    """130 mm / 336 px = 0.387 mm/px. The 160 mm / 224 px alternative is
    0.714 mm/px — nearly 2x coarser, and a 1-3 mm meniscal tear is 1-4 px
    there. Guards against someone 'tidying' CROP_MM back to dicom.py's 160."""
    assert CROP_MM == 130.0
    assert round(CROP_MM / IMG, 3) == 0.387


def test_group_is_the_rgb_triplet():
    assert GROUP == 3, "the encoder takes three channels; 2.5D depends on it"


# --------------------------------------------------------------------------
# 2. Laterality — the mirror that fixes five of twelve targets
# --------------------------------------------------------------------------


def _stack():
    """(4, 2, 3) volume where every slice and every column is distinguishable."""
    return np.arange(4 * 2 * 3, dtype=np.uint8).reshape(4, 2, 3)


def test_right_knee_is_never_touched():
    volume = _stack()
    assert np.array_equal(normalise_laterality(volume, "Sagittal", "R"), volume)
    assert np.array_equal(normalise_laterality(volume, "Coronal", "R"), volume)


def test_unknown_side_is_left_alone_rather_than_guessed():
    """A wrong mirror is an actively misleading example; an unmirrored one is
    merely one the model has to tolerate."""
    volume = _stack()
    assert np.array_equal(normalise_laterality(volume, "Coronal", None), volume)


def test_sagittal_left_reverses_slice_order_and_leaves_frames_alone():
    """On sagittal the medial-lateral axis IS the stack axis."""
    volume = _stack()
    out = normalise_laterality(volume, "Sagittal", "L")
    assert np.array_equal(out, volume[::-1])
    for i in range(volume.shape[0]):
        assert np.array_equal(out[i], volume[volume.shape[0] - 1 - i]), "frame was altered"


@pytest.mark.parametrize("plane", ["Coronal", "Axial"])
def test_coronal_and_axial_left_flip_the_frame_and_keep_slice_order(plane):
    """On these planes medial-lateral runs across the image."""
    volume = _stack()
    out = normalise_laterality(volume, plane, "L")
    assert np.array_equal(out, volume[:, :, ::-1])
    # Slice order preserved: slice i still derives from slice i.
    for i in range(volume.shape[0]):
        assert np.array_equal(out[i], volume[i][:, ::-1])


def test_the_two_planes_do_not_apply_the_same_transform():
    """The whole reason this cannot be a single flip. If a refactor ever makes
    sagittal and coronal agree, the mirror has become wrong for one of them."""
    volume = _stack()
    assert not np.array_equal(
        normalise_laterality(volume, "Sagittal", "L"),
        normalise_laterality(volume, "Coronal", "L"),
    )


def test_mirroring_twice_returns_the_original():
    """Both branches must be involutions, or repeated cache builds drift."""
    volume = _stack()
    for plane in ("Sagittal", "Coronal", "Axial"):
        once = normalise_laterality(volume, plane, "L")
        twice = normalise_laterality(once, plane, "L")
        assert np.array_equal(twice, volume), plane


# --------------------------------------------------------------------------
# 3. Side from geometry — the "centre, not corner" bug
# --------------------------------------------------------------------------

# row_dir = +x, col_dir = +y
_ORIENT = [1, 0, 0, 0, 1, 0]


def test_positive_x_is_left_and_negative_x_is_right():
    """+x is the patient's LEFT in DICOM patient space."""
    assert side_from_geometry(_ORIENT, [50, 0, 0], rows=2, cols=2, pixel_mm=1.0) == "L"
    assert side_from_geometry(_ORIENT, [-50, 0, 0], rows=2, cols=2, pixel_mm=1.0) == "R"


def test_side_is_read_from_the_image_centre_not_the_corner():
    """ImagePositionPatient is the centre of the FIRST VOXEL — a corner. For a
    knee, corner and centre differ by ~half a field of view, which is enough to
    land on the wrong sign. Here the corner sits at x=-50 (would read 'R') while
    the true centre is x=+50 ('L'). A regression to corner-based reading flips
    this assertion."""
    side = side_from_geometry(_ORIENT, [-50, 0, 0], rows=200, cols=200, pixel_mm=1.0)
    assert side == "L"


def test_midline_dead_zone_returns_none():
    """Within 10 mm of the midline the reading is not trustworthy."""
    assert side_from_geometry(_ORIENT, [5, 0, 0], rows=2, cols=2, pixel_mm=1.0) is None
    # A centre landing exactly on the midline.
    assert side_from_geometry(_ORIENT, [-100, 0, 0], rows=200, cols=200, pixel_mm=1.0) is None


@pytest.mark.parametrize(
    "orientation,position",
    [
        ([1, 0, 0], [50, 0, 0]),        # orientation too short
        (_ORIENT, [50, 0]),             # position too short
        (_ORIENT, ["x", 0, 0]),         # unparseable
        (None, [50, 0, 0]),             # missing entirely
    ],
)
def test_malformed_geometry_returns_none_rather_than_raising(orientation, position):
    """Headers in this corpus are genuinely incomplete; a raise here would kill
    a cache build hours in."""
    assert side_from_geometry(orientation, position, rows=2, cols=2, pixel_mm=1.0) is None


def test_study_side_takes_a_majority_vote():
    def info(side):
        return SeriesInfo(uid="u", plane="Coronal", fluid=True, fatsat=True,
                          n_files=10, side=side, pixel_mm=1.0)

    assert study_side([info("L"), info("L"), info("R")]) == "L"
    assert study_side([info("R"), info("R"), info("L")]) == "R"
    # A single bad per-series read does not flip a study.
    assert study_side([info("L"), info("L"), info("L"), info("R")]) == "L"


def test_study_side_returns_none_on_a_tie_or_no_votes():
    def info(side):
        return SeriesInfo(uid="u", plane="Coronal", fluid=True, fatsat=True,
                          n_files=10, side=side, pixel_mm=1.0)

    assert study_side([info("L"), info("R")]) is None
    assert study_side([info(None), info(None)]) is None
    assert study_side([]) is None


# --------------------------------------------------------------------------
# 4. Sequence weighting
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["AX T1 TSE", "T1", "T1W", "t1w sag", "COR T1W", "T1-weighted", "T1W/SE"]
)
def test_t1_text_is_structural(text):
    fluid, _ = classify_weighting(text, None, None)
    assert fluid is False


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN BUG, deliberately not fixed here. `_T1_RX` uses \\b, and underscore "
        "is a word character in Python regex (\\w == [A-Za-z0-9_]), so \\bt1w\\b "
        "cannot match 't1w_sag' — there is no boundary between 'w' and '_'. Every "
        "T1 series whose description uses underscores is therefore classified "
        "fluid-sensitive, routed into a FLUID slot, and leaves COR_T1/SAG_T1 empty "
        "and masked out. Those slots carry the structural sequences the three OA "
        "labels read on. Underscore-separated names are the Siemens house style "
        "('t1_tse_sag'), so this is not a hypothetical spelling.\n\n"
        "Not fixed in place because classify_weighting feeds src/data/cache.py: "
        "changing it changes slot routing, which invalidates every published cache "
        "and makes new runs incomparable to the checkpoints behind LB 0.856. That "
        "is a rebuild-and-retrain decision, not a drive-by patch.\n\n"
        "Prevalence is UNMEASURED — series text lives in DICOM headers, which are "
        "Kaggle-only. train_series.csv does show the population at risk: 10,361 of "
        "24,371 series (42.5%) are structural, spread over all 4,407 studies, of "
        "which 9,182 are sagittal or coronal, i.e. SAG_T1/COR_T1 candidates.\n\n"
        "strict=True on purpose: when the regex is fixed this test starts passing "
        "and pytest fails it, forcing the marker to be removed rather than the fix "
        "going unnoticed."
    ),
)
@pytest.mark.parametrize("text", ["t1w_sag", "T1W_TSE", "t1_tse_sag", "SAG_T1W_TSE"])
def test_underscore_separated_t1_is_structural(text):
    fluid, _ = classify_weighting(text, None, None)
    assert fluid is False


@pytest.mark.parametrize("text", ["SAG T2 FSE", "PD FS", "proton density", "t2w"])
def test_t2_and_pd_text_are_fluid_sensitive(text):
    fluid, _ = classify_weighting(text, None, None)
    assert fluid is True


@pytest.mark.parametrize("text", ["STIR COR", "TIRM sag", "SPAIR ax"])
def test_stir_family_is_fluid_sensitive_even_when_t1_is_also_named(text):
    """STIR/TIRM/SPAIR name the fat-suppression preparation, not the weighting.
    A header saying both must not be read as structural."""
    fluid, fatsat = classify_weighting(text + " T1", None, None)
    assert fluid is True
    assert fatsat is True


@pytest.mark.parametrize(
    "text", ["SAG PD FS", "cor_fatsat", "T2 fat-sat", "ax fat sat", "water excitation"]
)
def test_fat_suppression_is_detected(text):
    _, fatsat = classify_weighting(text, None, None)
    assert fatsat is True


def test_plain_sequences_are_not_marked_fat_suppressed():
    _, fatsat = classify_weighting("SAG T2 TSE", None, None)
    assert fatsat is False


def test_tr_te_fallback_when_text_says_nothing():
    """Conventional teaching thresholds, used only when the text is silent."""
    assert classify_weighting("series 4", tr=500, te=15)[0] is False   # short/short -> T1
    assert classify_weighting("series 4", tr=3000, te=80)[0] is True   # long/long   -> T2
    assert classify_weighting("series 4", tr=3000, te=15)[0] is True   # long/short  -> PD


def test_unclassifiable_defaults_to_fluid_sensitive():
    """The conservative error: a structural series in a fluid slot costs
    contrast, an empty fluid slot costs the finding."""
    assert classify_weighting("", None, None)[0] is True


# --------------------------------------------------------------------------
# 5. Slot routing
# --------------------------------------------------------------------------


def _series(plane, fluid, fatsat, n_files=10, uid="u"):
    return SeriesInfo(uid=uid, plane=plane, fluid=fluid, fatsat=fatsat,
                      n_files=n_files, side="R", pixel_mm=0.4)


def test_pick_slot_series_always_returns_one_entry_per_slot():
    """Positional alignment with SLOTS is what the presence mask indexes."""
    assert len(pick_slot_series([])) == N_SLOT
    assert all(s is None for s in pick_slot_series([]))


def test_the_series_with_most_files_wins_its_slot():
    """Slice count is the proxy for 'covers the joint' vs 'localiser'."""
    small = _series("Sagittal", True, True, n_files=3, uid="small")
    big = _series("Sagittal", True, True, n_files=40, uid="big")
    chosen = pick_slot_series([small, big])
    assert chosen[0] is not None and chosen[0].uid == "big"


def test_slots_are_not_filled_across_tissue_contrast():
    """Deliberately no fallback: a fluid-sensitive coronal must NOT be used to
    fill COR_T1, or two different contrasts share one slot embedding."""
    fluid_coronal = _series("Coronal", True, True)
    chosen = pick_slot_series([fluid_coronal])
    names = [s[0] for s in SLOTS]
    assert chosen[names.index("COR_FLUID_FS")] is not None
    assert chosen[names.index("COR_T1")] is None, "COR_T1 was faked from a fluid series"


def test_fat_suppressed_and_unsuppressed_sagittals_route_separately():
    names = [s[0] for s in SLOTS]
    chosen = pick_slot_series([
        _series("Sagittal", True, True, uid="fs"),
        _series("Sagittal", True, False, uid="nofs"),
    ])
    assert chosen[names.index("SAG_FLUID_FS")].uid == "fs"
    assert chosen[names.index("SAG_FLUID_NOFS")].uid == "nofs"


# --------------------------------------------------------------------------
# 6. Frame preparation
# --------------------------------------------------------------------------


def test_physical_crop_reports_whether_it_applied():
    """The rate matters: if the crop is largely a no-op, any comparison against
    an uncropped run means nothing."""
    frame = np.zeros((400, 400), dtype=np.uint8)
    out, applied = physical_crop(frame, pixel_mm=0.5, crop_mm=100.0)
    assert applied is True
    assert out.shape == (200, 200)


@pytest.mark.parametrize("pixel_mm", [None, 0, -1.0])
def test_physical_crop_falls_back_when_spacing_is_missing(pixel_mm):
    frame = np.zeros((400, 400), dtype=np.uint8)
    out, applied = physical_crop(frame, pixel_mm=pixel_mm)
    assert applied is False
    assert out.shape == frame.shape


def test_physical_crop_falls_back_when_the_fov_does_not_fit():
    """Better an uncropped frame than a padded or wrapped one."""
    frame = np.zeros((100, 100), dtype=np.uint8)
    out, applied = physical_crop(frame, pixel_mm=1.0, crop_mm=500.0)
    assert applied is False
    assert out.shape == frame.shape


def test_physical_crop_is_centred():
    frame = np.zeros((10, 10), dtype=np.uint8)
    frame[5, 5] = 255  # just past centre
    out, applied = physical_crop(frame, pixel_mm=1.0, crop_mm=4.0)
    assert applied is True
    assert out.shape == (4, 4)
    assert out[2, 2] == 255, "crop was not centred"


def test_band_indices_avoid_the_ends_of_the_stack():
    """The first and last ~20% of a knee stack are soft tissue outside the joint."""
    idx = band_indices(100, 6)
    assert len(idx) == 6
    assert idx.min() >= 19 and idx.max() <= 80


@pytest.mark.parametrize("n_slices", [0, 1, 2, 3, 7, 400])
def test_band_indices_are_always_in_range_and_the_right_count(n_slices):
    """Some series are under 10 slices; short stacks must not raise or return
    fewer indices than asked for."""
    idx = band_indices(n_slices, GROUP)
    assert len(idx) == GROUP
    if n_slices > 0:
        assert idx.min() >= 0 and idx.max() <= n_slices - 1


def test_normalise_intensity_returns_uint8_spanning_the_range():
    volume = np.random.RandomState(0).rand(3, 8, 8) * 1000
    out = normalise_intensity(volume)
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255


def test_normalise_intensity_is_per_slot_not_per_slice():
    """Relative intensity BETWEEN slices is what makes an effusion stand out
    from the slice either side. Per-slice scaling would erase it by mapping
    every slice to the same range."""
    volume = np.zeros((2, 4, 4), dtype=float)
    volume[0] = 10.0
    volume[1] = 1000.0
    out = normalise_intensity(volume)
    assert out[0].max() < out[1].min(), "slices were normalised independently"


@pytest.mark.parametrize(
    "volume",
    [np.zeros((2, 4, 4)), np.full((2, 4, 4), 7.0), np.zeros((0, 4, 4))],
)
def test_normalise_intensity_degrades_to_zeros_rather_than_dividing_by_zero(volume):
    out = normalise_intensity(volume)
    assert out.dtype == np.uint8
    assert not np.any(np.isnan(out.astype(float)))


# --------------------------------------------------------------------------
# 7. The heads
# --------------------------------------------------------------------------


def test_slot_prior_table_covers_every_target_with_valid_indices():
    """A typo'd target name is silently ignored by the head's build loop, and a
    stale slot index would tilt the wrong acquisition."""
    assert set(SLOT_PRIOR_TABLE) == set(TARGETS)
    for target, slots in SLOT_PRIOR_TABLE.items():
        assert slots, f"{target} has no preferred slot"
        assert all(0 <= i < N_SLOT for i in slots), target


def test_every_target_belongs_to_exactly_one_family():
    assert len(GROUP_OF_TARGET) == len(TARGETS)
    flat = [t for group in TARGET_GROUPS.values() for t in group]
    assert sorted(flat) == sorted(TARGETS), "a target is missing or double-counted"


def test_slot_head_shapes():
    head = SlotHead(dim=32).eval()
    logits = head(torch.randn(2, N_SLOT, 32), torch.ones(2, N_SLOT))
    assert logits.shape == (2, len(TARGETS))


def test_slot_prior_travels_in_the_state_dict():
    """Registered as a buffer on purpose: a checkpoint trained with the tilt and
    loaded without it is silently a different model that still loads cleanly."""
    head = SlotHead(dim=32)
    assert "slot_prior" in head.state_dict()
    assert head.state_dict()["slot_prior"].shape == (len(TARGETS), N_SLOT)


def test_slot_prior_tilts_only_the_preferred_slots():
    head = SlotHead(dim=32, prior=True)
    tilt = head.state_dict()["slot_prior"]
    for target, slots in SLOT_PRIOR_TABLE.items():
        row = tilt[TARGETS.index(target)]
        for i in range(N_SLOT):
            expected = SLOT_PRIOR_STRENGTH if i in slots else 0.0
            assert row[i].item() == pytest.approx(expected), (target, i)


def test_disabling_the_prior_gives_a_zero_tilt():
    head = SlotHead(dim=32, prior=False)
    assert torch.count_nonzero(head.state_dict()["slot_prior"]) == 0


def test_absent_slots_cannot_influence_slot_head_logits():
    """THE mask test. If this fails, absent acquisitions are contributing —
    the model is learning from a black image the encoder read confidently."""
    torch.manual_seed(0)
    head = SlotHead(dim=32).eval()
    x = torch.randn(2, N_SLOT, 32)
    mask = torch.ones(2, N_SLOT)
    mask[:, 3] = 0.0  # slot 3 absent

    with torch.no_grad():
        before = head(x, mask)
        polluted = x.clone()
        polluted[:, 3] = torch.randn(2, 32) * 100  # garbage in the absent slot
        after = head(polluted, mask)

    assert torch.allclose(before, after, atol=1e-5), "a masked slot changed the output"


def test_present_slots_do_still_influence_slot_head_logits():
    """Guards the opposite failure: a mask that excludes everything would pass
    the test above trivially."""
    torch.manual_seed(0)
    head = SlotHead(dim=32).eval()
    x = torch.randn(2, N_SLOT, 32)
    mask = torch.ones(2, N_SLOT)

    with torch.no_grad():
        before = head(x, mask)
        changed = x.clone()
        changed[:, 3] = torch.randn(2, 32) * 100
        after = head(changed, mask)

    assert not torch.allclose(before, after, atol=1e-5)


def test_xattn_head_shapes():
    head = XAttnHead(dim=32, pooled_dim=96).eval()
    logits = head(
        torch.randn(2, N_SLOT, 5, 32),   # (B, S, tokens, D)
        torch.randn(2, N_SLOT, 96),      # pooled path keeps the concat width
        torch.ones(2, N_SLOT),
    )
    assert logits.shape == (2, len(TARGETS))


def test_xattn_rejects_a_pooled_width_that_is_not_the_concatenation():
    """Passing `dim` where `pooled_dim` belongs must raise immediately rather
    than train something subtly wrong — the docstring says this is how the bug
    was originally caught."""
    head = XAttnHead(dim=32, pooled_dim=96).eval()
    with pytest.raises(RuntimeError):
        head(torch.randn(2, N_SLOT, 5, 32), torch.randn(2, N_SLOT, 32),
             torch.ones(2, N_SLOT))


def test_absent_slots_cannot_influence_xattn_logits():
    torch.manual_seed(0)
    head = XAttnHead(dim=32, pooled_dim=96).eval()
    tokens = torch.randn(2, N_SLOT, 5, 32)
    pooled = torch.randn(2, N_SLOT, 96)
    mask = torch.ones(2, N_SLOT)
    mask[:, 2] = 0.0

    with torch.no_grad():
        before = head(tokens, pooled, mask)
        dirty_tokens, dirty_pooled = tokens.clone(), pooled.clone()
        dirty_tokens[:, 2] = torch.randn(2, 5, 32) * 100
        dirty_pooled[:, 2] = torch.randn(2, 96) * 100
        after = head(dirty_tokens, dirty_pooled, mask)

    assert torch.allclose(before, after, atol=1e-5)


def test_heads_are_finite_when_a_study_has_only_one_slot():
    """The realistic worst case in this corpus — many studies are missing most
    acquisitions. Softmax over a single unmasked entry must stay finite."""
    torch.manual_seed(0)
    mask = torch.zeros(1, N_SLOT)
    mask[0, 0] = 1.0

    slot = SlotHead(dim=32).eval()
    xattn = XAttnHead(dim=32, pooled_dim=96).eval()
    with torch.no_grad():
        assert torch.isfinite(slot(torch.randn(1, N_SLOT, 32), mask)).all()
        assert torch.isfinite(
            xattn(torch.randn(1, N_SLOT, 5, 32), torch.randn(1, N_SLOT, 96), mask)
        ).all()
