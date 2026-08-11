"""Slot-based study representation: six acquisitions, laterality-normalised.

**This module is shared by cache building and inference. Do not fork it.**
Train/test preprocessing skew is the classic way to score well in CV and badly
on the leaderboard.

Why this exists, and what it replaces
-------------------------------------
`dicom.py` builds a study as *three planes*, taking one fluid-sensitive series
per plane and dropping the rest. Three measured problems with that:

1. **No laterality normalisation.** Five of the twelve targets are side-specific
   (Medial/Lateral Meniscus, Medial/Lateral OA, and MCL is medial by
   definition). Left and right knees are mirror images, so "medial" falls on
   opposite sides of the frame depending on which knee was scanned. Without
   normalising, the model is asked to learn a direction that flips at random
   across ~42% of the macro metric.

2. **T1/structural series are discarded entirely.** `pick_series` prefers
   `Fluid_Sensitive == 1` and never returns anything else. Cartilage thinning
   and marrow signal — the OA labels — read on the structural sequences.

3. **The CSV flags cannot do the routing.** `docs/03-data-guide.md` records that
   `Fluid_Sensitive` and `Fat_Suppression` are identical in all 24,371 rows of
   `train_series.csv`, so between them they carry one bit, not two. The
   weighting has to be recovered from the DICOM headers instead.

The six slots below are plane x acquisition weighting. A study rarely has all
six, which is what the presence mask is for — a missing slot is masked out of
the attention rather than fed in as zeros.

Coordinate conventions
----------------------
DICOM patient space for a head-first supine study: +x is the patient's LEFT,
+y is posterior, +z is superior. `ImagePositionPatient` is the centre of the
*first voxel*, not of the image, so the side of the body a knee sits on is read
from the image CENTRE — the corner is offset by half a field of view and for a
knee that is enough to land on the wrong sign.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

#: (name, plane, fluid_sensitive, fat_suppressed). `None` means "don't care".
#:
#: Ordering is load-bearing: SLOT_PRIOR in the model indexes into this list, and
#: a cache written under one ordering cannot be read under another.
SLOTS: tuple[tuple[str, str, bool | None, bool | None], ...] = (
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, None),
    ("SAG_T1", "Sagittal", False, None),
)
N_SLOT = len(SLOTS)

#: Square physical field of view every frame is cropped to, in millimetres.
#:
#: 130 mm, not the 160 mm `dicom.py` uses. A knee joint spans ~140 mm, so 160 mm
#: includes margin — but the margin is paid for in resolution, and resolution is
#: what the fine findings need. At 130 mm / 336 px one pixel is 0.387 mm; at
#: 160 mm / 224 px it is 0.714 mm. A 1-3 mm meniscal tear is 1-4 pixels at the
#: latter, which is why the 160 mm experiment moved CV by 0.002.
CROP_MM = 130.0

#: Output side in pixels. 336 = 14 x 24, a multiple of the DINOv2 patch size, so
#: the same cache feeds a patch-14 ViT without resampling. 256 is NOT (18.29).
IMG = 336

#: Slices per encoder input, stacked as the three RGB channels (the 2.5D trick).
GROUP = 3

#: How many GROUPs are cached per slot. Training draws one group per step, which
#: acts as augmentation along the stack; inference averages over all of them.
N_GROUP = 1

#: Fraction of the ordered stack slices are drawn from. The first and last ~20%
#: of a knee stack are mostly soft tissue outside the joint.
SLICE_BAND = (0.20, 0.80)

_FATSAT_RX = re.compile(
    r"\bfs\b|fatsat|fat[ _-]?sat|\bstir\b|\bspair\b|\bspir\b|\btirm\b|"
    r"water[ _-]?excit|\bfatsup\b|\bspectral\b",
    re.I,
)
_T1_RX = re.compile(r"\bt1\b|\bt1w\b|\bt1[ _-]?tse\b", re.I)
_T2_RX = re.compile(r"\bt2\b|\bt2w\b", re.I)
_PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|\bdens", re.I)


@dataclass
class SeriesInfo:
    """One series, as read from a single header of it.

    `tag_side` and `geom_side` are kept apart from `side` so the two routes can
    be audited against each other. They agree or they do not, and which it is
    decides whether the mirror is a fix or a new defect — a mirror applied to
    the wrong half of the corpus is strictly worse than no mirror at all.
    """

    uid: str
    plane: str
    fluid: bool
    fatsat: bool
    n_files: int
    side: str | None          # 'L' / 'R' / None — the one actually used
    pixel_mm: float | None
    tag_side: str | None = None    # from Laterality (0020,0060), when present
    geom_side: str | None = None   # from image centre in patient space


def classify_weighting(text: str, tr: float | None, te: float | None) -> tuple[bool, bool]:
    """(fluid_sensitive, fat_suppressed) from series text plus TR/TE.

    Text first because it is explicit when present. TR/TE are the fallback and
    are genuinely ambiguous at the boundaries, so the thresholds below are the
    conventional teaching ones rather than anything fitted:

    - T1: short TR (<800 ms) and short TE (<30 ms)
    - T2: long TR (>1500 ms) and long TE (>60 ms)
    - PD: long TR with short TE — fluid-sensitive, and the workhorse sequence
      for meniscal tears in this corpus

    A series that matches nothing is called fluid-sensitive. That is the
    majority class here and the conservative error: putting a structural series
    in a fluid slot costs contrast, while leaving a fluid slot empty costs the
    finding entirely.
    """
    fatsat = bool(_FATSAT_RX.search(text))

    if _T1_RX.search(text):
        fluid = False
    elif _T2_RX.search(text) or _PD_RX.search(text):
        fluid = True
    elif tr is not None and te is not None:
        if tr < 800 and te < 30:
            fluid = False
        else:
            fluid = True
    else:
        fluid = True

    # STIR/SPAIR/TIRM are fat-suppressed fluid-sensitive sequences by
    # construction; the text match above already set fatsat, but a header that
    # names STIR and also says T1 is naming the preparation, not the weighting.
    if re.search(r"\bstir\b|\btirm\b|\bspair\b", text, re.I):
        fluid = True

    return fluid, fatsat


def side_from_geometry(orientation, position, rows, cols, pixel_mm) -> str | None:
    """'L' / 'R' / None from where the image centre sits in patient space.

    `Laterality` (0020,0060) is Type 2C and legitimately absent on roughly half
    this corpus, so geometry is the primary route rather than the fallback.

    The centre is used, not `ImagePositionPatient` itself: that tag is the
    centre of the first transmitted voxel, i.e. a corner of the image. For a
    knee the two differ by ~half a field of view, which is comfortably enough to
    put the sign on the wrong side of zero.

    Returns None when the geometry is missing or the centre lands within
    `DEAD_ZONE_MM` of the midline, where the reading is not trustworthy.
    """
    DEAD_ZONE_MM = 10.0
    try:
        orientation = np.asarray([float(v) for v in orientation], dtype=float)
        position = np.asarray([float(v) for v in position], dtype=float)
        if orientation.shape[0] < 6 or position.shape[0] < 3:
            return None
        row_dir, col_dir = orientation[:3], orientation[3:6]
        spacing = float(pixel_mm) if pixel_mm else 1.0
        centre = (
            position
            + row_dir * spacing * (float(cols) / 2.0)
            + col_dir * spacing * (float(rows) / 2.0)
        )
    except (TypeError, ValueError, IndexError):
        return None

    x = float(centre[0])
    if abs(x) < DEAD_ZONE_MM:
        return None
    # +x is the patient's LEFT in the DICOM patient coordinate system.
    return "L" if x > 0 else "R"


def normalise_laterality(volume: np.ndarray, plane: str, side: str | None) -> np.ndarray:
    """Mirror a left knee so every study presents as a right knee.

    `volume` is (n_slices, H, W), already ordered along the stack.

    The axis that carries medial-lateral depends on the plane, which is why this
    cannot be a single flip:

    - **Coronal / Axial** — medial-lateral runs across the image, so the frames
      are flipped horizontally.
    - **Sagittal** — the medial-lateral axis IS the stack axis (each slice is a
      different depth through the knee), so the SLICE ORDER is reversed and the
      frames are left alone. Flipping a sagittal frame horizontally would mirror
      anterior-posterior instead, which relabels nothing but destroys the
      position of the patella and the popliteal fossa.

    A study whose side could not be determined is returned untouched. Guessing
    would be worse than the inconsistency: a wrong mirror is an actively
    misleading example, while an unmirrored one is merely one the model has to
    tolerate.
    """
    if side != "L":
        return volume
    if plane == "Sagittal":
        return volume[::-1].copy()
    return volume[:, :, ::-1].copy()


def physical_crop(frame: np.ndarray, pixel_mm: float | None, crop_mm: float = CROP_MM):
    """Centre-crop to a constant millimetre field of view.

    Returns (cropped_frame, applied). `applied` is reported rather than counted
    in a module global because the caller may run this across a process fork,
    where a global increments in the child and reads zero in the parent — the
    exact failure that made the first crop experiment uninterpretable.

    Falls back to the untouched frame when spacing is missing or the requested
    extent does not fit. A fallback frame is at a different scale from a cropped
    one, so the caller should track the rate: if it is high, the crop is largely
    a no-op and any comparison against an uncropped run means nothing.
    """
    if not pixel_mm or pixel_mm <= 0:
        return frame, False
    height, width = frame.shape[:2]
    side = int(round(crop_mm / float(pixel_mm)))
    if side < 2 or side > height or side > width:
        return frame, False
    top, left = (height - side) // 2, (width - side) // 2
    return frame[top : top + side, left : left + side], True


def band_indices(n_slices: int, n_take: int, band: tuple[float, float] = SLICE_BAND):
    """Evenly spaced indices across the central band of an ordered stack.

    Clamped into range so a short series (some are under 10 slices) still yields
    `n_take` indices rather than raising or silently returning fewer.
    """
    if n_slices <= 0:
        return np.zeros(n_take, dtype=int)
    lo = int(np.floor(band[0] * (n_slices - 1)))
    hi = int(np.ceil(band[1] * (n_slices - 1)))
    if hi <= lo:
        lo, hi = 0, n_slices - 1
    idx = np.linspace(lo, hi, n_take)
    return np.clip(np.round(idx), 0, n_slices - 1).astype(int)


def normalise_intensity(volume: np.ndarray, low: float = 1.0, high: float = 99.0):
    """Percentile-clip to uint8 across the whole slot.

    Per-slot rather than per-slice so relative intensity between slices — which
    is what makes an effusion or a marrow-oedema cloud stand out from the slice
    either side of it — survives.
    """
    if volume.size == 0:
        return np.zeros_like(volume, dtype=np.uint8)
    lo, hi = np.percentile(volume, [low, high])
    if hi <= lo:
        return np.zeros(volume.shape, dtype=np.uint8)
    scaled = np.clip((volume - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).round().astype(np.uint8)


def pick_slot_series(infos: list[SeriesInfo]) -> list[SeriesInfo | None]:
    """Best series for each of the six slots, or None where the study has none.

    Where several series match a slot the one with the most files wins: slice
    count is the best cheap proxy for which acquisition covers the joint rather
    than a localiser or a single-slice repeat.

    Deliberately no fallback across slots. Filling COR_T1 from a fluid-sensitive
    coronal would put a different tissue contrast behind the same slot
    embedding, and the presence mask exists precisely so that an absent slot can
    be declared absent instead of faked.
    """
    chosen: list[SeriesInfo | None] = []
    for _name, plane, want_fluid, want_fatsat in SLOTS:
        candidates = [
            s
            for s in infos
            if s.plane == plane
            and (want_fluid is None or s.fluid == want_fluid)
            and (want_fatsat is None or s.fatsat == want_fatsat)
        ]
        chosen.append(max(candidates, key=lambda s: s.n_files) if candidates else None)
    return chosen


def study_side(infos: list[SeriesInfo]) -> str | None:
    """One side per study, by majority vote over its series.

    Voting rather than trusting the first series: the geometry read is per
    series and an occasional one is wrong or absent, while a knee study is of
    one knee. A tie returns None and the study is left unmirrored.
    """
    votes = [s.side for s in infos if s.side in ("L", "R")]
    if not votes:
        return None
    left = votes.count("L")
    right = votes.count("R")
    if left == right:
        return None
    return "L" if left > right else "R"
