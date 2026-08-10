"""DICOM loading and volume construction.

**This module is shared by preprocessing and inference on purpose.** Train/test
preprocessing skew is the classic way to score well in CV and badly on the
leaderboard, so both paths must call exactly the same code. Do not fork it.

Design decisions and the evidence behind them (see docs/03-data-guide.md):

- **Sort slices by geometry, not filename.** `ImagePositionPatient` projected
  onto the slice normal gives true anatomical order; filenames are UIDs and
  carry no order at all.
- **Normalize per series, not globally.** MRI intensities are not absolute
  (unlike CT Hounsfield units) — a value of 400 means nothing without context.
  Percentile clipping survives the bright outliers MRI produces.
- **Route by plane only, not by sequence flags.** `Fluid_Sensitive` and
  `Fat_Suppression` are *identical in all 24,371 rows* of train_series.csv
  (verified session 3), so they carry one bit between them, not two. Any
  routing logic treating them as independent is fiction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PLANES = ("Sagittal", "Coronal", "Axial")


def read_slice(path: Path | str):
    """Read one DICOM, returning (pixel_array, dataset) or (None, None).

    Tolerates the four transfer syntaxes in this dataset (Explicit/Implicit VR
    LE, JPEG Lossless, JPEG 2000). A failure here is expected occasionally and
    must never kill a whole study — the caller drops the slice and continues.
    """
    import pydicom

    try:
        ds = pydicom.dcmread(str(path))
        return ds.pixel_array, ds
    except Exception:
        return None, None


def _slice_position(ds) -> float:
    """Projection of the slice origin onto the slice normal.

    This is the only reliable ordering key. Falls back to InstanceNumber, then
    0.0, so a series with degraded metadata still produces *an* order rather
    than raising.
    """
    try:
        orientation = [float(v) for v in ds.ImageOrientationPatient]
        position = [float(v) for v in ds.ImagePositionPatient]
        row = np.array(orientation[:3])
        col = np.array(orientation[3:])
        normal = np.cross(row, col)
        return float(np.dot(np.array(position), normal))
    except Exception:
        try:
            return float(ds.InstanceNumber)
        except Exception:
            return 0.0


def normalize_series(volume: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Percentile-clip to [0,1] across the whole series.

    Per-series (not per-slice) so relative intensity between slices — which is
    what makes an effusion or a marrow-edema cloud stand out — is preserved.
    """
    if volume.size == 0:
        return volume.astype(np.float32)
    lo, hi = np.percentile(volume, [low, high])
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def resize_volume(volume: np.ndarray, n_slices: int, size: int) -> np.ndarray:
    """Resample to a fixed (n_slices, size, size) tensor.

    Slice sampling is *even across the stack*, not a centre crop: pathology in
    this dataset is not reliably central (a Baker's cyst sits posteriorly, a
    meniscal tear a few slices off-centre), so cropping to the middle would
    systematically discard findings.
    """
    import cv2

    if volume.shape[0] == 0:
        return np.zeros((n_slices, size, size), dtype=np.float32)

    idx = np.linspace(0, volume.shape[0] - 1, n_slices).round().astype(int)
    sampled = volume[idx]
    out = np.zeros((n_slices, size, size), dtype=np.float32)
    for i, frame in enumerate(sampled):
        out[i] = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return out


#: Side of the square field of view every frame is cropped to, in millimetres.
#: A knee joint spans roughly 140 mm; 160 mm keeps the joint plus a margin for
#: the Baker's cyst region, which sits posteriorly and is the finding most
#: easily cropped away.
FOV_MM = 160.0

#: How often the crop actually applied vs fell back. A silent fallback would
#: make this whole change a no-op while still looking correct, so training and
#: inference print it. Reset with `CROP_STATS.update(cropped=0, fallback=0)`.
CROP_STATS = {"cropped": 0, "fallback": 0}


def physical_crop(frame: np.ndarray, ds, fov_mm: float = FOV_MM) -> np.ndarray:
    """Centre-crop `frame` to a fixed *millimetre* field of view.

    Why this is not optional: `PixelSpacing` varies across this corpus, so a
    fixed-pixel resize hands the encoder the same anatomy at different scales
    depending on which scanner produced the study — a 0.3 mm/px and a 0.6 mm/px
    series of the same knee become images that differ by a factor of two. The
    encoder then has to learn scale invariance it was never given the data to
    learn, and the variation correlates with site, which is exactly the nuisance
    our grouped CV is built to punish.

    Cropping to a constant physical extent *before* the resize makes one output
    pixel mean the same number of millimetres in every study.

    Falls back to the untouched frame when spacing is missing or the requested
    extent does not fit — degrading to the old behaviour for that slice beats
    dropping it.
    """
    spacing = getattr(ds, "PixelSpacing", None)
    if spacing is None:
        CROP_STATS['fallback'] += 1
        return frame
    try:
        row_mm, col_mm = float(spacing[0]), float(spacing[1])
    except (TypeError, ValueError, IndexError):
        CROP_STATS['fallback'] += 1
        return frame
    if row_mm <= 0 or col_mm <= 0:
        CROP_STATS['fallback'] += 1
        return frame

    height, width = frame.shape
    crop_h, crop_w = int(round(fov_mm / row_mm)), int(round(fov_mm / col_mm))
    if crop_h < 2 or crop_w < 2 or crop_h > height or crop_w > width:
        CROP_STATS['fallback'] += 1
        return frame

    top, left = (height - crop_h) // 2, (width - crop_w) // 2
    CROP_STATS['cropped'] += 1
    return frame[top : top + crop_h, left : left + crop_w]


def load_series(series_dir: Path | str, n_slices: int = 16, size: int = 224) -> np.ndarray:
    """One series directory -> a normalized (n_slices, size, size) volume."""
    series_dir = Path(series_dir)
    frames: list[tuple[float, np.ndarray]] = []

    for dcm_path in series_dir.glob("*.dcm"):
        pixels, ds = read_slice(dcm_path)
        if pixels is None:
            continue
        # Crop to a constant physical extent first, so the resize below is the
        # same anatomical scale for every scanner in the corpus.
        frames.append((_slice_position(ds), physical_crop(pixels.astype(np.float32), ds)))

    if not frames:
        return np.zeros((n_slices, size, size), dtype=np.float32)

    frames.sort(key=lambda pair: pair[0])
    # Series can mix resolutions; resize each frame before stacking so the
    # np.stack below cannot raise on ragged shapes.
    import cv2

    stack = np.stack(
        [cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) for _, frame in frames]
    )
    return resize_volume(normalize_series(stack), n_slices, size)


def pick_series(series_meta, study_uid: str) -> dict[str, str | None]:
    """Choose one series per anatomical plane for a study.

    Prefers fluid-sensitive series — that is where effusion, contusion,
    Baker's cyst and acute ligament injury are visible at all (see
    docs/02-domain-primer.md) — and falls back to any series in the plane so a
    study missing a fluid-sensitive acquisition still yields a volume rather
    than a hole.
    """
    rows = series_meta[series_meta["StudyInstanceUID"] == study_uid]
    chosen: dict[str, str | None] = {}
    for plane in PLANES:
        in_plane = rows[rows["Anatomical_Plane"] == plane]
        if len(in_plane) == 0:
            chosen[plane] = None
            continue
        fluid = in_plane[in_plane["Fluid_Sensitive"] == 1]
        pick = fluid if len(fluid) else in_plane
        chosen[plane] = str(pick.iloc[0]["SeriesInstanceUID"])
    return chosen


def load_study(
    root: Path | str,
    study_uid: str,
    series_meta,
    n_slices: int = 16,
    size: int = 224,
) -> np.ndarray:
    """One study -> (3, n_slices, size, size), one plane per channel group.

    Missing planes become zero volumes rather than errors: roughly every study
    has sagittal and coronal, but axial is less consistent (5,898 axial series
    vs 9,864 sagittal across 4,407 studies), and the model must tolerate that
    at inference on unseen data.
    """
    root = Path(root)
    picks = pick_series(series_meta, study_uid)
    planes = []
    for plane in PLANES:
        series_uid = picks.get(plane)
        if series_uid is None:
            planes.append(np.zeros((n_slices, size, size), dtype=np.float32))
        else:
            planes.append(load_series(root / study_uid / series_uid, n_slices, size))
    return np.stack(planes)
