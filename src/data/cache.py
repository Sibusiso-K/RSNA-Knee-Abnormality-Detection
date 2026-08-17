"""DICOM -> slot tensor. The single decode path for train AND test.

**Do not fork this module.** Train/test preprocessing skew is the classic way
to score well in CV and badly on the leaderboard, and it is especially easy to
introduce here because the training cache is built in one notebook, weeks
before the submission notebook builds the test cache in another. If those two
ever drift, every weight learned against the first is being applied to pixels
it has never seen, and nothing in the pipeline will complain.

`src/data/slots.py` holds the pure decisions (routing, laterality, crop,
sampling) and is unit-tested without any DICOM. This module is the I/O layer
around it: find the files, read the headers, decode the chosen frames.
"""

from __future__ import annotations

import glob
import os

import numpy as np

from src.data.slots import (
    GROUP,
    IMG,
    N_GROUP,
    N_SLOT,
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
from src.model.validation import FINGERPRINT_TAGS, _clean_tag

N_SLICE = GROUP * N_GROUP

#: Tags parsed per header. Restricting the set is not a micro-optimisation: a
#: full build opens ~800k files and full parsing of each is the difference
#: between fitting the 12 h CPU cap and not.
HDR_TAGS = [
    "SeriesInstanceUID", "SeriesDescription", "SequenceName", "ScanOptions",
    "ScanningSequence", "RepetitionTime", "EchoTime", "Laterality",
    "PixelSpacing", "Rows", "Columns", "ImagePositionPatient",
    "ImageOrientationPatient", "InstanceNumber",
    *FINGERPRINT_TAGS,
]

PLANES = ("Sagittal", "Coronal", "Axial")


def _num(ds, name):
    try:
        value = getattr(ds, name, None)
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def read_header(path):
    import pydicom

    try:
        return pydicom.dcmread(path, stop_before_pixels=True, specific_tags=HDR_TAGS)
    except Exception:
        return None


def slice_key(ds) -> float:
    """Projection of the slice origin onto the slice normal.

    The only reliable ordering key. Filenames are SOP Instance UIDs, assigned
    to be unique rather than ordered, so sorting by name scrambles the stack —
    and a scrambled stack silently destroys the 2.5D triplet, which assumes the
    three channels are adjacent anatomy.
    """
    try:
        orientation = [float(v) for v in ds.ImageOrientationPatient]
        position = [float(v) for v in ds.ImagePositionPatient]
        normal = np.cross(np.array(orientation[:3]), np.array(orientation[3:6]))
        return float(np.dot(np.array(position), normal))
    except Exception:
        value = _num(ds, "InstanceNumber")
        return value if value is not None else 0.0


def probe_study(series_dir: str, study_uid: str, plane_of: dict):
    """One header per series -> ([SeriesInfo], {uid: files}, fingerprint)."""
    infos: list[SeriesInfo] = []
    files_of: dict[str, list[str]] = {}
    fingerprint = ""

    study_dir = os.path.join(series_dir, study_uid)
    if not os.path.isdir(study_dir):
        return infos, files_of, fingerprint

    for series_uid in sorted(os.listdir(study_dir)):
        files = glob.glob(os.path.join(study_dir, series_uid, "*.dcm"))
        if not files:
            continue
        ds = read_header(files[0])
        if ds is None:
            continue

        if not fingerprint:
            fingerprint = "|".join(
                _clean_tag(tag, getattr(ds, tag, None)) for tag in FINGERPRINT_TAGS
            )

        plane = plane_of.get(series_uid)
        if not isinstance(plane, str) or plane not in PLANES:
            continue

        text = " ".join(
            str(getattr(ds, tag, "") or "")
            for tag in ("SeriesDescription", "SequenceName", "ScanOptions",
                        "ScanningSequence")
        )
        fluid, fatsat = classify_weighting(
            text, _num(ds, "RepetitionTime"), _num(ds, "EchoTime")
        )

        try:
            pixel_mm = float(ds.PixelSpacing[0])
        except Exception:
            pixel_mm = None

        # Both laterality routes are computed even when the tag settles it: the
        # extra cost is a dot product on a header already in memory, and the
        # agreement rate is the only check on whether the geometry rule is
        # right. Without it the mirror is unfalsifiable.
        laterality = str(getattr(ds, "Laterality", "") or "").strip().upper()
        tag_side = laterality if laterality in ("L", "R") else None
        geom_side = side_from_geometry(
            getattr(ds, "ImageOrientationPatient", None),
            getattr(ds, "ImagePositionPatient", None),
            getattr(ds, "Rows", 0), getattr(ds, "Columns", 0), pixel_mm,
        )

        infos.append(SeriesInfo(
            uid=series_uid, plane=plane, fluid=fluid, fatsat=fatsat,
            n_files=len(files), side=tag_side or geom_side, pixel_mm=pixel_mm,
            tag_side=tag_side, geom_side=geom_side,
        ))
        files_of[series_uid] = files

    return infos, files_of, fingerprint


def read_slot(files: list[str], plane: str, side: str | None):
    """A chosen series -> (uint8 (N_SLICE, IMG, IMG), fully_cropped).

    Headers of every file are read to establish the true geometric order, then
    only the N_SLICE frames actually wanted are decoded. Header parsing is
    cheap next to pixel decoding, so this beats decoding the series and
    discarding most of it by a wide margin.
    """
    import cv2
    import pydicom

    keyed = []
    for path in files:
        ds = read_header(path)
        if ds is not None:
            keyed.append((slice_key(ds), path, ds))
    if not keyed:
        return None, False

    keyed.sort(key=lambda t: t[0])
    picks = band_indices(len(keyed), N_SLICE)

    frames, cropped = [], 0
    for i in picks:
        _key, path, ds = keyed[i]
        try:
            pixels = pydicom.dcmread(path).pixel_array.astype(np.float32)
        except Exception:
            # A slice that will not decode is replaced by its nearest already
            # decoded neighbour, never by zeros: a black frame inside a
            # three-channel group is a strong, entirely artificial edge.
            frames.append(frames[-1] if frames else np.zeros((IMG, IMG), np.float32))
            continue

        try:
            pixel_mm = float(ds.PixelSpacing[0])
        except Exception:
            pixel_mm = None

        pixels, applied = physical_crop(pixels, pixel_mm)
        cropped += int(applied)
        frames.append(cv2.resize(pixels, (IMG, IMG), interpolation=cv2.INTER_AREA))

    volume = normalise_laterality(np.stack(frames), plane, side)
    return normalise_intensity(volume), cropped == len(picks)


def read_slot_multi(files: list[str], plane: str, side: str | None, takes):
    """`read_slot` for SEVERAL slice counts at once: {n_take: (volume, cropped)}.

    Members trained at different slices-per-slot need different caches, and the
    two grids do not nest — `band_indices` is a linspace, so six points sit at
    fractions k/5 of the band and twelve at k/11, sharing only the endpoints.
    Subsampling a twelve-slice cache down to six would hand a model triplets
    spaced 5/11 of what it trained on: right shape, wrong anatomy, and a score
    that looks like a bad model rather than a bad cache.

    Building each grid with a separate `build_study` call would re-open every
    header in the study per grid, and header parsing is the part that opens
    ~800k files. So headers are read once, the UNION of the wanted indices is
    decoded once, and each grid is assembled from that shared pool. Six plus
    twelve is at most eighteen unique frames and usually fewer — the grids share
    both endpoints — against twenty-four for two independent passes.

    Each grid is then normalised and mirrored on its own frames, because
    `normalise_intensity` percentile-clips across the whole slot: sharing that
    step between grids would make the six-slice volume depend on which other
    grids happened to be requested.
    """
    import cv2
    import pydicom

    keyed = []
    for path in files:
        ds = read_header(path)
        if ds is not None:
            keyed.append((slice_key(ds), path, ds))
    if not keyed:
        return {k: (None, False) for k in takes}

    keyed.sort(key=lambda t: t[0])
    picks_of = {k: band_indices(len(keyed), k) for k in takes}

    decoded: dict[int, tuple] = {}
    for i in sorted({int(i) for picks in picks_of.values() for i in picks}):
        _key, path, ds = keyed[i]
        try:
            pixels = pydicom.dcmread(path).pixel_array.astype(np.float32)
        except Exception:                                   # noqa: BLE001
            decoded[i] = None                               # marked, not zeroed
            continue
        try:
            pixel_mm = float(ds.PixelSpacing[0])
        except Exception:                                   # noqa: BLE001
            pixel_mm = None
        pixels, applied = physical_crop(pixels, pixel_mm)
        decoded[i] = (cv2.resize(pixels, (IMG, IMG), interpolation=cv2.INTER_AREA),
                      applied)

    out = {}
    for k, picks in picks_of.items():
        frames, cropped = [], 0
        for i in picks:
            got = decoded[int(i)]
            if got is None:
                # Same rule as read_slot: an undecodable slice becomes its
                # nearest already-taken neighbour IN THIS GRID, never zeros. A
                # black frame inside a triplet is a strong artificial edge.
                frames.append(frames[-1] if frames
                              else np.zeros((IMG, IMG), np.float32))
                continue
            frames.append(got[0])
            cropped += int(got[1])
        volume = normalise_laterality(np.stack(frames), plane, side)
        out[k] = (normalise_intensity(volume), cropped == len(picks))
    return out


def build_study_multi(series_dir: str, study_uid: str, plane_of: dict, takes):
    """`build_study` for several slices-per-slot at once.

    Returns ({n_take: uint8 (N_SLOT, n_take, IMG, IMG)}, mask, row, infos,
    (crop_ok, crop_total)) — one volume stack per grid, sharing a single pass
    over the study's headers. The crop counters are reported from the largest
    grid; they are a health metric, and counting the same slot once per grid
    would only inflate the denominator.
    """
    from src.data.slots import SLOTS

    takes = tuple(sorted({int(k) for k in takes}))
    volumes = {k: np.zeros((N_SLOT, k, IMG, IMG), dtype=np.uint8) for k in takes}
    mask = np.zeros(N_SLOT, dtype=np.uint8)

    infos, files_of, fingerprint = probe_study(series_dir, study_uid, plane_of)
    side = study_side(infos)
    row = {"StudyInstanceUID": study_uid, "side": side or "",
           "fingerprint": fingerprint}

    crop_ok = crop_total = 0
    for slot_i, info in enumerate(pick_slot_series(infos)):
        name = SLOTS[slot_i][0]
        if info is None:
            row[name] = ""
            continue
        got = read_slot_multi(files_of[info.uid], info.plane, side, takes)
        if got[takes[-1]][0] is None:
            row[name] = ""
            continue
        for k in takes:
            volumes[k][slot_i] = got[k][0]
        mask[slot_i] = 1
        row[name] = info.uid
        crop_ok += int(got[takes[-1]][1])
        crop_total += 1

    return volumes, mask, row, infos, (crop_ok, crop_total)


def build_study(series_dir: str, study_uid: str, plane_of: dict):
    """One study -> (uint8 (N_SLOT, N_SLICE, IMG, IMG), mask, row).

    `row` carries the side, the scanner fingerprint and the series chosen for
    each slot, so a cache can be audited after the fact without re-reading a
    single DICOM.
    """
    volumes = np.zeros((N_SLOT, N_SLICE, IMG, IMG), dtype=np.uint8)
    mask = np.zeros(N_SLOT, dtype=np.uint8)

    infos, files_of, fingerprint = probe_study(series_dir, study_uid, plane_of)
    side = study_side(infos)
    row = {"StudyInstanceUID": study_uid, "side": side or "",
           "fingerprint": fingerprint}

    from src.data.slots import SLOTS

    crop_ok = crop_total = 0
    for slot_i, info in enumerate(pick_slot_series(infos)):
        name = SLOTS[slot_i][0]
        if info is None:
            row[name] = ""
            continue
        volume, fully_cropped = read_slot(files_of[info.uid], info.plane, side)
        if volume is None:
            row[name] = ""
            continue
        volumes[slot_i] = volume
        mask[slot_i] = 1
        row[name] = info.uid
        crop_ok += int(fully_cropped)
        crop_total += 1

    return volumes, mask, row, infos, (crop_ok, crop_total)
