"""Cross-validation that doesn't lie to you.

The single most important file in the modelling pipeline. A forum probe
(docs/01-competition.md) measured DICOM metadata alone at **0.6516 macro AUC on
random folds but 0.5981 under scanner-grouped folds** — a 0.053 gap that is pure
site memorisation. Studies from the same scanner leak between train and
validation, and the model learns "site 12 images a lot of arthritic knees"
instead of how to read a knee.

0.053 is roughly the difference between a top-10 finish and mid-table. Random
KFold here is not a minor methodological wobble; it is the mistake that makes
a solution look finished when it isn't.

So: **GroupKFold on the scanner fingerprint, always.** If CV and the public
leaderboard disagree, trust CV — with ~1,300 test studies split public/private,
the public LB is a small, noisy sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Tags that together identify a physical scanner + software revision.
#: Finer-grained than "institution" on purpose: the same hospital running two
#: scanners produces two distinguishable image distributions, and grouping only
#: by site would let one leak into the other.
FINGERPRINT_TAGS = (
    "Manufacturer",
    "ManufacturerModelName",
    "SoftwareVersions",
    "ImagingFrequency",
    "ReceiveCoilName",
)


def study_fingerprint(root: Path | str, study_uid: str, series_uid: str) -> str:
    """Scanner fingerprint for one study, read from a single slice header.

    Reads with `stop_before_pixels=True` — orders of magnitude faster than
    decoding, which matters when this runs over 4,407 studies.
    """
    import pydicom

    series_dir = Path(root) / study_uid / series_uid
    for dcm_path in series_dir.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        except Exception:
            continue
        parts = []
        for tag in FINGERPRINT_TAGS:
            value = getattr(ds, tag, "")
            parts.append(str(value).strip() if value is not None else "")
        return "|".join(parts)
    return "unknown"


def build_fingerprints(root: Path | str, series_meta, studies) -> dict[str, str]:
    """Fingerprint every study, using its first available series."""
    fingerprints: dict[str, str] = {}
    for study_uid in studies:
        rows = series_meta[series_meta["StudyInstanceUID"] == study_uid]
        if len(rows) == 0:
            fingerprints[study_uid] = "unknown"
            continue
        fingerprints[study_uid] = study_fingerprint(
            root, study_uid, str(rows.iloc[0]["SeriesInstanceUID"])
        )
    return fingerprints


def grouped_folds(groups, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Fold assignment with every group confined to one fold.

    Uses GroupKFold's size-balancing rather than a random group split: the
    fingerprint distribution is heavily skewed (the top 20 of 265 fingerprints
    cover ~45% of studies), so a naive random assignment of groups produces
    wildly uneven folds.
    """
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    folds = np.full(len(groups), -1, dtype=int)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (_, val_idx) in enumerate(splitter.split(groups, groups=groups)):
        folds[val_idx] = fold
    return folds


def macro_auc(y_true, y_pred) -> tuple[float, dict[str, float]]:
    """Competition metric: unweighted mean AUC over the twelve labels.

    Labels with only one class present in a fold are skipped rather than
    scored 0.5 — on rare labels a fold can genuinely contain no positives, and
    counting that as chance-level would misreport the model.
    """
    from sklearn.metrics import roc_auc_score

    from src.labels import TARGETS

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    per_label: dict[str, float] = {}
    for i, label in enumerate(TARGETS):
        column = y_true[:, i]
        if len(np.unique(column)) < 2:
            continue
        per_label[label] = float(roc_auc_score(column, y_pred[:, i]))
    macro = float(np.mean(list(per_label.values()))) if per_label else float("nan")
    return macro, per_label
