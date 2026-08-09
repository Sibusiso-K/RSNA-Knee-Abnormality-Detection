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


def _clean_tag(tag: str, value) -> str:
    """Normalise one tag value into a stable, coarse string.

    `ImagingFrequency` needs special handling and it is the whole reason this
    function exists. It is the Larmor frequency in MHz (~63.9 at 1.5 T, ~127.7
    at 3 T) and it **drifts in the decimals between scanning sessions on the
    same physical magnet**. Used raw it is effectively a per-study nonce: the
    first real run produced **3,229 distinct fingerprints across 4,349
    studies** — nearly one group per study, which silently reduces GroupKFold
    to random KFold and reinstates exactly the ~0.053 AUC of site leakage the
    grouping exists to prevent.

    Rounding to whole MHz keeps the distinction that matters (1.5 T vs 3 T,
    and different magnets) while collapsing session-to-session drift.
    """
    if value is None:
        return ""
    if tag == "ImagingFrequency":
        try:
            return str(int(round(float(value))))
        except (TypeError, ValueError):
            return ""
    # SoftwareVersions is often multi-valued; str() of a pydicom MultiValue is
    # order-sensitive, so join explicitly for stability.
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return ",".join(str(v).strip() for v in value)
    return str(value).strip()


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
        return "|".join(
            _clean_tag(tag, getattr(ds, tag, "")) for tag in FINGERPRINT_TAGS
        )
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


def check_grouping(groups, warn_ratio: float = 0.25) -> dict:
    """Fail loudly when the grouping key is too fine to protect anything.

    A fingerprint approaching one-group-per-study makes GroupKFold identical to
    random KFold while still *looking* rigorous — the worst kind of bug,
    because every downstream number stays plausible. This is a tripwire for
    that, added after `ImagingFrequency` drift produced 3,229 groups over 4,349
    studies on the first real run.
    """
    groups = np.asarray(groups)
    _, counts = np.unique(groups, return_counts=True)
    stats = {
        "n_studies": int(len(groups)),
        "n_groups": int(len(counts)),
        "ratio": float(len(counts) / max(len(groups), 1)),
        "largest": sorted(counts.tolist(), reverse=True)[:10],
        "singletons": int((counts == 1).sum()),
    }
    print(
        f"  grouping: {stats['n_groups']} groups / {stats['n_studies']} studies "
        f"(ratio {stats['ratio']:.2f}), singletons {stats['singletons']}, "
        f"largest {stats['largest'][:5]}"
    )
    if stats["ratio"] > warn_ratio:
        print(
            "  *** WARNING: grouping key is near-unique per study. GroupKFold "
            "is providing little or no protection here — treat any CV number "
            "from this run as OPTIMISTIC (random-fold equivalent). ***"
        )
    return stats


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


def macro_auc(y_true, y_pred, threshold: float = 0.5) -> tuple[float, dict[str, float]]:
    """Competition metric: unweighted mean AUC over the twelve labels.

    `y_true` is **binarised at `threshold`** before scoring. Our training
    targets are deliberately soft (a hedged "possible tear" scores ~0.6 — see
    docs/04-method.md), and `roc_auc_score` rejects continuous ground truth
    outright with "continuous format is not supported". The real competition
    labels are binary, so thresholding is what makes this CV number comparable
    to the leaderboard rather than a different metric wearing the same name.

    Labels with only one class present in a fold are skipped rather than
    scored 0.5 — on rare labels a fold can genuinely contain no positives, and
    counting that as chance-level would misreport the model.
    """
    from sklearn.metrics import roc_auc_score

    from src.labels import TARGETS

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    per_label: dict[str, float] = {}
    for i, label in enumerate(TARGETS):
        column = (y_true[:, i] >= threshold).astype(int)
        if len(np.unique(column)) < 2:
            continue
        per_label[label] = float(roc_auc_score(column, y_pred[:, i]))
    macro = float(np.mean(list(per_label.values()))) if per_label else float("nan")
    return macro, per_label
