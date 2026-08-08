"""Measuring the extractor against the ~58 gold-labelled studies.

Those studies are the only ground truth in the competition. Two rules:

1. **Measure on them, never train on them.** With ~58 samples it is trivially
   easy to tune patterns until they memorise the set and generalise to nothing.
2. **Watch prevalence too, not just agreement.** 58 studies is far too few to
   trust per-label F1 on rare findings — Fracture may have one positive. Checking
   that extracted prevalence across *all* studies looks clinically sane is a
   weaker but much better-powered signal.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.labels import TARGETS


@dataclass
class LabelReport:
    label: str
    support: int          # gold positives
    predicted: int        # predicted positives at threshold
    true_positive: int
    false_positive: int
    false_negative: int
    auc: float | None = None

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else float("nan")

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p != p or r != r or (p + r) == 0:  # NaN-safe
            return float("nan")
        return 2 * p * r / (p + r)


def evaluate(gold, predicted, threshold: float = 0.5, id_column: str = "StudyInstanceUID"):
    """Compare predicted scores against gold labels.

    Both arguments are DataFrames with `id_column` plus the twelve label columns.
    Only rows present in `gold` with non-null labels are scored.

    Returns (list[LabelReport], macro_auc | None).
    """
    import numpy as np
    import pandas as pd

    merged = gold.merge(predicted, on=id_column, suffixes=("_gold", "_pred"))
    reports: list[LabelReport] = []
    aucs: list[float] = []

    for label in TARGETS:
        gold_col, pred_col = f"{label}_gold", f"{label}_pred"
        if gold_col not in merged or pred_col not in merged:
            continue

        mask = merged[gold_col].notna()
        truth = merged.loc[mask, gold_col].astype(int).to_numpy()
        score = merged.loc[mask, pred_col].astype(float).to_numpy()
        if truth.size == 0:
            continue

        hard = (score >= threshold).astype(int)
        auc = _safe_auc(truth, score)
        if auc is not None:
            aucs.append(auc)

        reports.append(
            LabelReport(
                label=label,
                support=int(truth.sum()),
                predicted=int(hard.sum()),
                true_positive=int(np.sum((hard == 1) & (truth == 1))),
                false_positive=int(np.sum((hard == 1) & (truth == 0))),
                false_negative=int(np.sum((hard == 0) & (truth == 1))),
                auc=auc,
            )
        )

    macro = float(sum(aucs) / len(aucs)) if aucs else None
    return reports, macro


def _safe_auc(truth, score) -> float | None:
    """AUC, or None when a label has only one class present."""
    from sklearn.metrics import roc_auc_score

    if len(set(truth.tolist())) < 2:
        return None
    return float(roc_auc_score(truth, score))


def _fmt(value: float | None) -> str:
    """Format a metric, tolerating None and NaN."""
    if value is None or value != value:
        return "-"
    return f"{value:.3f}"


def format_report(reports: list[LabelReport], macro: float | None) -> str:
    lines = [
        f"{'label':<18}{'n+':>4}{'pred':>6}{'TP':>4}{'FP':>4}{'FN':>4}"
        f"{'prec':>7}{'rec':>7}{'F1':>7}{'AUC':>7}",
        "-" * 74,
    ]
    for r in reports:
        lines.append(
            f"{r.label:<18}{r.support:>4}{r.predicted:>6}{r.true_positive:>4}"
            f"{r.false_positive:>4}{r.false_negative:>4}"
            f"{_fmt(r.precision):>7}{_fmt(r.recall):>7}{_fmt(r.f1):>7}"
            f"{_fmt(r.auc):>7}"
        )
    lines.append("-" * 74)
    lines.append(
        f"macro AUC over labels with both classes present: {_fmt(macro)}"
        if macro is not None
        else "macro AUC: n/a (no label had both classes present)"
    )
    return "\n".join(lines)


def prevalence(predicted, threshold: float = 0.5) -> dict[str, float]:
    """Fraction of studies predicted positive per label.

    Sanity check against clinical expectation. If Fracture comes out at 40%,
    the extractor is broken regardless of what the gold studies say.
    """
    return {
        label: float((predicted[label].astype(float) >= threshold).mean())
        for label in TARGETS
        if label in predicted
    }


def disagreements(gold, predicted, reports_text, threshold: float = 0.5,
                  id_column: str = "StudyInstanceUID"):
    """Rows where extraction and gold disagree, with the report text attached.

    The single most useful debugging output there is: read twenty of these and
    the pattern gaps become obvious.
    """
    merged = gold.merge(predicted, on=id_column, suffixes=("_gold", "_pred"))
    merged = merged.merge(reports_text, on=id_column, how="left")

    rows = []
    for _, row in merged.iterrows():
        for label in TARGETS:
            gold_value = row.get(f"{label}_gold")
            if gold_value != gold_value or gold_value is None:  # NaN-safe
                continue
            predicted_hard = float(row.get(f"{label}_pred", 0.0)) >= threshold
            if bool(int(gold_value)) != predicted_hard:
                rows.append(
                    {
                        id_column: row[id_column],
                        "label": label,
                        "gold": int(gold_value),
                        "predicted": float(row.get(f"{label}_pred", 0.0)),
                        "kind": "missed" if gold_value else "spurious",
                        "Report": row.get("Report", ""),
                    }
                )

    import pandas as pd

    return pd.DataFrame(rows)
