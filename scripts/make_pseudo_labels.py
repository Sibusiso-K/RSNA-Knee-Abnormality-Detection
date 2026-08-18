"""Blend text-derived labels with out-of-fold imaging predictions.

    python scripts/make_pseudo_labels.py --oof oof_fold*.csv \
        --labels data/labels_blend_v1.csv --out data/labels_pseudo_v1.csv

Why this exists
---------------
`docs/00-state.md` records the observation this is built on: **Synovitis scores
0.836 from imaging while its text labels score 0.630 against gold.** The model
learns that finding markedly better than the targets it was trained on, which
means the imaging predictions carry information the text extraction missed.
Folding some of it back into the labels attacks the 0.8930 label ceiling
directly, rather than working underneath it.

The three ways this goes wrong, and what is done about each
----------------------------------------------------------
1. **Circularity.** A prediction from a model that trained on the study is not
   evidence about that study. Only out-of-fold predictions are accepted, and
   `--oof` inputs must carry a `fold` column so an accidental in-fold dump is
   caught rather than silently averaged in. A study appearing in two folds
   means the fold assignment changed between runs and the OOF set is invalid.

2. **Overfitting the 58 gold studies.** Gold is the only ground truth and it is
   tiny. `docs/00-state.md` already refuses per-label *selection* against it for
   this reason. So the blend weight here is **not fitted**: alpha is either a
   single constant for every label, or set per label from the *text-vs-imaging
   gap measured during training*, which is a property of the fold scores rather
   than of the gold set. Gold is used only to REPORT what happened, never to
   choose it. `--report-gold` prints the before/after and nothing in the
   pipeline reads that number.

3. **Believing the result.** n=58 cannot resolve a change under roughly 0.03
   (the LLM's +0.024 was already inside that band). The gold score printed here
   is a sanity check for gross damage, not evidence of improvement. **The real
   test is a fold-0 retrain against the same cache**, which is ~35 min.

Round 2 and beyond
-------------------
Round 1 (`labels_pseudo_a5.csv`, alpha 0.5) measured +0.0111 mean CV across all 5
folds — every fold improved, none regressed. That model's own OOF predictions
are a new, independent input: round 2 blends `labels_pseudo_a5.csv` (not the
original `labels_blend_v1.csv`) with fresh OOF from the round-1 checkpoints.

This is where the circularity risk stops being hypothetical: round 2's OOF
comes from a model that was itself trained on blended labels, so "the imaging
model agrees with the labels" is now partly true by construction rather than
purely informative. Two flags exist for exactly this:

    --drift-ref  labels_blend_v1.csv   # the ORIGINAL text labels, never
                                        # last round's output
    --prev-oof   round1_oof_fold*.csv  # last round's OOF, to see if this
                                        # round moved at all

If `--drift-ref` shows the output moving steadily further from the original
text with each round, that is the labels walking away from their only
external signal, not converging on truth — round CV can rise the whole time
this happens, because CV is scored against a moving target too closely
related to the training signal. If `--prev-oof` shows near-zero movement, the
model has converged and another round buys nothing but that same drift risk
for no gain.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.labels import TARGETS  # noqa: E402

ID = "StudyInstanceUID"


def load_oof(patterns: list[str]) -> pd.DataFrame:
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        raise SystemExit(f"no OOF files matched {patterns}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = {ID, "fold"} - set(frame.columns)
        if missing:
            raise SystemExit(f"{path} lacks {missing} — is it really an OOF dump?")
        frames.append(frame)
        print(f"  {path}: {len(frame):5d} studies, fold {sorted(frame.fold.unique())}")

    oof = pd.concat(frames, ignore_index=True)

    # A study in two folds means the fold assignment moved between runs, so at
    # least one of these predictions is in-fold. Averaging them would quietly
    # leak; refuse instead.
    dupes = oof[ID].duplicated().sum()
    if dupes:
        raise SystemExit(
            f"{dupes} studies appear in more than one OOF file. The fold "
            f"assignment differs between runs, so these are not all "
            f"out-of-fold. Rebuild the OOF set from a single fold split."
        )
    return oof


def blend(labels: pd.DataFrame, oof: pd.DataFrame, alpha: float,
          per_label: dict[str, float] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = labels.merge(oof, on=ID, how="left", suffixes=("", "_oof"))
    covered = merged[f"{TARGETS[0]}_oof"].notna()
    print(f"\nstudies with an OOF prediction: {int(covered.sum()):,} of {len(merged):,}")

    out = labels.copy()
    rows = []
    for target in TARGETS:
        a = (per_label or {}).get(target, alpha)
        text = merged[target].to_numpy(dtype=np.float32)
        pred = merged[f"{target}_oof"].to_numpy(dtype=np.float32)
        # Studies without an OOF prediction keep their text label untouched:
        # absence of a prediction is not evidence about the finding.
        new = np.where(covered, a * text + (1.0 - a) * pred, text)
        out[target] = new
        rows.append({
            "target": target, "alpha": a,
            "text_mean": float(np.nanmean(text)),
            "oof_mean": float(np.nanmean(pred)),
            "blend_mean": float(np.nanmean(new)),
            "shift": float(np.nanmean(np.abs(new - text))),
        })
    return out, pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", nargs="+", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--alpha", type=float, default=0.7,
        help="weight on the TEXT label. 1.0 keeps labels unchanged, 0.0 "
             "replaces them with model predictions. Default 0.7 is "
             "deliberately conservative: the text labels are the only signal "
             "with an external source, and the model was trained on them, so "
             "it cannot be trusted to overrule them wholesale.",
    )
    ap.add_argument(
        "--trust-imaging", nargs="*", default=[],
        help="labels where imaging beat text by a wide margin during training "
             "(e.g. Synovitis). These get alpha reduced by --trust-delta. "
             "Choose them from the per-label FOLD scores, never from gold — "
             "picking them by gold is fitting the only ground truth there is.",
    )
    ap.add_argument("--trust-delta", type=float, default=0.2)
    ap.add_argument("--report-gold", default="",
                    help="train.csv, to print before/after gold AUC. REPORTING "
                         "ONLY — nothing downstream reads this number.")
    ap.add_argument(
        "--drift-ref", default="",
        help="Reference labels to measure drift against — normally the ORIGINAL "
             "labels_blend_v1.csv. On round 2+ this is the guard that matters: "
             "each round blends in predictions from a model trained on the "
             "previous round's output, so the targets can walk away from the "
             "external text signal while every internal metric still improves. "
             "Reported, never used to change the output.",
    )
    ap.add_argument(
        "--prev-oof", nargs="*", default=[],
        help="Previous round's OOF files. Reports how far this round's "
             "predictions moved. If they barely moved, the round adds nothing "
             "and the extra drift buys no new information.",
    )
    args = ap.parse_args()

    print("loading OOF predictions:")
    oof = load_oof(args.oof)
    labels = pd.read_csv(args.labels)
    print(f"labels: {args.labels} ({len(labels):,} studies)")

    per_label = {t: max(0.0, args.alpha - args.trust_delta)
                 for t in args.trust_imaging if t in TARGETS}
    unknown = set(args.trust_imaging) - set(TARGETS)
    if unknown:
        raise SystemExit(f"unknown target(s) in --trust-imaging: {unknown}")
    if per_label:
        print(f"reduced alpha for: {sorted(per_label)} "
              f"({args.alpha} -> {args.alpha - args.trust_delta})")

    out, summary = blend(labels, oof, args.alpha, per_label)
    print("\nper-label effect:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if args.drift_ref:
        report_drift(args.drift_ref, out)
    if args.prev_oof:
        report_oof_movement(args.prev_oof, oof)
    if args.report_gold:
        report_gold(args.report_gold, labels, out)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print("\nThis is NOT yet evidence of improvement. n=58 gold cannot resolve "
          "a change under ~0.03.\nRetrain fold 0 against the same cache and "
          "compare CV — that is the real test.")


def _macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Macro AUC, skipping labels that are single-class in this slice.

    Matches the training harness: a label with no negatives (or no positives)
    among 58 studies has no defined AUC, and scoring it as 0.5 would drag the
    macro toward the middle for a reason that has nothing to do with quality.
    """
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    scores = []
    for i in range(y.shape[1]):
        col = y[:, i]
        keep = ~np.isnan(col)
        if keep.sum() == 0 or len(np.unique(col[keep])) < 2:
            continue
        scores.append(roc_auc_score(col[keep], p[keep, i]))
    return float(np.mean(scores)) if scores else float("nan")


def report_drift(ref_path: str, out: pd.DataFrame) -> None:
    """How far THIS round's output has moved from the original text labels.

    Round 1 blends text with OOF from a model trained on text. Round 2 blends
    round-1 output with OOF from a model trained on round-1 output — so the
    external signal (the text extraction) can shrink round over round while
    every internal metric (fold CV, even gold if it happens to move) keeps
    looking fine. This is the number that would catch that, because it is the
    only one measured against something that never came from a checkpoint.
    """
    ref = pd.read_csv(ref_path)
    merged = out.merge(ref, on=ID, suffixes=("", "_ref"))
    diffs = [np.abs(merged[t] - merged[f"{t}_ref"]).mean() for t in TARGETS]
    print(f"\ndrift from {ref_path} (the ORIGINAL text labels, not last round's output):")
    print(f"  mean |this_round - original_text| = {np.mean(diffs):.4f}")
    print("  (compare across rounds - if this keeps climbing, the labels are "
          "walking away from the text signal, not converging)")


def report_oof_movement(prev_paths: list[str], oof: pd.DataFrame) -> None:
    """How much THIS round's OOF predictions moved from last round's.

    If a new round of training barely changes what the model predicts, the
    round has converged and blending it in again buys nothing but drift risk.
    """
    prev = load_oof(prev_paths)
    merged = oof.merge(prev, on=ID, suffixes=("", "_prev"))
    diffs = [np.abs(merged[t] - merged[f"{t}_prev"]).mean() for t in TARGETS]
    print(f"\nOOF movement vs previous round: mean |delta| = {np.mean(diffs):.4f}")
    print("  (near zero means the model's predictions have stopped changing - "
          "further rounds are unlikely to add anything)")


def report_gold(train_csv: str, before: pd.DataFrame, after: pd.DataFrame) -> None:
    truth = pd.read_csv(train_csv)
    gold = truth[truth[TARGETS].notna().any(axis=1)]
    if gold.empty:
        print("\nno gold rows found — skipping gold report")
        return
    print(f"\ngold studies: {len(gold)}  (REPORTING ONLY — not used to choose anything)")
    for name, frame in (("before", before), ("after ", after)):
        merged = gold[[ID] + TARGETS].merge(frame, on=ID, suffixes=("_true", "_pred"))
        y = merged[[f"{t}_true" for t in TARGETS]].to_numpy(dtype=float)
        p = merged[[f"{t}_pred" for t in TARGETS]].to_numpy(dtype=float)
        print(f"  {name}: macro AUC {_macro_auc(y, p):.4f}  (n={len(merged)})")


if __name__ == "__main__":
    main()
