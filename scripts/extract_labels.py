#!/usr/bin/env python
"""Run the report -> label extractor.

    python scripts/extract_labels.py --demo        # no data needed; sanity check
    python scripts/extract_labels.py               # extract to data/labels_v1.csv
    python scripts/extract_labels.py --evaluate    # score against the gold studies
    python scripts/extract_labels.py --audit 20    # eyeball mentions in 20 reports
    python scripts/extract_labels.py --disagree    # dump gold/extraction conflicts

`--demo` works before the competition data exists and is the fastest way to see
whether a pattern change did what you intended.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extract import ExtractorConfig, RuleExtractor  # noqa: E402
from src.labels import ID_COLUMN, TARGETS  # noqa: E402

DATA = ROOT / "data"

#: Synthetic reports covering the traps the extractor must not fall into.
#: These are invented, not competition data.
DEMO_REPORTS: list[tuple[str, str]] = [
    (
        "plain positive",
        "FINDINGS: Complete tear of the anterior cruciate ligament. "
        "Moderate joint effusion.",
    ),
    (
        "negation",
        "FINDINGS: No evidence of anterior cruciate ligament tear. "
        "The menisci are intact. No joint effusion.",
    ),
    (
        "negation with scope break",
        "IMPRESSION: No joint effusion, but there is a tear of the "
        "posterior horn of the medial meniscus.",
    ),
    (
        "uncertainty",
        "IMPRESSION: Possible tear of the lateral meniscus. "
        "Cannot exclude a small Baker's cyst.",
    ),
    (
        "laterality both",
        "FINDINGS: Degenerative changes of the medial and lateral "
        "tibiofemoral compartments.",
    ),
    (
        "patellofemoral routing",
        "IMPRESSION: Patellofemoral osteoarthritis with cartilage loss. "
        "Medial compartment osteoarthritis.",
    ),
    (
        "bone findings",
        "FINDINGS: Bone marrow oedema in the lateral femoral condyle "
        "consistent with contusion. No fracture.",
    ),
    (
        "spanish",
        "HALLAZGOS: Rotura del ligamento cruzado anterior. "
        "Derrame articular moderado. Sin fractura.",
    ),
    (
        "french",
        "CONCLUSION: Dechirure du menisque interne. "
        "Pas d'epanchement articulaire.",
    ),
    (
        "german",
        "BEURTEILUNG: Ruptur des vorderen Kreuzbandes. "
        "Kein Gelenkerguss. Bakerzyste.",
    ),
]


def build_extractor(args: argparse.Namespace) -> RuleExtractor:
    return RuleExtractor(
        ExtractorConfig(
            uncertain_score=args.uncertain_score,
            unresolved_laterality=args.unresolved_laterality,
        )
    )


def cmd_demo(args: argparse.Namespace) -> int:
    extractor = build_extractor(args)
    for name, report in DEMO_REPORTS:
        extraction = extractor.extract(report, study_uid=name)
        positive = {
            label: round(score, 2)
            for label, score in extraction.scores.items()
            if score > 0
        }
        print(f"\n=== {name} ===")
        print(f"  {report}")
        print(f"  -> {positive or '(nothing)'}")
        for mention in extraction.mentions:
            print(
                f"     · {mention.concept:<10} {mention.polarity:<9}"
                f" side={mention.laterality or '-':<14} '{mention.text}'"
            )
        for mention in extraction.unresolved:
            print(f"     ! {mention.concept} dropped: no laterality — '{mention.text}'")
    return 0


def _load_train():
    import pandas as pd

    path = DATA / "train.csv"
    if not path.exists():
        print(
            f"{path} not found.\n"
            "Join the competition, place kaggle.json, then run "
            "`bash scripts/download_data.sh`. See docs/07-environment.md.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return pd.read_csv(path)


def cmd_extract(args: argparse.Namespace) -> int:
    train = _load_train()
    extractor = build_extractor(args)
    scores = extractor.extract_frame(train, id_column=ID_COLUMN)

    DATA.mkdir(exist_ok=True)
    out = DATA / args.output
    scores.to_csv(out, index=False)
    print(f"wrote {out}  ({len(scores)} studies)")

    from src.extract.evaluate import prevalence

    print("\nextracted prevalence (sanity-check against clinical expectation):")
    for label, rate in prevalence(scores, args.threshold).items():
        print(f"  {label:<18}{rate:6.1%}")
    return 0


def _gold_rows(train):
    """Studies carrying real per-condition labels (~58 of them)."""
    present = [label for label in TARGETS if label in train.columns]
    if not present:
        print("train.csv has no label columns", file=sys.stderr)
        raise SystemExit(2)
    labelled = train[train[present].notna().any(axis=1)]
    return labelled[[ID_COLUMN, *present]]


def cmd_evaluate(args: argparse.Namespace) -> int:
    from src.extract.evaluate import evaluate, format_report

    train = _load_train()
    gold = _gold_rows(train)
    print(f"gold studies: {len(gold)}")

    extractor = build_extractor(args)
    predicted = extractor.extract_frame(
        train[train[ID_COLUMN].isin(gold[ID_COLUMN])], id_column=ID_COLUMN
    )
    reports, macro = evaluate(gold, predicted, args.threshold, ID_COLUMN)
    print()
    print(format_report(reports, macro))
    print(
        "\nReminder: ~58 studies is too few to trust per-label F1 on rare "
        "findings. Use this to catch gross errors, not to fine-tune."
    )
    return 0


def cmd_disagree(args: argparse.Namespace) -> int:
    from src.extract.evaluate import disagreements

    train = _load_train()
    gold = _gold_rows(train)
    extractor = build_extractor(args)
    predicted = extractor.extract_frame(
        train[train[ID_COLUMN].isin(gold[ID_COLUMN])], id_column=ID_COLUMN
    )
    frame = disagreements(
        gold, predicted, train[[ID_COLUMN, "Report"]], args.threshold, ID_COLUMN
    )
    if frame.empty:
        print("no disagreements")
        return 0

    out = DATA / "disagreements.csv"
    frame.to_csv(out, index=False)
    print(f"{len(frame)} disagreements -> {out}\n")
    for _, row in frame.head(args.limit).iterrows():
        print(f"--- {row['label']} ({row['kind']}, score={row['predicted']:.2f})")
        print(f"    {str(row['Report'])[:400]}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    train = _load_train()
    sample = train.sample(min(args.audit, len(train)), random_state=0)
    extractor = build_extractor(args)

    for _, row in sample.iterrows():
        extraction = extractor.extract(row.get("Report") or "", str(row[ID_COLUMN]))
        print(f"\n=== {row[ID_COLUMN]} ===")
        print(f"  {str(row.get('Report'))[:500]}")
        for mention in extraction.mentions:
            print(
                f"     · {mention.concept:<10} {mention.polarity:<9}"
                f" side={mention.laterality or '-':<14} '{mention.text}'"
            )
        for mention in extraction.unresolved:
            print(f"     ! {mention.concept} dropped (no side) — '{mention.text}'")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run on synthetic reports")
    parser.add_argument("--evaluate", action="store_true", help="score against gold")
    parser.add_argument("--disagree", action="store_true", help="dump conflicts")
    parser.add_argument("--audit", type=int, metavar="N", help="inspect N reports")
    parser.add_argument("--output", default="labels_v1.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--uncertain-score", dest="uncertain_score", type=float, default=0.6)
    parser.add_argument(
        "--unresolved-laterality", choices=["drop", "both"], default="drop"
    )
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    if args.demo:
        return cmd_demo(args)
    if args.evaluate:
        return cmd_evaluate(args)
    if args.disagree:
        return cmd_disagree(args)
    if args.audit:
        return cmd_audit(args)
    return cmd_extract(args)


if __name__ == "__main__":
    raise SystemExit(main())
