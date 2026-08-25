#!/usr/bin/env python
"""Run the LLM-based report -> label extractor against a local Ollama model.

    python scripts/extract_labels_llm.py --demo              # sanity check, no data needed
    python scripts/extract_labels_llm.py --evaluate           # score against the 58 gold studies
    python scripts/extract_labels_llm.py --compare             # LLM vs rule extractor, side by side
    python scripts/extract_labels_llm.py --api-evaluate --confirm-spend
                                                                # frontier API vs the 58 gold studies

--api-evaluate calls a COMMERCIALLY HOSTED LLM (default: Anthropic, see
src/extract/api.py) and is real, billed API usage — 58 short calls, expected
cost is cents. --confirm-spend is required and does nothing else; it exists
so this can't fire by accident. Requires ANTHROPIC_API_KEY. Permitted under
the host's 2026-08-08 ruling (discussion/733965); see
docs/08-model-and-rules.md for the "minimal cost" constraint this must stay
inside before scaling to the full 4,407-report corpus.

Requires an Ollama server running locally (`ollama serve`, or the desktop
app) with a model pulled — defaults to llama3.2. Pick a different one with
--model, e.g. --model qwen2.5-coder:7b.

Deliberately scoped to the 58 gold studies, not the full 4,407-report
corpus: this machine has no GPU (see docs/07-environment.md), and CPU
inference at one HTTP call per report makes a full run impractical here.
That run belongs on a Kaggle GPU notebook — see docs/00-state.md for status.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extract import RuleExtractor  # noqa: E402
from src.extract.llm import DEFAULT_MODEL, LLMExtractor, LLMExtractorConfig  # noqa: E402
from src.labels import ID_COLUMN, TARGETS  # noqa: E402
from scripts.extract_labels import DEMO_REPORTS  # noqa: E402

DATA = ROOT / "data"


def build_extractor(args: argparse.Namespace) -> LLMExtractor:
    return LLMExtractor(LLMExtractorConfig(model=args.model, timeout_seconds=args.timeout))


def _check_server(extractor: LLMExtractor) -> bool:
    import requests

    try:
        requests.get(f"{extractor.config.base_url}/api/tags", timeout=5).raise_for_status()
        return True
    except requests.RequestException as exc:
        print(
            f"Can't reach Ollama at {extractor.config.base_url}: {exc}\n"
            "Start it with `ollama serve` (or the desktop app), and make "
            f"sure the model is pulled: `ollama pull {extractor.config.model}`.",
            file=sys.stderr,
        )
        return False


def cmd_demo(args: argparse.Namespace) -> int:
    extractor = build_extractor(args)
    if not _check_server(extractor):
        return 2

    for name, report in DEMO_REPORTS:
        start = time.time()
        extraction = extractor.extract(report, study_uid=name)
        elapsed = time.time() - start
        positive = {
            label: round(score, 2)
            for label, score in extraction.scores.items()
            if score > 0
        }
        print(f"\n=== {name} ({elapsed:.1f}s) ===")
        print(f"  {report}")
        print(f"  -> {positive or '(nothing)'}")
    return 0


def _load_gold(limit: int | None = None):
    import pandas as pd

    path = DATA / "train.csv"
    if not path.exists():
        print(f"{path} not found — run scripts/download_data.sh first.", file=sys.stderr)
        raise SystemExit(2)
    train = pd.read_csv(path)
    present = [label for label in TARGETS if label in train.columns]
    gold = train[train[present].notna().any(axis=1)]
    gold = gold[[ID_COLUMN, "Report", *present]]
    if limit:
        # Shortest reports first: --limit is for a quick smoke test on this
        # CPU-only machine, not a real benchmark subset, so bias toward fast
        # cases rather than risk the first study being a multi-thousand-
        # character worst case.
        gold = gold.sort_values("Report", key=lambda s: s.str.len()).head(limit)
    return gold


def cmd_evaluate(args: argparse.Namespace) -> int:
    from src.extract.evaluate import evaluate, format_report

    extractor = build_extractor(args)
    if not _check_server(extractor):
        return 2

    gold = _load_gold(args.limit)
    print(f"gold studies: {len(gold)}  (model: {args.model})")

    start = time.time()
    predicted = extractor.extract_frame(gold, id_column=ID_COLUMN)
    elapsed = time.time() - start
    print(f"extraction took {elapsed:.0f}s ({elapsed / len(gold):.1f}s/study)\n")

    reports, macro = evaluate(gold, predicted, args.threshold, ID_COLUMN)
    print(format_report(reports, macro))

    out = DATA / f"llm_scores_{args.model.replace(':', '_')}.csv"
    predicted.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from src.extract.evaluate import evaluate, format_report

    llm = build_extractor(args)
    if not _check_server(llm):
        return 2

    gold = _load_gold(args.limit)
    print(f"gold studies: {len(gold)}\n")

    print(f"--- rule extractor (baseline) ---")
    rule_scores = RuleExtractor().extract_frame(gold, id_column=ID_COLUMN)
    rule_reports, rule_macro = evaluate(gold, rule_scores, args.threshold, ID_COLUMN)
    print(format_report(rule_reports, rule_macro))

    print(f"\n--- LLM extractor ({args.model}) ---")
    start = time.time()
    llm_scores = llm.extract_frame(gold, id_column=ID_COLUMN)
    elapsed = time.time() - start
    print(f"extraction took {elapsed:.0f}s ({elapsed / len(gold):.1f}s/study)\n")
    llm_reports, llm_macro = evaluate(gold, llm_scores, args.threshold, ID_COLUMN)
    print(format_report(llm_reports, llm_macro))

    print(f"\n=== summary ===")
    print(f"rule macro AUC: {rule_macro:.3f}" if rule_macro else "rule macro AUC: n/a")
    print(f"llm  macro AUC: {llm_macro:.3f}" if llm_macro else "llm  macro AUC: n/a")
    return 0


def cmd_api_evaluate(args: argparse.Namespace) -> int:
    if not args.confirm_spend:
        print(
            "--api-evaluate makes real, billed calls to a commercial LLM API.\n"
            "Re-run with --confirm-spend to proceed.",
            file=sys.stderr,
        )
        return 2

    from src.extract.api import DEFAULT_MODEL as DEFAULT_API_MODEL
    from src.extract.api import APIExtractor, APIExtractorConfig
    from src.extract.evaluate import evaluate, format_report

    model = args.api_model or DEFAULT_API_MODEL
    try:
        extractor = APIExtractor(APIExtractorConfig(model=model, timeout_seconds=args.timeout))
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    gold = _load_gold(args.limit)
    print(f"gold studies: {len(gold)}  (model: {model})")

    start = time.time()
    predicted = extractor.extract_frame(gold, id_column=ID_COLUMN)
    elapsed = time.time() - start
    print(f"extraction took {elapsed:.0f}s ({elapsed / len(gold):.1f}s/study)\n")

    reports, macro = evaluate(gold, predicted, args.threshold, ID_COLUMN)
    print(format_report(reports, macro))

    out = DATA / f"api_scores_{model.replace(':', '_')}.csv"
    predicted.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--api-evaluate", action="store_true")
    parser.add_argument(
        "--confirm-spend", action="store_true",
        help="required alongside --api-evaluate; makes real, billed API calls",
    )
    parser.add_argument("--api-model", default=None, help="override the API model id")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only run the N shortest gold reports — smoke-test before a full run",
    )
    args = parser.parse_args()

    if args.demo:
        return cmd_demo(args)
    if args.compare:
        return cmd_compare(args)
    if args.api_evaluate:
        return cmd_api_evaluate(args)
    if args.evaluate:
        return cmd_evaluate(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
