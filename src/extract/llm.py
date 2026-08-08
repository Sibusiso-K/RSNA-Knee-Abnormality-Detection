"""LLM-based report extraction: report text in, twelve label confidences out.

Phase 1 step 2 — the rule extractor (rules.py) is the baseline this has to
beat. Where regex needs a language's vocabulary hand-written before it can
see a finding at all, an instruction-tuned multilingual LLM can in principle
read a language it was never explicitly given patterns for, and can make the
same negation/uncertainty/laterality judgment calls a human reader makes
rather than pattern-matching them. See docs/04-method.md.

Runs against a local Ollama server (http://localhost:11434) so report text
never leaves the machine — open weights only. This sidesteps the unresolved
forum question about whether commercial LLM APIs are permitted on
competition text at all (see docs/01-competition.md): a model running
entirely locally is safe under either reading of that rule.

Not a drop-in replacement for the rule extractor in one respect: it is slow.
CPU inference on this machine (no NVIDIA GPU — docs/07-environment.md) makes
a full 4,407-report run impractical; scripts/extract_labels_llm.py therefore
defaults to running against just the 58 gold studies, where a real
apples-to-apples comparison is still possible. The full-corpus run belongs on
a Kaggle GPU notebook.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import requests

from src.extract.types import StudyExtraction
from src.labels import TARGETS

DEFAULT_MODEL = "llama3.2"
DEFAULT_BASE_URL = "http://localhost:11434"

# Keep the label list and its exact spelling (including the apostrophe in
# "Baker's") front and center in the prompt — a model that paraphrases a key
# produces a JSON object we can't score.
_LABEL_LIST = "\n".join(f'  "{label}"' for label in TARGETS)

SYSTEM_PROMPT = f"""You are assisting with a research competition that scores knee MRI \
radiology reports for twelve findings. You will be given the free-text report of one \
study. Reports may be written in English, Spanish, Portuguese, French, German, Italian, \
Dutch, Turkish, Greek, Bulgarian, Croatian, or another European language — read the \
report in its original language, do not ask for translation.

Score each of these twelve findings with a confidence from 0.0 to 1.0 that it is \
PRESENT in this study:
{_LABEL_LIST}

Rules for scoring:
- 1.0 = clearly and explicitly stated as present.
- 0.0 = explicitly denied ("no evidence of...", "intact", "normal"), OR never
  mentioned anywhere in the report.
- 0.4-0.7 = hedged language ("possible", "cannot exclude", "suspicious for",
  a bare "?"). Use judgment on where in that range: a weak hedge like
  "probably represents X" belongs higher than a bare "?".
- Watch for negation scope: "no effusion, but a meniscal tear" does NOT
  negate the tear. "Intact", "normal", "unremarkable", "preserved" all negate
  the structure they describe even though they appear AFTER it.
- "Medial Meniscus" and "Lateral Meniscus" are separate labels — a bare
  "meniscal tear" with no stated side should not be scored highly on either
  unless the report gives no other way to tell.
- "Medial OA", "Lateral OA", and "PF OA" are three separate compartments
  (medial tibiofemoral, lateral tibiofemoral, patellofemoral). Osteoarthritis
  affecting "all three compartments" should score all three.
- Do not infer a finding from a diagnosis code or referral question/reason
  for the exam — only from what the radiologist actually observed and
  reported.

Respond with ONLY a JSON object mapping each of the twelve exact label names \
above to its confidence score. No other text, no markdown fences, no \
explanation."""


@dataclass
class LLMExtractorConfig:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.0
    timeout_seconds: int = 180
    max_retries: int = 2


@dataclass
class LLMExtractionResult:
    """Extends the plain score dict with what the model actually said, for
    auditing when the score disagrees with the rule extractor or with gold."""

    scores: dict[str, float]
    raw_response: str = ""
    parse_ok: bool = True
    error: str | None = None


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class LLMExtractor:
    """Extracts the twelve label confidences from a radiology report via a
    local open-weights LLM served by Ollama."""

    def __init__(self, config: LLMExtractorConfig | None = None) -> None:
        self.config = config or LLMExtractorConfig()

    # -- public API, mirrors RuleExtractor -------------------------------

    def extract(self, report: str, study_uid: str = "") -> StudyExtraction:
        result = StudyExtraction(study_uid=study_uid)
        if not report or not report.strip():
            result.scores = {label: 0.0 for label in TARGETS}
            return result

        llm_result = self._call(report)
        result.scores = llm_result.scores
        return result

    def extract_frame(
        self,
        frame,
        text_column: str = "Report",
        id_column: str = "StudyInstanceUID",
        progress: bool = True,
    ):
        """Run over a DataFrame of reports, returning a DataFrame of scores.

        One HTTP call per row, sequential — this is the CPU-inference-speed
        bottleneck the module docstring warns about. Fine for the 58 gold
        studies; not a tool for the full corpus on this machine.

        `progress=True` prints one line per study, flushed immediately. This
        is not optional decoration — a run that goes silent for 40+ minutes
        is indistinguishable from a hang. One report (a ~4,700-char worst
        case in this corpus, several times longer than typical) can push a
        7B CPU model well past a minute; without a per-study line there is no
        way to tell "slow" from "stuck" until it is too late to intervene.
        """
        import sys
        import time

        import pandas as pd

        rows = []
        total = len(frame)
        for i, (_, row) in enumerate(frame.iterrows(), 1):
            report = row.get(text_column) or ""
            uid = str(row.get(id_column, ""))
            if progress:
                print(
                    f"[{i}/{total}] {uid[-8:]} ({len(report)} chars)...",
                    end=" ",
                    flush=True,
                )
            start = time.time()
            extraction = self.extract(report, uid)
            if progress:
                print(f"{time.time() - start:.1f}s", flush=True)
            rows.append({id_column: row.get(id_column), **extraction.scores})
        return pd.DataFrame(rows, columns=[id_column, *TARGETS])

    # -- internals --------------------------------------------------------

    def _call(self, report: str) -> LLMExtractionResult:
        cfg = self.config
        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"REPORT:\n{report}"},
            ],
            "format": "json",  # Ollama constrains sampling to valid JSON
            "stream": False,
            "options": {"temperature": cfg.temperature},
        }

        last_error: str | None = None
        for _ in range(cfg.max_retries + 1):
            try:
                response = requests.post(
                    f"{cfg.base_url}/api/chat", json=payload, timeout=cfg.timeout_seconds
                )
                response.raise_for_status()
                content = response.json()["message"]["content"]
                return self._parse(content)
            except (requests.RequestException, KeyError, ValueError) as exc:
                last_error = str(exc)

        return LLMExtractionResult(
            scores={label: 0.0 for label in TARGETS},
            parse_ok=False,
            error=f"request failed after retries: {last_error}",
        )

    def _parse(self, content: str) -> LLMExtractionResult:
        parsed = self._try_json(content)
        if parsed is None:
            return LLMExtractionResult(
                scores={label: 0.0 for label in TARGETS},
                raw_response=content,
                parse_ok=False,
                error="no valid JSON object found in response",
            )

        scores: dict[str, float] = {}
        for label in TARGETS:
            value = parsed.get(label)
            scores[label] = _coerce_score(value)

        return LLMExtractionResult(scores=scores, raw_response=content, parse_ok=True)

    @staticmethod
    def _try_json(content: str) -> dict | None:
        try:
            return json.loads(content)
        except ValueError:
            pass
        # format="json" should make this unnecessary, but models sometimes
        # wrap the object in prose or a markdown fence regardless.
        match = _JSON_OBJECT.search(content)
        if match:
            try:
                return json.loads(match.group(0))
            except ValueError:
                return None
        return None


def _coerce_score(value) -> float:
    """Clamp whatever the model produced into [0, 1], defaulting missing or
    unparseable values to 0.0 (absent) rather than guessing."""
    if value is None:
        return 0.0
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))
