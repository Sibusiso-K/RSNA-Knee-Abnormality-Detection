"""LLM-based report extraction against a commercially hosted API.

Phase 1 step 2, frontier-LLM variant. The host ruling of 2026-08-08
(discussion/733965) permits sending report text to a commercial LLM API,
provided the service is "reasonably accessible to all participants and of
minimal cost" — see docs/08-model-and-rules.md. Report text is not used at
inference time (there is no `Report` column on the test set), so this is a
training-time, one-off job over the 4,407-report corpus, not part of the
scored pipeline.

Mirrors src/extract/llm.py's interface and prompt exactly so the two are a
fair comparison: same SYSTEM_PROMPT, same scoring rubric, same TARGETS
schema. The only difference is the transport — an HTTPS call to a hosted API
instead of a local Ollama server — and that this one costs real money per
call, so callers should validate on the 58 gold studies before spending on
the full corpus.

Requires ANTHROPIC_API_KEY in the environment. Uses the Messages API
directly (no SDK dependency) to keep this consistent with the rest of the
extract package, which has no other network dependencies beyond `requests`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

from src.extract.llm import SYSTEM_PROMPT, LLMExtractionResult, _coerce_score
from src.extract.types import StudyExtraction
from src.labels import TARGETS

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class APIExtractorConfig:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None  # falls back to ANTHROPIC_API_KEY
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_seconds: int = 60
    max_retries: int = 2


class APIExtractor:
    """Extracts the twelve label confidences from a radiology report via a
    commercially hosted LLM (default: Anthropic Messages API).

    Same public interface as LLMExtractor (extract / extract_frame) so
    scripts/extract_labels_llm.py's --compare can add this as a third column
    without new plumbing.
    """

    def __init__(self, config: APIExtractorConfig | None = None) -> None:
        self.config = config or APIExtractorConfig()
        key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "No API key. Set ANTHROPIC_API_KEY or pass api_key= explicitly. "
                "This extractor makes real, billed API calls — do not wire it "
                "into a script that runs unattended without that being obvious "
                "to whoever runs it."
            )
        self._api_key = key

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
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": f"REPORT:\n{report}"}],
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        last_error: str | None = None
        for _ in range(cfg.max_retries + 1):
            try:
                response = requests.post(
                    f"{cfg.base_url}/v1/messages",
                    json=payload,
                    headers=headers,
                    timeout=cfg.timeout_seconds,
                )
                response.raise_for_status()
                content = response.json()["content"][0]["text"]
                return self._parse(content)
            except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
                last_error = str(exc)

        return LLMExtractionResult(
            scores={label: 0.0 for label in TARGETS},
            parse_ok=False,
            error=f"request failed after retries: {last_error}",
        )

    def _parse(self, content: str) -> LLMExtractionResult:
        import json
        import re

        match = re.search(r"\{.*\}", content, re.DOTALL)
        raw = match.group(0) if match else content
        try:
            parsed = json.loads(raw)
        except ValueError:
            return LLMExtractionResult(
                scores={label: 0.0 for label in TARGETS},
                raw_response=content,
                parse_ok=False,
                error="no valid JSON object found in response",
            )

        scores = {label: _coerce_score(parsed.get(label)) for label in TARGETS}
        return LLMExtractionResult(scores=scores, raw_response=content, parse_ok=True)
