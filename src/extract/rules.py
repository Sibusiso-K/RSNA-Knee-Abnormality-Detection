"""The rule-based extractor: report text in, twelve label confidences out.

Pipeline per report:

    normalize -> split sentences -> match concepts -> for each match decide
    negation / uncertainty / laterality -> aggregate mentions into scores

Scores are **soft** (0..1), not binary. The competition metric is rank-based
AUC, so "possible meniscal tear" belonging strictly between absent and present
is information worth keeping rather than thresholding away. See docs/04-method.md.

This is the Phase 1 baseline. It is meant to be auditable and fast, and to give
the LLM-based extractor something to be measured against — not to be the final
word. See docs/05-plan.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.extract import context, laterality as lat
from src.extract.patterns import CONCEPT_PATTERNS, ConceptPattern
from src.extract.text import normalize, section_kind, section_of, split_sentences
from src.extract.types import Mention, StudyExtraction
from src.labels import LATERALIZED, TARGETS, labels_for


@dataclass
class ExtractorConfig:
    """Tunables. Every default here is a guess until measured against the gold
    studies — `scripts/extract_labels.py --evaluate` is how you check."""

    affirmed_score: float = 1.0
    uncertain_score: float = 0.6
    #: Score when the report explicitly denies the finding. Distinct from
    #: `absent_score` only if you later decide silence and denial differ.
    negated_score: float = 0.0
    #: Score when the report never mentions the concept at all.
    absent_score: float = 0.0

    #: Mentions in the impression carry more weight than passing remarks in
    #: findings. Multiplies the score; affects ranking only.
    impression_weight: float = 1.0
    findings_weight: float = 0.92
    other_weight: float = 0.92

    negation_window: int = 60
    uncertainty_window: int = 80
    laterality_window: int = 60

    #: What to do with "meniscal tear" when no side is given.
    #: "drop"   - ignore it (conservative; the mention lands in `unresolved`)
    #: "both"   - credit both sides, scaled by `unresolved_laterality_weight`
    #: A/B these against the gold studies; "drop" loses real positives, "both"
    #: manufactures false ones. Which is better is an empirical question.
    unresolved_laterality: str = "drop"
    unresolved_laterality_weight: float = 0.5

    #: If a sentence gives no side, inherit one from the sentence before it.
    laterality_lookback: bool = True


class RuleExtractor:
    """Extracts the twelve label confidences from a radiology report."""

    def __init__(self, config: ExtractorConfig | None = None) -> None:
        self.config = config or ExtractorConfig()
        self._compiled = {
            name: _CompiledConcept(pattern)
            for name, pattern in CONCEPT_PATTERNS.items()
        }

    # -- public API ------------------------------------------------------

    def extract(self, report: str, study_uid: str = "") -> StudyExtraction:
        result = StudyExtraction(study_uid=study_uid)
        if not report or not report.strip():
            result.scores = {label: self.config.absent_score for label in TARGETS}
            return result

        text = normalize(report)
        sentences = split_sentences(text)

        for sentence in sentences:
            previous = sentences[sentence.index - 1].text if sentence.index else None
            for mention in self._mentions_in(sentence.text, sentence.index, previous):
                located = _with_section(mention, text, sentence.start)
                if mention.concept in LATERALIZED and mention.laterality is None:
                    result.unresolved.append(located)
                    if self.config.unresolved_laterality != "both":
                        continue
                result.mentions.append(located)

        result.scores = self._aggregate(result.mentions)
        return result

    def extract_frame(self, frame, text_column: str = "Report", id_column: str = "StudyInstanceUID"):
        """Run over a DataFrame of reports, returning a DataFrame of scores.

        Kept import-light so this module does not require pandas to be present
        for the unit tests.
        """
        import pandas as pd

        rows = []
        for _, row in frame.iterrows():
            extraction = self.extract(row.get(text_column) or "", str(row.get(id_column, "")))
            rows.append({id_column: row.get(id_column), **extraction.scores})
        return pd.DataFrame(rows, columns=[id_column, *TARGETS])

    # -- internals -------------------------------------------------------

    def _mentions_in(
        self, sentence: str, index: int, previous: str | None
    ) -> list[Mention]:
        mentions: list[Mention] = []
        for name, compiled in self._compiled.items():
            for start, end in compiled.find(sentence):
                mentions.append(
                    self._build(name, sentence, start, end, index, previous)
                )
        return mentions

    def _build(
        self,
        concept: str,
        sentence: str,
        start: int,
        end: int,
        index: int,
        previous: str | None,
    ) -> Mention:
        cfg = self.config
        side: str | None = None

        if concept in LATERALIZED or concept == "tf_oa":
            side = lat.resolve(
                sentence,
                start,
                end,
                cfg.laterality_window,
                allow_patellofemoral=(concept == "tf_oa"),
            )
            if side is None and cfg.laterality_lookback and previous:
                side = lat.resolve(
                    previous, len(previous), len(previous), cfg.laterality_window
                )

        # An OA term sided to the kneecap is patellofemoral OA, not tibiofemoral.
        if concept == "tf_oa" and side == "patellofemoral":
            concept, side = "pf_oa", None

        return Mention(
            concept=concept,
            text=sentence[start:end],
            start=start,
            end=end,
            sentence_index=index,
            negated=context.is_negated(sentence, start, end, cfg.negation_window),
            uncertain=context.is_uncertain(sentence, start, end, cfg.uncertainty_window),
            laterality=side,
        )

    def _aggregate(self, mentions: list[Mention]) -> dict[str, float]:
        cfg = self.config
        scores = {label: cfg.absent_score for label in TARGETS}
        seen: set[str] = set()

        for mention in mentions:
            targets = labels_for(mention.concept, mention.laterality)
            scale = 1.0
            if not targets and mention.concept in LATERALIZED:
                if cfg.unresolved_laterality != "both":
                    continue
                target = _both_sides(mention.concept)
                targets, scale = target, cfg.unresolved_laterality_weight

            for label in targets:
                seen.add(label)
                if mention.negated:
                    value = cfg.negated_score
                elif mention.uncertain:
                    value = cfg.uncertain_score
                else:
                    value = cfg.affirmed_score
                value *= _section_weight(cfg, mention.section) * scale
                # Positive evidence anywhere outweighs a denial elsewhere: a
                # report saying "possible tear" then "no acute tear" should not
                # rank below one that never mentions the meniscus at all.
                scores[label] = max(scores[label], value)

        return scores


class _CompiledConcept:
    """Compiled matcher for one concept."""

    def __init__(self, pattern: ConceptPattern) -> None:
        self.pattern = pattern
        self.direct = pattern.compiled_direct()
        self.anchor = pattern.compiled_anchor()
        self.qualifier = pattern.compiled_qualifier()

    def find(self, sentence: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []

        if self.direct is not None:
            spans.extend((m.start(), m.end()) for m in self.direct.finditer(sentence))

        if self.anchor is not None and self.qualifier is not None:
            qualifiers = [(m.start(), m.end()) for m in self.qualifier.finditer(sentence)]
            for match in self.anchor.finditer(sentence):
                if _within(match.start(), match.end(), qualifiers, self.pattern.window):
                    spans.append((match.start(), match.end()))

        return _dedupe(spans)


def _within(
    start: int, end: int, others: list[tuple[int, int]], window: int
) -> bool:
    for other_start, other_end in others:
        if other_end <= start:
            gap = start - other_end
        elif other_start >= end:
            gap = other_start - end
        else:
            gap = 0
        if gap <= window:
            return True
    return False


def _dedupe(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop spans fully contained in another, so one finding counts once."""
    ordered = sorted(set(spans), key=lambda s: (s[0], -s[1]))
    kept: list[tuple[int, int]] = []
    for start, end in ordered:
        if any(k_start <= start and end <= k_end for k_start, k_end in kept):
            continue
        kept.append((start, end))
    return kept


def _both_sides(concept: str) -> list[str]:
    from src.labels import CONCEPTS

    target = CONCEPTS[concept]
    return list(target) if isinstance(target, tuple) else [target]


def _section_weight(config: ExtractorConfig, section: str | None) -> float:
    kind = section_kind(section)
    if kind == "impression":
        return config.impression_weight
    if kind == "findings":
        return config.findings_weight
    return config.other_weight


def _with_section(mention: Mention, full_text: str, sentence_start: int) -> Mention:
    from dataclasses import replace

    return replace(
        mention, section=section_of(full_text, sentence_start + mention.start)
    )
