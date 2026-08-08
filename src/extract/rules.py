"""The rule-based extractor: report text in, twelve label confidences out.

Pipeline per report:

    normalize -> parse sections -> split sentences -> for each non-excluded
    sentence: match concepts (direct, and header-anatomy fallback) -> decide
    negation / uncertainty / laterality, inheriting from the section header
    where the sentence itself is silent -> aggregate mentions into scores

Scores are **soft** (0..1), not binary. The competition metric is rank-based
AUC, so "possible meniscal tear" belonging strictly between absent and present
is information worth keeping rather than thresholding away. See docs/04-method.md.

This is the Phase 1 baseline. It is meant to be auditable and fast, and to give
the LLM-based extractor something to be measured against — not to be the final
word. See docs/05-plan.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.extract import context, laterality as lat
from src.extract import sections as S
from src.extract.patterns import CONCEPT_PATTERNS, ConceptPattern
from src.extract.text import normalize, split_sentences
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

    negation_window: int = 60
    uncertainty_window: int = 80
    laterality_window: int = 60

    #: What to do with "meniscal tear" when no side is given anywhere — not
    #: in-sentence, not from the previous sentence, and not from a section
    #: header.
    #: "drop" - ignore it (conservative; the mention lands in `unresolved`)
    #: "both" - credit both sides, scaled by `unresolved_laterality_weight`
    unresolved_laterality: str = "drop"
    unresolved_laterality_weight: float = 0.5

    #: If a sentence gives no side, inherit one from the sentence before it.
    laterality_lookback: bool = True

    #: If still no side, inherit the section header's laterality (e.g. a bare
    #: "Tear of the posterior horn." under a "Medial Meniscus:" header).
    laterality_from_section: bool = True


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
        report_sections = S.parse(text)
        sentences = split_sentences(text)

        for sentence in sentences:
            ctx = S.context_at(report_sections, sentence.start)
            if ctx.excluded:
                # Referral question / technique / history: never a finding.
                continue

            previous = sentences[sentence.index - 1].text if sentence.index else None
            for mention in self._mentions_in(sentence.text, sentence.index, previous, ctx):
                if mention.concept in LATERALIZED and mention.laterality is None:
                    result.unresolved.append(mention)
                    if self.config.unresolved_laterality != "both":
                        continue
                result.mentions.append(mention)

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
        self,
        sentence: str,
        index: int,
        previous: str | None,
        ctx: S.SectionContext,
    ) -> list[Mention]:
        mentions: list[Mention] = []
        matched: set[str] = set()

        for name, compiled in self._compiled.items():
            for start, end in compiled.find(sentence):
                concept = name
                # An OA term under a header that already names the
                # compartment is that compartment's OA, not a coin flip
                # between tibiofemoral and patellofemoral.
                if concept == "tf_oa" and ctx.concept in ("tf_oa", "pf_oa"):
                    concept = ctx.concept
                mentions.append(
                    self._build(concept, sentence, start, end, index, previous, ctx)
                )
                matched.add(concept)

        # Header-anatomy fallback: the structure/compartment was named by the
        # section heading, not the sentence, so match on the qualifier alone.
        # This is the fix for templated reports like "Medial Meniscus: Tear
        # of the posterior horn." — see the module docstring in sections.py.
        if ctx.concept == "meniscus" and "meniscus" not in matched:
            qualifier = self._compiled["meniscus"].qualifier
            if qualifier is not None:
                for match in qualifier.finditer(sentence):
                    mentions.append(
                        self._build(
                            "meniscus", sentence, match.start(), match.end(),
                            index, previous, ctx, from_section=True,
                        )
                    )

        if ctx.concept == "pf_oa" and "pf_oa" not in matched:
            qualifier = self._compiled["pf_oa"].qualifier
            if qualifier is not None:
                for match in qualifier.finditer(sentence):
                    mentions.append(
                        self._build(
                            "pf_oa", sentence, match.start(), match.end(),
                            index, previous, ctx, from_section=True,
                        )
                    )

        # Bone-section fallback: under "Osseous Structures:", "oedema" alone
        # means bone marrow oedema without needing the word "bone" nearby.
        if ctx.bone_context and "contusion" not in matched:
            anchor = self._compiled["contusion"].anchor
            if anchor is not None:
                for match in anchor.finditer(sentence):
                    mentions.append(
                        self._build(
                            "contusion", sentence, match.start(), match.end(),
                            index, previous, ctx, from_section=True,
                        )
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
        ctx: S.SectionContext,
        *,
        from_section: bool = False,
    ) -> Mention:
        cfg = self.config
        side: str | None = None
        lateralizable = concept in LATERALIZED or concept == "tf_oa"

        if lateralizable:
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
            if side is None and cfg.laterality_from_section:
                side = ctx.laterality

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
            from_section_context=from_section,
            in_impression=ctx.impression,
        )

    def _aggregate(self, mentions: list[Mention]) -> dict[str, float]:
        cfg = self.config
        scores = {label: cfg.absent_score for label in TARGETS}

        for mention in mentions:
            targets = labels_for(mention.concept, mention.laterality)
            scale = 1.0
            if not targets and mention.concept in LATERALIZED:
                if cfg.unresolved_laterality != "both":
                    continue
                targets, scale = _both_sides(mention.concept), cfg.unresolved_laterality_weight

            weight = cfg.impression_weight if mention.in_impression else cfg.findings_weight
            for label in targets:
                if mention.negated:
                    value = cfg.negated_score
                elif mention.uncertain:
                    value = cfg.uncertain_score
                else:
                    value = cfg.affirmed_score
                value *= weight * scale
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
