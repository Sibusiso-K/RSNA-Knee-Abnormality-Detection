"""Data structures shared across the extractor."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Sentence:
    """One sentence of a report, with its offset into the original text."""

    text: str
    start: int
    index: int

    @property
    def end(self) -> int:
        return self.start + len(self.text)


@dataclass(frozen=True)
class Mention:
    """A single place in the report where a concept was asserted or denied.

    One report can produce several mentions of the same concept — possibly
    contradicting each other, e.g. a hedged "possible tear" in Findings followed
    by a confident one in the Impression. Aggregation happens in rules.py.
    """

    concept: str
    text: str
    start: int
    end: int
    sentence_index: int
    section: str | None = None
    negated: bool = False
    uncertain: bool = False
    laterality: str | None = None  # "medial" | "lateral" | "both" | None
    #: True when this mention came from a header-anatomy fallback (e.g. "Tear."
    #: under a "Medial Meniscus:" header) rather than naming the structure
    #: itself. Kept for auditing — see sections.py.
    from_section_context: bool = False
    #: True when the section is the radiologist's summary (Impression/
    #: Conclusion). Scored higher than a passing remark in Findings.
    in_impression: bool = False

    @property
    def polarity(self) -> str:
        if self.negated:
            return "negated"
        return "uncertain" if self.uncertain else "affirmed"


@dataclass
class StudyExtraction:
    """Everything the extractor concluded about one report."""

    study_uid: str
    #: label -> confidence in [0, 1]. Soft, because the metric is rank-based.
    scores: dict[str, float] = field(default_factory=dict)
    mentions: list[Mention] = field(default_factory=list)
    #: Concepts seen but dropped for unresolved laterality — a quality signal.
    unresolved: list[Mention] = field(default_factory=list)
    language_hint: str | None = None

    def evidence_for(self, label: str) -> list[Mention]:
        """Mentions that contributed to `label`, for eyeballing disagreements."""
        from src.labels import labels_for

        return [
            m
            for m in self.mentions
            if label in labels_for(m.concept, m.laterality)
        ]
