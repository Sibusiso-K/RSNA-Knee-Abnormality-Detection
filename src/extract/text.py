"""Text preparation: normalization, sentence splitting, section detection.

Deliberately language-agnostic. Reports arrive in roughly nine languages and we
do not attempt language identification — see patterns.py for why the pattern
sets are unioned rather than dispatched on language.
"""

from __future__ import annotations

import re
import unicodedata

from src.extract.types import Sentence

# Sentence boundaries: terminal punctuation, semicolons, newlines, and bullets.
# Radiology reports are heavily line-broken and often use lists rather than
# prose, so a newline is at least as reliable a boundary as a full stop.
_BOUNDARY = re.compile(r"(?<=[.;:!?])\s+|\n+|(?:^|\s)[-–—*•]\s+", re.UNICODE)

# "IMPRESSION:", "CONCLUSIONE:", "BEURTEILUNG:" ... a short all-caps-ish run
# followed by a colon at the start of a line.
_SECTION_HEADER = re.compile(
    r"^\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ /]{2,30}?)\s*:", re.UNICODE | re.MULTILINE
)

#: Section names that mark the radiologist's summary. The impression is the
#: highest-signal, lowest-noise part of a report — findings sections describe
#: everything looked at, impressions describe what matters.
IMPRESSION_HEADERS = {
    "impression",
    "impressions",
    "conclusion",
    "conclusions",
    "conclusione",
    "conclusioni",
    "conclusao",
    "conclusion clinique",
    "opinion",
    "assessment",
    "summary",
    "diagnostico",
    "diagnosis",
    "beurteilung",
    "zusammenfassung",
    "besluit",
    "sonuc",
    "impresion",
}

FINDINGS_HEADERS = {
    "findings",
    "finding",
    "report",
    "hallazgos",
    "achados",
    "resultats",
    "resultat",
    "befund",
    "befunde",
    "bevindingen",
    "reperti",
    "descripcion",
    "descricao",
    "bulgular",
}


def strip_accents(text: str) -> str:
    """Fold accents so 'derrame' and 'dérrame' match one pattern."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """Lowercase and fold accents.

    NOT length-preserving — `.lower()` and NFKD decomposition can both change
    character counts (Turkish dotted capital I becomes two code points, ligatures
    expand). So offsets computed on normalized text are only valid *against
    normalized text*. The extractor therefore works in normalized space
    throughout and never maps spans back onto the raw report.
    """
    folded = strip_accents(text.lower())
    # Unify the several dash and quote characters reports use interchangeably.
    folded = folded.translate(str.maketrans({"–": "-", "—": "-", "’": "'", "`": "'"}))
    return folded


def split_sentences(text: str) -> list[Sentence]:
    """Split into sentences, keeping each one's offset into `text`."""
    sentences: list[Sentence] = []
    cursor = 0
    for match in _BOUNDARY.finditer(text):
        chunk = text[cursor : match.start()]
        if chunk.strip():
            offset = len(chunk) - len(chunk.lstrip())
            sentences.append(
                Sentence(chunk.strip(), cursor + offset, len(sentences))
            )
        cursor = match.end()

    tail = text[cursor:]
    if tail.strip():
        offset = len(tail) - len(tail.lstrip())
        sentences.append(Sentence(tail.strip(), cursor + offset, len(sentences)))

    return sentences


def section_of(text: str, position: int) -> str | None:
    """Which section heading is in force at `position`.

    Returns a normalized heading, or None before the first heading. Used to
    weight impression-section mentions above findings-section ones.
    """
    current: str | None = None
    for match in _SECTION_HEADER.finditer(text):
        if match.start() > position:
            break
        current = normalize(match.group(1)).strip()
    return current


def section_kind(section: str | None) -> str:
    """Bucket a raw section heading into 'impression' | 'findings' | 'other'."""
    if section is None:
        return "other"
    if section in IMPRESSION_HEADERS:
        return "impression"
    if section in FINDINGS_HEADERS:
        return "findings"
    return "other"
