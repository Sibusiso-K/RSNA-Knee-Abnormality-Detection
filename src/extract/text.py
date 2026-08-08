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

# NOTE: section-header parsing (headings, impression detection, excluded
# sections like "Indication:") lives in sections.py now — it grew into its own
# module once the anatomy-templated reports turned up. See docs/00-state.md.


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
    # Also fold U+00B5 MICRO SIGN to U+03BC GREEK SMALL LETTER MU: the Greek
    # reports in this dataset (~7% of the corpus, discovered session 4) use
    # the micro sign in place of mu throughout — almost certainly a font/OCR
    # substitution bug upstream, since both render near-identically. Every
    # Greek word containing mu is silently wrong without this fold, and Greek
    # patterns in patterns.py are written assuming it has already happened.
    folded = folded.translate(
        str.maketrans({"–": "-", "—": "-", "’": "'", "`": "'", "µ": "μ"})
    )
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


