"""Negation and uncertainty detection.

A NegEx-style approach: find the clause a mention sits in, then look for a
negation trigger before it (or a "is intact"-style trigger after it), bounded by
both the clause and a character window.

Why clauses matter: "No joint effusion, but a tear of the medial meniscus" has a
negation trigger before the tear, yet the tear is real. The word "but" closes the
negation's scope. Getting this wrong flips labels to exactly the opposite of the
truth, which is worse than missing them.
"""

from __future__ import annotations

import re

from src.extract import patterns as P


def _compile_any(options: list[str]) -> re.Pattern[str]:
    return re.compile("(?:" + "|".join(options) + ")", re.UNICODE)


_NEG_PRE = _compile_any(P.NEGATION_PRE)
_NEG_POST = _compile_any(P.NEGATION_POST)
_TERMINATION = _compile_any(P.TERMINATION)
_UNCERTAIN = _compile_any(P.UNCERTAINTY)


def clause_bounds(sentence: str, start: int, end: int) -> tuple[int, int]:
    """The span of the clause containing [start, end) within `sentence`.

    Bounded by termination terms ("but", "however", ";") on either side.
    """
    left = 0
    for match in _TERMINATION.finditer(sentence, 0, start):
        left = match.end()

    right_match = _TERMINATION.search(sentence, end)
    right = right_match.start() if right_match else len(sentence)

    return left, right


def is_negated(sentence: str, start: int, end: int, window: int = 60) -> bool:
    """Whether the mention at [start, end) is denied rather than asserted."""
    left, right = clause_bounds(sentence, start, end)

    pre = sentence[max(left, start - window) : start]
    if _NEG_PRE.search(pre):
        return True

    post = sentence[end : min(right, end + window)]
    return bool(_NEG_POST.search(post))


def is_uncertain(sentence: str, start: int, end: int, window: int = 80) -> bool:
    """Whether the mention is hedged ("possible", "cannot exclude", "?")."""
    left, right = clause_bounds(sentence, start, end)
    scope = sentence[max(left, start - window) : min(right, end + window)]
    return bool(_UNCERTAIN.search(scope))
