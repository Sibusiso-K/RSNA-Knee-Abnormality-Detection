"""Medial vs lateral resolution.

Six of the twelve labels are laterality-specific (medial/lateral meniscus,
medial/lateral OA), so a mention that cannot be sided is a mention we cannot use.
Reports place the side on either end of the structure — "medial meniscus tear"
but also "tear of the posterior horn of the medial meniscus" — so we search both
directions and take the nearest.
"""

from __future__ import annotations

import re

from src.extract import patterns as P

_MEDIAL = re.compile("(?:" + "|".join(P.LATERALITY_MEDIAL) + ")", re.UNICODE)
_LATERAL = re.compile("(?:" + "|".join(P.LATERALITY_LATERAL) + ")", re.UNICODE)
_BOTH = re.compile("(?:" + "|".join(P.LATERALITY_BOTH) + ")", re.UNICODE)
_PATELLOFEMORAL = re.compile(
    "(?:" + "|".join(P.LATERALITY_PATELLOFEMORAL) + ")", re.UNICODE
)


def _nearest(pattern: re.Pattern[str], text: str, start: int, end: int) -> int | None:
    """Distance in characters from [start, end) to the closest match, if any."""
    best: int | None = None
    for match in pattern.finditer(text):
        if match.end() <= start:
            distance = start - match.end()
        elif match.start() >= end:
            distance = match.start() - end
        else:
            distance = 0  # overlapping, e.g. "medial meniscus" as one span
        if best is None or distance < best:
            best = distance
    return best


def resolve(
    sentence: str,
    start: int,
    end: int,
    window: int = 60,
    *,
    allow_patellofemoral: bool = False,
) -> str | None:
    """Side of the mention at [start, end).

    Returns "medial", "lateral", "both", "patellofemoral" (only when
    `allow_patellofemoral`), or None when the sentence does not say.

    "Both" wins outright when present, since "medial and lateral compartments"
    would otherwise resolve to whichever side happens to sit closer.
    """
    scope_start = max(0, start - window)
    scope_end = min(len(sentence), end + window)
    scope = sentence[scope_start:scope_end]
    rel_start, rel_end = start - scope_start, end - scope_start

    if _BOTH.search(scope):
        return "both"

    if allow_patellofemoral:
        pf = _nearest(_PATELLOFEMORAL, scope, rel_start, rel_end)
        if pf is not None and pf <= window:
            return "patellofemoral"

    medial = _nearest(_MEDIAL, scope, rel_start, rel_end)
    lateral = _nearest(_LATERAL, scope, rel_start, rel_end)

    if medial is None and lateral is None:
        return None
    if lateral is None:
        return "medial"
    if medial is None:
        return "lateral"
    if medial == lateral:
        # Equidistant is genuinely ambiguous; better to drop than to guess.
        return None
    return "medial" if medial < lateral else "lateral"
