"""The twelve competition labels, and the concepts we extract them from.

Submission column order is fixed by the competition — see docs/01-competition.md.
"""

from __future__ import annotations

TARGETS: list[str] = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

ID_COLUMN = "StudyInstanceUID"

# We extract ten *concepts* from the report text, not twelve labels. Two concepts
# ("meniscus", "tf_oa") are laterality-split into a medial and a lateral label,
# because reports describe them as one structure qualified by side.
#
#   concept -> label, or (medial_label, lateral_label) when laterality applies
CONCEPTS: dict[str, str | tuple[str, str]] = {
    "acl": "ACL",
    "mcl": "MCL",
    "meniscus": ("Medial Meniscus", "Lateral Meniscus"),
    "tf_oa": ("Medial OA", "Lateral OA"),
    "pf_oa": "PF OA",
    "effusion": "Effusion",
    "synovitis": "Synovitis",
    "bakers": "Baker's",
    "contusion": "Contusion",
    "fracture": "Fracture",
}

#: Concepts whose label depends on resolving medial vs lateral.
LATERALIZED = {name for name, target in CONCEPTS.items() if isinstance(target, tuple)}


def labels_for(concept: str, laterality: str | None) -> list[str]:
    """Map an extracted concept to the label column(s) it asserts.

    `laterality` is "medial", "lateral", "both", or None. An unresolved
    laterality on a lateralized concept yields no labels — the caller decides
    how to handle it (see ExtractorConfig.unresolved_laterality).
    """
    target = CONCEPTS[concept]
    if isinstance(target, str):
        return [target]

    medial, lateral = target
    if laterality == "medial":
        return [medial]
    if laterality == "lateral":
        return [lateral]
    if laterality == "both":
        return [medial, lateral]
    return []
