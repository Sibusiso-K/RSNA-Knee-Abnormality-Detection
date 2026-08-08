"""Section structure of a radiology report.

Many reports in this dataset are templated by anatomy rather than written as
prose. Measured header counts across all 4,407 training reports:

    795 findings   687 conclusion   680 impresion   663 impression
    465 medial meniscus     454 lateral meniscus    406 indication
    380 medial compartment  380 lateral compartment 363 patellofemoral compartment
    354 cruciate ligaments  345 osseous structures

That structure carries information the sentences below it omit. Under a
`Medial Meniscus:` header, "Tear of the posterior horn." names neither the
structure nor the side — the header already did. A sentence-local extractor
throws all of it away.

Two jobs here:

1. **Propagate context downward.** A header supplies a concept and/or a
   laterality to every sentence beneath it, until the next header.
2. **Exclude the sections that lie.** `Indication:` and
   `Diagnostische vraagstelling:` hold the *referral question* — "?meniscal
   tear" there is the reason for the scan, not a finding. Extracting from them
   manufactures false positives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.extract.text import normalize

# A header can start a line, but just as often several sections run together
# on one line separated by periods rather than newlines — e.g.
# "Técnica: ... Resultados: Rotura del LCA. ... Impresión: ...". Anchoring on
# `^` alone found only the first header in reports shaped that way and
# silently swallowed everything after it into that section. So a header may
# also follow a sentence-ending period, semicolon, or bullet, not only start
# of line — AND a header may immediately follow another header's own colon,
# as in "FINDINGS: Medial Meniscus: Tear of the posterior horn." (two nested
# headers, zero sentence content between them), so ':' belongs in the
# boundary set too.
_HEADER = re.compile(
    r"(?:^|(?<=[.\n;:])|(?<=[.\n;:]\s))[\s*\-–—]*"
    r"([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9 /()'-]{2,40}?)\s*:",
    re.UNICODE | re.MULTILINE,
)


@dataclass(frozen=True)
class SectionContext:
    """What a header tells us about the sentences underneath it."""

    #: Referral questions, technique, history — never contain findings.
    excluded: bool = False
    #: Concept asserted by a bare pathology word in this section. Under
    #: "Medial Meniscus:", "Tear." alone means a medial meniscus tear.
    concept: str | None = None
    #: Side supplied by the header, used when the sentence does not say.
    laterality: str | None = None
    #: Section is about bone, so "oedema" needs no explicit bone qualifier.
    bone_context: bool = False
    #: The radiologist's summary — weighted above findings.
    impression: bool = False


NEUTRAL = SectionContext()

#: Sections that describe why the scan was done or how, not what was seen.
EXCLUDED_HEADERS = {
    "indication", "indications", "clinical indication", "clinical information",
    "clinical history", "history", "clinical details", "reason for exam",
    "reason for examination", "question", "clinical question",
    "technique", "technic", "protocol", "scanprotocol", "scan protocol",
    "comparison", "comparisons", "prior studies", "exam type",
    "exam date and time", "examination", "contrast", "sequences",
    # Spanish / Portuguese
    "tecnica", "indicacion", "indicacao", "antecedentes", "historia clinica",
    "datos clinicos", "comparacion", "protocolo", "informacion clinica",
    # French
    "indication clinique", "renseignements cliniques", "protocole",
    "comparaison", "technique d'examen",
    # German
    "technik", "klinik", "fragestellung", "anamnese", "vergleich",
    "klinische angaben", "untersuchungstechnik",
    # Dutch
    "klinische inlichtingen", "diagnostische vraagstellling",
    "diagnostische vraagstelling", "vraagstelling", "techniek", "protocol",
    "klinische gegevens",
    # Turkish
    "tetkik protokolu", "klinik bilgi", "endikasyon", "teknik", "protokol",
    "klinik", "karsilastirma",
    # Italian
    "tecnica di esame", "indicazione", "confronto", "anamnesi",
}

#: Headers naming the radiologist's summary.
IMPRESSION_HEADERS = {
    "impression", "impressions", "impresion", "impresiones",
    "conclusion", "conclusions", "conclusione", "conclusioni", "conclusao",
    "opinion", "assessment", "summary", "diagnosis", "diagnostico",
    "beurteilung", "zusammenfassung", "besluit", "conclusie",
    "sonuc", "yorum", "impressao",
}

#: Anatomy-named headers, longest key first when matching as a substring.
ANATOMICAL_HEADERS: dict[str, SectionContext] = {
    "medial meniscus": SectionContext(concept="meniscus", laterality="medial"),
    "lateral meniscus": SectionContext(concept="meniscus", laterality="lateral"),
    "menisco medial": SectionContext(concept="meniscus", laterality="medial"),
    "menisco lateral": SectionContext(concept="meniscus", laterality="lateral"),
    "menisco interno": SectionContext(concept="meniscus", laterality="medial"),
    "menisco externo": SectionContext(concept="meniscus", laterality="lateral"),
    "menisque interne": SectionContext(concept="meniscus", laterality="medial"),
    "menisque externe": SectionContext(concept="meniscus", laterality="lateral"),
    "innenmeniskus": SectionContext(concept="meniscus", laterality="medial"),
    "aussenmeniskus": SectionContext(concept="meniscus", laterality="lateral"),
    "mediale meniscus": SectionContext(concept="meniscus", laterality="medial"),
    "laterale meniscus": SectionContext(concept="meniscus", laterality="lateral"),
    "medial meniskus": SectionContext(concept="meniscus", laterality="medial"),
    "lateral meniskus": SectionContext(concept="meniscus", laterality="lateral"),

    "medial compartment": SectionContext(concept="tf_oa", laterality="medial"),
    "lateral compartment": SectionContext(concept="tf_oa", laterality="lateral"),
    "medial femorotibial": SectionContext(concept="tf_oa", laterality="medial"),
    "lateral femorotibial": SectionContext(concept="tf_oa", laterality="lateral"),
    "compartimento medial": SectionContext(concept="tf_oa", laterality="medial"),
    "compartimento lateral": SectionContext(concept="tf_oa", laterality="lateral"),
    "compartiment interne": SectionContext(concept="tf_oa", laterality="medial"),
    "compartiment externe": SectionContext(concept="tf_oa", laterality="lateral"),

    "patellofemoral compartment": SectionContext(concept="pf_oa"),
    "patellofemoral joint": SectionContext(concept="pf_oa"),
    "patellofemoral cartilage": SectionContext(concept="pf_oa"),
    "patellofemoral tracking": SectionContext(concept="pf_oa"),
    "femoropatellar": SectionContext(concept="pf_oa"),
    "compartimento patelofemoral": SectionContext(concept="pf_oa"),
    "patellofemoralgelenk": SectionContext(concept="pf_oa"),

    # Bone sections: "marrow oedema" here needs no explicit bone word.
    "osseous structures": SectionContext(bone_context=True),
    "osseous": SectionContext(bone_context=True),
    "bone marrow": SectionContext(bone_context=True),
    "bones": SectionContext(bone_context=True),
    "estructuras oseas": SectionContext(bone_context=True),
    "knochen": SectionContext(bone_context=True),
    "kemik yapilar": SectionContext(bone_context=True),
}

# NOTE: deliberately NOT mapped — "cruciate ligaments" and "collateral
# ligaments" are ambiguous between the ACL/PCL and MCL/LCL of each pair, and
# only one of each is a competition label. A bare "torn" under those headers
# cannot be assigned safely, so we let the sentence name the ligament itself.

_ANATOMICAL_KEYS = sorted(ANATOMICAL_HEADERS, key=len, reverse=True)


@dataclass(frozen=True)
class Section:
    header: str
    context: SectionContext
    start: int
    end: int


def classify(header: str) -> SectionContext:
    """Map a normalized header to the context it supplies."""
    name = header.strip().strip("*-–— ").strip()

    if name in EXCLUDED_HEADERS:
        return SectionContext(excluded=True)
    if name in IMPRESSION_HEADERS:
        return SectionContext(impression=True)
    if name in ANATOMICAL_HEADERS:
        return ANATOMICAL_HEADERS[name]

    # "medial compartment cartilage" should inherit "medial compartment".
    for key in _ANATOMICAL_KEYS:
        if key in name:
            return ANATOMICAL_HEADERS[key]
    for key in EXCLUDED_HEADERS:
        if name.startswith(key):
            return SectionContext(excluded=True)

    return NEUTRAL


def parse(text: str) -> list[Section]:
    """Split normalized report text into sections at its headers."""
    matches = list(_HEADER.finditer(text))
    sections: list[Section] = []

    for i, match in enumerate(matches):
        header = normalize(match.group(1)).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(Section(header, classify(header), match.end(), end))

    return sections


def context_at(sections: list[Section], position: int) -> SectionContext:
    """Context in force at `position`; NEUTRAL before the first header."""
    for section in sections:
        if section.start <= position < section.end:
            return section.context
    return NEUTRAL
