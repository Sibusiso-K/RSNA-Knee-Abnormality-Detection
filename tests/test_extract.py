"""Unit tests for the report -> label extractor.

All report text here is INVENTED, not competition data. The point is to pin the
behaviours that are easy to break while editing patterns — especially negation,
which flips a label to exactly the wrong value when it goes wrong.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extract import ExtractorConfig, RuleExtractor  # noqa: E402
from src.extract.context import is_negated, is_uncertain  # noqa: E402
from src.extract.laterality import resolve  # noqa: E402
from src.extract.text import normalize, split_sentences  # noqa: E402
from src.labels import TARGETS  # noqa: E402


@pytest.fixture
def extractor() -> RuleExtractor:
    return RuleExtractor()


def scores(extractor: RuleExtractor, report: str) -> dict[str, float]:
    return extractor.extract(report).scores


# -- text ----------------------------------------------------------------

def test_normalize_folds_accents_and_case():
    assert normalize("Déchirure MÉNISQUE") == "dechirure menisque"


def test_split_sentences_handles_newlines_and_stops():
    parts = split_sentences("ACL tear.\nNo effusion; small cyst")
    assert [p.text for p in parts] == ["ACL tear.", "No effusion;", "small cyst"]


def test_normalize_folds_turkish_dotless_i():
    """U+0131 (Turkish dotless ı) is a distinct base letter, not an accented
    "i" — NFKD accent-stripping does not touch it. Every Turkish pattern is
    written with ASCII "i" and silently failed against real words like
    "kırık" (fracture) and "sıvı" (fluid) until this fold (session 5)."""
    assert normalize("KIRIK") == "kirik"
    assert "ı" not in normalize("sıvı artışı")


def test_normalize_folds_greek_micro_sign():
    """U+00B5 MICRO SIGN stands in for Greek mu (U+03BC) throughout this
    corpus — an upstream font/OCR artifact, not a legitimate variant."""
    assert normalize("µηνίσκου") == normalize("μηνίσκου")


# -- negation ------------------------------------------------------------

def test_negation_before_finding():
    sentence = "no evidence of acl tear"
    assert is_negated(sentence, sentence.index("acl"), sentence.index("acl") + 3)


def test_negation_does_not_cross_but():
    sentence = "no joint effusion, but a tear of the medial meniscus"
    start = sentence.index("meniscus")
    assert not is_negated(sentence, start, start + 8)


def test_intact_negates_after_the_finding():
    sentence = "the anterior cruciate ligament is intact"
    assert is_negated(sentence, 4, 33)


def test_french_elision_negates():
    """"pas d'epanchement" — elision before a vowel, not "pas de"."""
    sentence = "pas d'epanchement articulaire"
    start = sentence.index("epanchement")
    assert is_negated(sentence, start, start + 11)


def test_uncertainty_detected():
    sentence = "possible tear of the lateral meniscus"
    start = sentence.index("meniscus")
    assert is_uncertain(sentence, start, start + 8)


# -- laterality ----------------------------------------------------------

@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("medial meniscus tear", "medial"),
        ("tear of the posterior horn of the lateral meniscus", "lateral"),
        ("medial and lateral meniscal tears", "both"),
        ("meniscal tear", None),
    ],
)
def test_laterality_resolution(sentence, expected):
    start = sentence.index("menisc")
    assert resolve(sentence, start, start + 6) == expected


# -- end to end ----------------------------------------------------------

def test_positive_finding(extractor):
    result = scores(extractor, "Complete tear of the anterior cruciate ligament.")
    assert result["ACL"] == pytest.approx(1.0, abs=0.1)


def test_negated_finding_scores_zero(extractor):
    result = scores(extractor, "No evidence of anterior cruciate ligament tear.")
    assert result["ACL"] == 0.0


def test_negation_scope_break_keeps_the_real_finding(extractor):
    report = "No joint effusion, but a tear of the posterior horn of the medial meniscus."
    result = scores(extractor, report)
    assert result["Effusion"] == 0.0
    assert result["Medial Meniscus"] > 0.5
    assert result["Lateral Meniscus"] == 0.0


def test_uncertain_scores_between_absent_and_present(extractor):
    result = scores(extractor, "Possible tear of the lateral meniscus.")
    assert 0.0 < result["Lateral Meniscus"] < 1.0


def test_both_sides(extractor):
    report = "Degenerative changes of the medial and lateral tibiofemoral compartments."
    result = scores(extractor, report)
    assert result["Medial OA"] > 0.5
    assert result["Lateral OA"] > 0.5


def test_patellofemoral_oa_routes_away_from_tibiofemoral(extractor):
    result = scores(extractor, "Patellofemoral osteoarthritis with cartilage loss.")
    assert result["PF OA"] > 0.5
    assert result["Medial OA"] == 0.0
    assert result["Lateral OA"] == 0.0


def test_contusion_requires_the_explicit_word(extractor):
    """Session 4: bare "bone marrow edema" was tried as sufficient on its own
    (edema + a nearby bone word), but measured against the gold studies that
    fired on subchondral/reactive edema from OA and adjacent lesions just as
    often as on real contusions — 17 false positives vs 12 true positives.
    Gold labels reserve Contusion for the explicit bruise/contusion wording,
    so that's what the pattern now requires."""
    bare_edema = scores(extractor, "Bone marrow oedema in the lateral femoral condyle.")
    named = scores(extractor, "Bone contusion of the lateral femoral condyle.")
    soft = scores(extractor, "Soft tissue oedema in the popliteal fat.")
    assert bare_edema["Contusion"] == 0.0
    assert named["Contusion"] > 0.5
    assert soft["Contusion"] == 0.0


def test_unresolved_laterality_is_dropped_by_default(extractor):
    result = scores(extractor, "There is a meniscal tear.")
    assert result["Medial Meniscus"] == 0.0
    assert result["Lateral Meniscus"] == 0.0


def test_unresolved_laterality_can_credit_both_sides():
    both = RuleExtractor(ExtractorConfig(unresolved_laterality="both"))
    result = both.extract("There is a meniscal tear.").scores
    assert result["Medial Meniscus"] > 0.0
    assert result["Lateral Meniscus"] > 0.0


@pytest.mark.parametrize(
    ("report", "label"),
    [
        ("Rotura del ligamento cruzado anterior.", "ACL"),
        ("Derrame articular moderado.", "Effusion"),
        ("Dechirure du menisque interne.", "Medial Meniscus"),
        ("Rotura completa del menisco externo.", "Lateral Meniscus"),
        ("Ruptur des vorderen Kreuzbandes.", "ACL"),
        ("Kein Gelenkerguss.", None),
        ("Bakerzyste.", "Baker's"),
    ],
)
def test_multilingual(extractor, report, label):
    result = scores(extractor, report)
    if label is None:
        assert result["Effusion"] == 0.0
    else:
        assert result[label] > 0.5


def test_empty_report_is_all_absent(extractor):
    result = scores(extractor, "")
    assert set(result) == set(TARGETS)
    assert all(value == 0.0 for value in result.values())


def test_every_target_is_always_present(extractor):
    result = scores(extractor, "Normal knee MRI.")
    assert list(result) == TARGETS


# -- section-header awareness ---------------------------------------------
# These pin the two real bugs found once actual competition reports were
# available (session 3): the header regex only matched at a literal line
# start, and OA vocabulary was missing "fissuring"/"spurring"/"chondrosis".

def test_bare_finding_under_anatomical_header(extractor):
    """"Medial Meniscus: Tear..." names no structure in the sentence itself —
    the header supplies it. This was the original motivating bug: the
    sentence-only extractor produced no mention at all here."""
    report = "FINDINGS: Medial Meniscus: Tear of the posterior horn."
    result = scores(extractor, report)
    assert result["Medial Meniscus"] > 0.5
    assert result["Lateral Meniscus"] == 0.0


def test_headers_on_a_single_line_are_still_split(extractor):
    """Many real reports run every section together on one line, separated
    by periods rather than newlines: "Technique: ... Findings: ... Derrame."
    A header regex anchored only to line-start finds just the first header
    and swallows everything after it into that (often excluded) section —
    which silently deleted real findings until fixed."""
    report = "Technique: MRI of the knee. Findings: Joint effusion present."
    result = scores(extractor, report)
    assert result["Effusion"] > 0.5


def test_indication_section_is_excluded(extractor):
    """A referral question ("?meniscal tear") is not a finding."""
    report = (
        "Indication: Possible meniscal tear, please assess. "
        "Findings: Menisci are intact."
    )
    result = scores(extractor, report)
    assert result["Medial Meniscus"] == 0.0
    assert result["Lateral Meniscus"] == 0.0


def test_compartment_header_supplies_laterality_for_bare_oa_terms(extractor):
    report = "Medial Compartment: Cartilage fissuring. Marginal spurring."
    result = scores(extractor, report)
    assert result["Medial OA"] > 0.5
    assert result["Lateral OA"] == 0.0


def test_turkish_dotless_i_in_fracture_word(extractor):
    """Real false negative from session 5: "subkondral kırığı" (subchondral
    fracture) used dotless ı throughout; the word "kirik" was already in the
    vocabulary but written with ASCII i, so it never matched."""
    result = scores(extractor, "Subkondral kırığı izlenmektedir.")
    assert result["Fracture"] > 0.5


def test_bulgarian_meniscus_tear(extractor):
    result = scores(extractor, "Разкъсване на медиалния менискус.")
    assert result["Medial Meniscus"] > 0.5
    assert result["Lateral Meniscus"] == 0.0


def test_bulgarian_negation(extractor):
    result = scores(extractor, "Без данни за фрактура. Ставите са нормални.")
    assert result["Fracture"] == 0.0


def test_greek_effusion_and_negation(extractor):
    result = scores(
        extractor,
        "Παρατηρείται μέτρια ποσότητα υγρού ενδαρθρικά. Δεν παρατηρείται κάταγμα.",
    )
    assert result["Effusion"] > 0.5
    assert result["Fracture"] == 0.0


def test_chondrosis_and_spurring_are_recognised(extractor):
    result = scores(extractor, "Tricompartmental chondrosis with marginal spurring.")
    # Not lateralized -> dropped by default config, but must be *seen*
    # (i.e. land in `unresolved`) rather than missed by the vocabulary.
    extraction = extractor.extract(
        "Tricompartmental chondrosis with marginal spurring."
    )
    assert any(m.concept == "tf_oa" for m in extraction.unresolved)
