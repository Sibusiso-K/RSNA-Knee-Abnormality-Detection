"""Concept patterns for the ten extractable concepts.

STATUS: written from domain knowledge before seeing a single real report. Treat
every pattern here as a hypothesis. The first job once `train.csv` lands is to
run `scripts/extract_labels.py --audit` and fix what these miss. Recall failures
are invisible unless you go looking for them.

Two conventions:

1. **Patterns match NORMALIZED text** (lowercased, accents folded — see
   text.normalize). So write "epanchement", never "épanchement".

2. **All languages are unioned, not dispatched.** We do not identify the report
   language first. Radiology vocabulary collides rarely across languages, and a
   union is far more robust than a language detector that is wrong 5% of the
   time. Cost: a few false friends, which the audit should surface.

Concept shape:
  - `direct`   : matching this alone asserts the finding (e.g. "effusion")
  - `anchor` + `qualifier`: BOTH must appear in the same sentence within
    `window` characters (e.g. "meniscus" + "tear"). Anchors name a structure;
    qualifiers name the pathology. A meniscus is not a finding; a torn one is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def _alt(*options: str) -> str:
    """Join alternatives into one non-capturing group."""
    return "(?:" + "|".join(options) + ")"


# --------------------------------------------------------------------------
# Shared vocabulary
# --------------------------------------------------------------------------

#: Structural failure of a ligament or meniscus, across languages.
INJURY = _alt(
    # `ruptur\w*` covers rupture/ruptured/ruptura/rupturas and bare German
    # "Ruptur" in one pattern — inflection varies more than it looks.
    r"tears?", r"torn", r"tearing", r"ruptur\w*",
    r"roturas?", r"desgarros?", r"dechirures?", r"rompu[es]?", r"risse?",
    r"rissbildung", r"lacerations?", r"laceracao", r"lacerazione",
    r"discontinuity", r"disrupt(?:ion|ed)", r"avulsions?", r"fissur\w*",
    # Turkish "yırtık" (tear) softens its final k to ğ before a vowel suffix
    # ("yırtığı" = "its tear") — normal Turkish consonant mutation, not a
    # typo. After the ı->i fold both spellings differ only in that last
    # consonant, hence [kg] rather than a fixed "k".
    r"lesion\w*", r"lezyon\w*", r"scheur\w*", r"yirti[kg]\w*", r"rottura",
    r"strappo", r"riss\w*", r"insufficien\w*", r"non-?visualiz\w*",
    # Greek: ρήξη/ρήξεις (rupture/tear) is the standard term; "εκφυλιστική
    # ρήξη" (degenerative tear) is common phrasing. διάσπαση (disruption).
    r"ρηξ\w*", r"διασπασ\w*",
    # Bulgarian (Cyrillic, ~5% of the corpus, found session 5): руптура
    # (rupture, cognate), разкъсване (tear).
    r"руптур\w*", r"разкъсв\w*",
)

#: Lower-grade ligament injury.
SPRAIN = _alt(
    r"sprains?", r"esguince\w*", r"entorse\w*", r"zerrung", r"distorsione",
    r"θλασ\w*",  # Greek: θλάση (strain/sprain)
)

#: Degenerative / osteoarthritic change.
OA_TERMS = _alt(
    r"osteoarthritis", r"osteoarthrosis", r"osteoarthritic", r"arthrosis",
    r"arthrose", r"arthrosic", r"artrosis", r"artrose", r"artrosi",
    r"gonarthrosis", r"gonartrosis", r"gonarthrose",
    r"degenerative\s+\w*\s*(?:changes?|disease|arthropathy)",
    r"cambios\s+degenerativos", r"alteracoes\s+degenerativas",
    r"degenerative?\s+veranderungen", r"osteoartrit\w*",
    r"cartilage\s+(?:loss|thinning|defect|damage|fissur\w*|heterogeneity)",
    r"chondral\s+(?:loss|thinning|defect|damage|fissur\w*)",
    r"chondromalacia", r"condromalacia", r"chondropath\w*", r"condropat\w*",
    r"chondrosis", r"condrosis",
    r"knorpel(?:schaden|verlust|defekt|fissur\w*)", r"kartilaj\s+kayb\w*",
    r"perdida\s+de\s+cartilago", r"perte\s+de\s+cartilage",
    r"joint\s+space\s+narrowing", r"pincement\s+articulaire",
    # osteophytes: "osteophyt-" covers most Indo-European spellings already;
    # "spur/spurring" is the everyday English synonym radiologists use just
    # as often and shares none of that root.
    r"osteophyt\w*", r"osteofit\w*", r"osteophyten",
    r"marginal\s+spur\w*", r"bony\s+spur\w*", r"spurring",
    # Greek: οστεοαρθρίτιδα/αρθρίτιδα (osteoarthritis), εκφυλιστικές
    # αλλοιώσεις (degenerative changes), οστεόφυτα (osteophytes), and the
    # circumlocution "εξάλειψη [του] ... χόνδρου" (elimination of cartilage,
    # i.e. full-thickness cartilage loss) seen verbatim in the corpus.
    r"οστεοαρθριτ\w*", r"αρθριτιδ\w*", r"γοναρθρ\w*",
    r"εκφυλιστικ\w*\s+αλλοιωσ\w*", r"οστεοφυτ\w*",
    r"εξαλειψ\w*.{0,25}χονδρ\w*", r"απωλει\w*\s+χονδρ\w*",
    # Bulgarian: артроза/остеоартрит (arthrosis/osteoarthritis), дегенеративни
    # промени (degenerative changes), остеофити (osteophytes), загуба на
    # хрущял (cartilage loss).
    r"артроз\w*", r"остеоартрит\w*", r"дегенеративн\w*\s+промен\w*",
    r"остеофит\w*", r"загуба\s+на\s+хрущял\w*",
)

#: Words that put a finding inside bone rather than soft tissue.
BONE = _alt(
    r"bone", r"bony", r"osseous", r"osseus", r"marrow", r"medullary",
    r"osea?s?", r"osseas?", r"osseux", r"osseuse", r"knochen\w*", r"knochig",
    r"kemik", r"been(?:merg)?", r"ossea", r"midollo", r"medular", r"subchondral",
    r"subcondral", r"spongiosa",
    # Greek: οστ- is the "bone" root (οστό/οστική/οστεομυελικό...).
    r"οστ\w*",
    # Bulgarian: кост- is the "bone" root (кост/костномозъчен...).
    r"кост\w*",
)


@dataclass(frozen=True)
class ConceptPattern:
    """How to recognise one concept in text."""

    concept: str
    direct: list[str] = field(default_factory=list)
    anchor: list[str] = field(default_factory=list)
    qualifier: list[str] = field(default_factory=list)
    #: Max characters between anchor and qualifier for them to count as related.
    window: int = 80

    def compiled_direct(self) -> re.Pattern[str] | None:
        return _compile(self.direct)

    def compiled_anchor(self) -> re.Pattern[str] | None:
        return _compile(self.anchor)

    def compiled_qualifier(self) -> re.Pattern[str] | None:
        return _compile(self.qualifier)


def _compile(options: list[str]) -> re.Pattern[str] | None:
    if not options:
        return None
    return re.compile(_alt(*options), re.UNICODE)


# --------------------------------------------------------------------------
# The ten concepts
# --------------------------------------------------------------------------

_ACL_STRUCTURE = [
    r"\bacl\b", r"\blca\b", r"\bvkb\b", r"\blcae?\b",
    r"anterior\s+cruciate(?:\s+ligament)?",
    r"ligamento\s+cruzado\s+anterior",
    r"ligament\s+croise\s+anterieur",
    r"legamento\s+crociato\s+anteriore",
    # German adjectives decline: vordere/vorderen/vorderem/vorderer/vorderes.
    r"vordere[nmrs]?\s+kreuzband\w*", r"voorste\s+kruisband",
    r"on\s+capraz\s+bag",
    # Greek: πρόσθιος χιαστός σύνδεσμος (anterior cruciate ligament); the
    # structure alone ("χιαστός σύνδεσμος") is ambiguous between ACL/PCL, so
    # this requires the "anterior" qualifier, unlike some other languages
    # above where the ligament name is unambiguous.
    r"προσθιο\w*\s+χιαστ\w*(?:\s+συνδεσμ\w*)?",
    # Bulgarian: предна кръстна връзка (anterior cruciate ligament).
    r"предна\s+кръстна\s+връзк\w*",
]

_MCL_STRUCTURE = [
    r"\bmcl\b", r"\blcm\b", r"\bmkl\b",
    r"medial\s+collateral(?:\s+ligament)?",
    r"ligamento\s+colateral\s+medial",
    r"ligament\s+collateral\s+medial",
    r"ligament\s+lateral\s+interne",
    r"legamento\s+collaterale\s+mediale",
    r"innenband", r"mediales?\s+kollateralband",
    r"mediale\s+collaterale\s+band",
    r"ic\s+yan\s+bag",
    # Greek: έσω πλάγιος σύνδεσμος (medial collateral ligament).
    r"εσω\s+πλαγιο\w*\s+συνδεσμ\w*",
    # Bulgarian: вътрешна странична връзка (medial/inner collateral ligament).
    r"вътрешн\w*\s+странична?\w*\s+връзк\w*",
]

_MENISCUS_STRUCTURE = [
    r"menisc\w*", r"menisq\w*", r"menisk\w*", r"menisco\w*",
    r"μηνισκ\w*",  # Greek
    r"мениск\w*",  # Bulgarian
]

_PATELLOFEMORAL = [
    r"patell?ofemoral", r"femoropatell?ar", r"femoro-?patell?aire",
    r"patelo?femoral", r"retropatell?ar", r"retropatell?aire",
    r"patell?ar\s+(?:cartilage|facet)", r"trochlear?", r"troclea\w*",
    r"patell?a\b", r"rotula", r"kniescheibe", r"patellofemoralgelenk",
    r"patellofemoral\s+eklem",
    r"μηροκνημιαι\w*", r"επιγονατιδομηριαι\w*",  # Greek: patellofemoral
]

CONCEPT_PATTERNS: dict[str, ConceptPattern] = {
    # --- Ligaments: structure + injury required -------------------------
    "acl": ConceptPattern(
        concept="acl",
        anchor=_ACL_STRUCTURE,
        qualifier=[INJURY, SPRAIN],
    ),
    "mcl": ConceptPattern(
        concept="mcl",
        anchor=_MCL_STRUCTURE,
        qualifier=[INJURY, SPRAIN],
    ),
    # --- Menisci: structure + injury, then laterality-split -------------
    # NOTE: the true radiological criterion is signal *reaching an articular
    # surface*; internal signal alone is degeneration, not a tear. Reports
    # usually say so ("grade 2 signal, not extending to the surface"). Handling
    # that distinction is a known refinement — see docs/05-plan.md Phase 1.
    "meniscus": ConceptPattern(
        concept="meniscus",
        anchor=_MENISCUS_STRUCTURE,
        qualifier=[INJURY],
    ),
    # --- Osteoarthritis -------------------------------------------------
    # tf_oa fires on an OA term and is kept only if laterality resolves to
    # medial or lateral. "Patellofemoral osteoarthritis" therefore drops out of
    # tf_oa naturally (patellofemoral is neither) and is caught by pf_oa.
    "tf_oa": ConceptPattern(
        concept="tf_oa",
        direct=[OA_TERMS],
    ),
    "pf_oa": ConceptPattern(
        concept="pf_oa",
        anchor=_PATELLOFEMORAL,
        qualifier=[OA_TERMS],
        window=100,
    ),
    # --- Fluid and inflammation ----------------------------------------
    "effusion": ConceptPattern(
        concept="effusion",
        direct=[
            r"effusions?", r"efusion\w*", r"derrame\s*\w*", r"epanchement\w*",
            r"erguss", r"gelenkerguss", r"versamento", r"gewrichtsvocht",
            r"hidrartros\w*", r"hydrarthros\w*", r"efuzyon",
            r"eklem\s+sivisi", r"joint\s+fluid\s+(?:collection|distension)",
            r"suprapatell?ar\s+distension",
            # "hydrops" (of the knee/joint) is the everyday Dutch/German term
            # for effusion, distinct from "Erguss" and easy to miss because
            # in English "hydrops" usually means something unrelated (fetal
            # hydrops) — but this corpus is multilingual radiology, where a
            # bare "hydrops" next to a knee study is joint effusion.
            r"hydrops",
            # Dutch circumlocution seen verbatim: "opzetting van [de]
            # suprapatellaire recessus" (distension of the suprapatellar
            # recess) — the effusion is named by its location, not a word
            # meaning "fluid".
            r"opzetting\s+van\s+(?:de\s+)?suprapatellaire\s+recessus",
            # Turkish: reports here favour "sıvı artışı/artmış" (fluid
            # increase) over a dedicated word for effusion. Loosely bound to
            # a joint/bursa word so it doesn't fire on unrelated fluid
            # increases (e.g. muscle edema) elsewhere in the same report.
            r"(?:eklem\w*|suprapatellar\w*|bursa\w*)[^.;]{0,40}sivi\w*\s+art\w*",
            r"sivi\w*\s+art\w*[^.;]{0,40}(?:eklem\w*|suprapatellar\w*|bursa\w*)",
            # Greek: "υγρό ενδαρθρικά" / "ενδαρθρικό υγρό" (intra-articular
            # fluid) and "συλλογή υγρού" (fluid collection).
            r"υγρ\w*\s+ενδαρθρικ\w*", r"ενδαρθρικ\w*\s+υγρ\w*",
            r"συλλογ\w*\s+υγρου",
            # Bulgarian: ставен излив ("joint effusion", literally
            # "articular discharge") and bare хидропс (hydrops, cognate).
            r"ставен\s+излив", r"хидропс",
        ],
    ),
    "synovitis": ConceptPattern(
        concept="synovitis",
        direct=[
            r"synovit\w*", r"sinovit\w*", r"synovialit\w*",
            r"synovial\s+(?:thickening|proliferation|hypertrophy|enhancement)",
            r"engrosamiento\s+sinovial", r"espessamento\s+sinovial",
            r"epaississement\s+synovial", r"synovialisproliferation",
            r"sinovyal\s+kalinlasma", r"pannus",
            r"συνοβιτιδ\w*", r"συνοβι\w*\s+παχυνσ\w*",  # Greek
            r"синовит\w*",  # Bulgarian
        ],
    ),
    "bakers": ConceptPattern(
        concept="bakers",
        direct=[
            r"bakers?\s*(?:-|\s)?\s*(?:cyst|zyste|kist\w*|cisto|cisti|kyste)?",
            r"poplite(?:al|o|us|ale)\s+(?:cyst|quiste|cisto|cisti|kist\w*)",
            r"kyste\s+poplite\w*", r"quiste\s+poplite\w*",
            r"cisto\s+poplite\w*", r"poplitealzyste", r"popliteal\s+kist\w*",
            r"gastrocnemio-?semimembranosus\s+bursa",
            # Greek: κύστη Baker is common as-is; also ιγνυακή κύστη
            # (popliteal cyst — "ιγνύα" = popliteal fossa) and the generic
            # "συνοβιακή κύστη" (synovial cyst) which in a knee-MRI report is
            # this finding.
            r"κυστη\s+baker", r"ιγνυακ\w*\s+κυστ\w*", r"συνοβιακ\w*\s+κυστ\w*",
            # Bulgarian: бейкърова киста (Baker's cyst), поплитеална киста
            # (popliteal cyst).
            r"бейкъров\w*\s+кист\w*", r"поплитеалн\w*\s+кист\w*",
        ],
    ),
    # --- Bone -----------------------------------------------------------
    # Anchor requires the EXPLICIT contusion/bruise word, not bare
    # "edema"/"oedema". Measured on the gold studies (session 4): matching
    # "edema" + a nearby bone word (osseous/subchondral/marrow/...) fires on
    # far more than contusions — subchondral bone marrow edema is just as
    # characteristic of osteoarthritis, an adjacent osteochondral defect, or
    # a reactive change near an unrelated ligament tear, and gold labels
    # reserve Contusion for the specific traumatic bone-bruise pattern.
    # Bare "edema"/"bone" gave 17 false positives against 12 true positives
    # (precision 0.41, the worst of the twelve labels) — requiring the word
    # actually used for the finding trades some recall for a lot of it.
    "contusion": ConceptPattern(
        concept="contusion",
        anchor=[
            r"contusion\w*", r"contusao", r"contusoes", r"bruis\w*",
            r"kontuzyon\w*",
            r"контузи\w*",  # Bulgarian
        ],
        qualifier=[BONE],
        window=60,
    ),
    "fracture": ConceptPattern(
        concept="fracture",
        direct=[
            r"fractures?", r"fractura\w*", r"fratura\w*", r"frattura\w*",
            # Same Turkish consonant softening as "yirtik" above: "kırık"
            # (fracture) -> "kırığı" (its fracture, k->ğ before the vowel
            # suffix). Both normalize to differ only in the final consonant.
            r"fraktur\w*", r"kiri[kg]\w*", r"breuk\w*", r"fissure\s+osseuse",
            r"avulsion\s+fragment",
            r"καταγμ\w*",  # Greek: κάταγμα (fracture)
            r"фрактур\w*",  # Bulgarian (cognate)
        ],
    ),
}


# --------------------------------------------------------------------------
# Context vocabulary: negation, uncertainty, laterality
# --------------------------------------------------------------------------

#: Negation appearing BEFORE the finding (most Indo-European languages).
NEGATION_PRE = [
    r"no\s+evidence\s+of", r"no\s+sign\w*\s+of", r"no\s+definite",
    r"\bno\b", r"\bnot\b", r"without", r"absence\s+of", r"\babsent\b",
    r"negative\s+for", r"free\s+of", r"rule[ds]?\s+out", r"denies",
    r"resolution\s+of",
    # Spanish / Portuguese
    r"\bsin\b", r"\bsem\b", r"ausencia\s+de", r"no\s+se\s+(?:observa|identifica|aprecia|visualiza)",
    r"no\s+hay", r"nao\s+ha", r"nao\s+se\s+(?:observa|identifica)",
    r"sin\s+evidencia\s+de", r"sem\s+evidencia\s+de",
    # French
    # French elides before a vowel: "pas d'epanchement", not "pas de". Missing
    # this scored a negated effusion as present.
    r"pas\s+d[e']", r"\bsans\b", r"absence\s+d[eu']", r"aucun[e]?",
    r"n'?est\s+pas", r"non\s+visualise\w*",
    # German
    r"\bkein[e]?[snmr]?\b", r"\bohne\b", r"kein\s+hinweis\s+auf",
    r"nicht\s+\w*", r"unauffallig\w*",
    # Italian
    r"\bnon\b", r"\bsenza\b", r"assenza\s+di", r"nessun[ao]?",
    # Dutch
    r"\bgeen\b", r"\bzonder\b",
    # Greek: δεν (not), χωρίς (without), απουσία (absence of).
    r"\bδεν\b", r"\bχωρις\b", r"απουσια\s+\w*",
    # Bulgarian: без (without), не (not/no), липса на (absence of).
    r"\bбез\b", r"\bне\b", r"липса\s+на",
]

#: Negation appearing AFTER the finding. Turkish puts it there structurally, and
#: "the ACL is intact" is the same thing in English.
NEGATION_POST = [
    r"\bintact\b", r"\bintacto?s?\b", r"\bintegr[oa]s?\b", r"\bintatto\b",
    r"is\s+normal", r"are\s+normal", r"appears?\s+normal", r"unremarkable",
    r"\bpreserved\b", r"\bconserv\w+\b", r"normale?n?\b", r"unauffallig\w*",
    r"is\s+not\s+(?:seen|identified|visualized)",
    r"are\s+not\s+(?:seen|identified|visualized)",
    # Turkish observation verbs are dangerous: "izlenmektedir" (IS observed —
    # positive) and "izlenmemektedir" (is NOT observed — negative) both start
    # with the substring "izlenme", because "-mekte(dir)" is a present-tense
    # suffix that happens to start with "me" regardless of polarity — it's
    # not the negation morpheme itself. A bare `izlenme\w*` therefore matches
    # the affirmative form too and negates real positive findings. Found
    # session 5 chasing a Fracture false negative that said "kırığı
    # izlenmektedir" (fracture IS observed) and got wrongly negated.
    # `saptanma\w*` had the identical bug. `gorulume\w*` had it AND a typo
    # (extra "u" — "görülme" normalizes to "gorulme", not "gorulume", so this
    # one likely never matched its intended word at all). Enumerate the
    # actual negative suffixes instead of wildcarding past the ambiguous root.
    r"izlen(?:medi\w*|memis\w*|memekte\w*)",
    r"saptan(?:madi\w*|mamis\w*|mamakta\w*)",
    r"gorul(?:medi\w*|memis\w*|memekte\w*)",
    r"\byoktur\b", r"\byok\b",
    r"\bmevcut\s+degil\b",
    # Greek: φυσιολογικ- (normal), ακέραι- (intact).
    r"φυσιολογικ\w*", r"ακεραι\w*",
    # Bulgarian: нормал- (normal), запазен- (preserved/intact).
    r"нормал\w*", r"запазен\w*",
]

#: Words that close a negation's scope. "No effusion, but a meniscal tear" —
#: the tear is NOT negated.
TERMINATION = [
    r"\bbut\b", r"however", r"although", r"though", r"\bwith\b", r"whereas",
    r"showing", r"demonstrat\w+", r"associated\s+with", r"\bplus\b",
    r"\bpero\b", r"\baunque\b", r"\bcon\b", r"\bmas\b", r"\bporem\b",
    r"\bcom\b", r"\bmais\b", r"\bavec\b", r"\btoutefois\b", r"\bcependant\b",
    r"\baber\b", r"\bjedoch\b", r"\bmit\b", r"\bma\b", r"\bcon\b",
    r"\bmaar\b", r"\bmet\b", r"\bancak\b", r"\bile\b", r"\bfakat\b",
    r"[;:]",
]

#: Hedging. The competition metric is rank-based, so hedged findings are best
#: scored *between* absent and present rather than thrown away.
UNCERTAINTY = [
    r"possib\w*", r"probabl\w*", r"suspicious\s+for", r"suspect\w*",
    r"cannot\s+(?:be\s+)?exclude[d]?", r"can\s+not\s+be\s+excluded",
    r"questionable", r"equivocal", r"may\s+represent", r"could\s+represent",
    r"\blikely\b", r"appears?\s+to", r"borderline", r"indeterminate",
    r"concerning\s+for", r"favor\w*",
    r"posible\w*", r"sospech\w*", r"sugestiv\w*", r"sugere",
    r"possivel", r"provavel", r"suspeit\w*",
    r"suspicion", r"evoque", r"probable\w*",
    r"moglich\w*", r"verdacht\s+auf", r"fraglich", r"am\s+ehesten",
    r"possibil\w*", r"sospett\w*",
    r"mogelijk", r"verdenking",
    r"suphel?i", r"olasi", r"muhtemel",
    # Greek: πιθαν- (possible/probable), ύποπτ- (suspicious).
    r"πιθαν\w*", r"υποπτ\w*",
    # Bulgarian: вероятно (probably), възможн- (possible), съмнение за
    # (suspicion of).
    r"вероятно", r"възможн\w*", r"съмнение\s+за",
    r"\?",
]

LATERALITY_MEDIAL = [
    # "interne" (fr) / "interno" (es,pt) / "intern" — one pattern for all.
    r"\bmedial\w*\b", r"\bmedio\w*\b", r"\binner\b", r"\bintern[eo]?s?\b",
    r"\binnen\w*\b", r"\bmediaal\b", r"\bmediale[nrs]?\b",
    r"\bic\s+yan\b", r"\bmedialis\b",
    r"\bεσω\b",  # Greek: έσω (medial/inner)
    r"медиал\w*",  # Bulgarian
]

LATERALITY_LATERAL = [
    r"\blateral\w*\b", r"\bextern[eo]?s?\b", r"\bouter\b", r"\bexternal\b",
    r"\bausse\w*\b", r"\blateraal\b", r"\blaterale[nrs]?\b",
    r"\bdis\s+yan\b", r"\blateralis\b",
    r"\bεξω\b",  # Greek: έξω (lateral/outer)
    r"латерал\w*",  # Bulgarian
]

#: "Both menisci", "medial and lateral compartments".
LATERALITY_BOTH = [
    r"\bboth\b", r"\bbilateral\b", r"\bambos\b", r"\bambas\b",
    r"\bles\s+deux\b", r"\bbeide[nr]?\b", r"\bentrambi\b",
    r"medial\s+and\s+lateral", r"lateral\s+and\s+medial",
    r"medial\w*\s+y\s+lateral\w*", r"medial\w*\s+e\s+lateral\w*",
    r"αμφοτεροπλευρ\w*",  # Greek: αμφοτερόπλευρα (bilaterally)
    r"двустранн\w*",  # Bulgarian: двустранно (bilaterally)
]

#: Patellofemoral compartment — resolves an OA mention away from tibiofemoral.
LATERALITY_PATELLOFEMORAL = _PATELLOFEMORAL
