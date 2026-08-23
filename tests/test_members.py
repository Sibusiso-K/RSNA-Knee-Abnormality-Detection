"""The member-metadata gate, against the shapes three publishers actually use.

Both bugs this file exists for were in reading someone else's configuration,
not in the model: a `band` recorded as a string killed a 35-member GPU run ten
minutes in, and before that five members trained at 224 px were scored on
336 px data without anything raising.
"""

from src.model.members import BAD_BAND, as_band, member_fingerprint, refuse_reason

WANT_BAND = (0.2, 0.8)
SBD = {"knee-slot-6slice-v1": 6, "knee-slot-bc-v1": 6, "rsna-knee-weights": 12}


def test_band_accepts_both_publishers_conventions():
    # A list from the twelve-slice members, a string from the champ ones. The
    # string is the case that raised: iterating it walks characters.
    assert as_band([0.2, 0.8]) == (0.2, 0.8)
    assert as_band("0.35,0.65") == (0.35, 0.65)
    assert as_band("0.2, 0.8") == (0.2, 0.8)
    assert as_band(None) is None


def test_unreadable_band_is_refused_not_ignored():
    # Returning None here would exempt the member from the check entirely,
    # which is the opposite of what "I cannot read this" should mean.
    assert as_band("early-late") is BAD_BAND
    assert refuse_reason(6, 336, BAD_BAND, 336, WANT_BAND, 3) is not None


def test_reads_each_publishers_schema():
    ours = {"slices_per_slot": 6, "size": 336}
    public = {"config": {"slices": 12, "img": 336, "band": [0.2, 0.8]}}
    champ = {"fingerprint": {"group": 3, "n_group": 3, "img": 224,
                             "window": "0.35,0.65"}}
    assert member_fingerprint("/kaggle/input/x/a.pth", ours, SBD) == (6, 336, None)
    assert member_fingerprint("/kaggle/input/x/m.pt", public, SBD) == (12, 336, (0.2, 0.8))
    assert member_fingerprint("/kaggle/input/x/c.pt", champ, SBD) == (9, 224, (0.35, 0.65))


def test_directory_fallback_only_for_our_early_members():
    # Our first six-slice family predates the field, so the mounted directory
    # supplies it. A checkpoint in an unknown directory must stay unknown.
    bare = {"size": 336}
    path = "/kaggle/input/knee-slot-6slice-v1/knee_slot_fold0.pth"
    assert member_fingerprint(path, bare, SBD)[0] == 6
    assert member_fingerprint("/kaggle/input/somewhere/x.pth", bare, SBD)[0] is None


def test_gate_refuses_the_members_that_actually_shipped_wrong():
    # champ: right head, right slots, wrong scale and wrong band. It loaded and
    # scored inside submission 55574007 before this existed.
    assert refuse_reason(9, 224, (0.35, 0.65), 336, WANT_BAND, 3) is not None
    assert refuse_reason(6, 336, None, 336, WANT_BAND, 3) is None
    assert refuse_reason(12, 336, (0.2, 0.8), 336, WANT_BAND, 3) is None
    # An undeclared grid is refused, never defaulted: the failure mode of a
    # wrong guess is a plausible score from the wrong anatomy.
    assert refuse_reason(None, 336, None, 336, WANT_BAND, 3) is not None
    assert refuse_reason(7, 336, None, 336, WANT_BAND, 3) is not None
