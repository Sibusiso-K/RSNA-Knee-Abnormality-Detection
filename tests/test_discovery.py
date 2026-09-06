"""find_files: prune skip dirs before descending, not after.

A recursive glob followed by filtering pays the FULL traversal cost of
everything it later throws away. Measured once already in this project on a
different search: ~1,100s per call against train_series/, most of a TPU
quota's worth, before a single training step (docs/00-state.md). These tests
check both that the skipped content is excluded from the result AND that it
is never actually descended into in the first place.
"""

import os

from src.data.discovery import find_files


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def test_finds_matching_files_under_a_root(tmp_path):
    _touch(tmp_path / "a" / "cache_train_0.npy")
    _touch(tmp_path / "b" / "cache_train_1.npy")
    _touch(tmp_path / "b" / "notes.txt")

    found = find_files([str(tmp_path)], "cache_train_*.npy")
    assert [os.path.basename(p) for p in found] == ["cache_train_0.npy", "cache_train_1.npy"]


def test_never_descends_into_a_skipped_directory(tmp_path):
    """Not just "the result excludes it" — the walk must never enter it."""
    _touch(tmp_path / "cache_train_0.npy")
    poison = tmp_path / "train_series"
    _touch(poison / "deep" / "nested" / "cache_train_99.npy")  # would match if ever reached

    visited = []
    real_walk = os.walk

    def spying_walk(top, *a, **kw):
        for directory, dirs, files in real_walk(top, *a, **kw):
            visited.append(directory)
            yield directory, dirs, files

    import src.data.discovery as discovery

    orig = discovery.os.walk
    discovery.os.walk = spying_walk
    try:
        found = find_files([str(tmp_path)], "cache_train_*.npy")
    finally:
        discovery.os.walk = orig

    assert len(found) == 1 and os.path.dirname(found[0]) == str(tmp_path)
    assert not any("train_series" in v for v in visited), (
        f"descended into a skip dir: {visited}"
    )


def test_matches_across_several_patterns_in_one_pass(tmp_path):
    _touch(tmp_path / "knee_slot_fold0.pth")
    _touch(tmp_path / "m_public7.pt")
    _touch(tmp_path / "champ_fold3.pt")
    _touch(tmp_path / "readme.md")

    found = find_files([str(tmp_path)], ["knee_slot_fold*.pth", "m_*.pt", "champ_fold*.pt"])
    assert {os.path.basename(p) for p in found} == {
        "knee_slot_fold0.pth", "m_public7.pt", "champ_fold3.pt",
    }


def test_the_same_file_reached_via_two_roots_is_not_a_conflict(tmp_path):
    _touch(tmp_path / "x" / "cache_train_0.npy")

    found = find_files([str(tmp_path / "x"), str(tmp_path)], "cache_train_*.npy")
    assert len(found) == 1


def test_two_different_files_sharing_a_basename_raise(tmp_path):
    """The exact shape of a real bug: two mounted datasets each with their
    own cache_train_0.npy, and a search that silently kept whichever it saw
    first trained on half the data and looked like a modelling result."""
    _touch(tmp_path / "dataset_a" / "cache_train_0.npy")
    _touch(tmp_path / "dataset_b" / "cache_train_0.npy")

    try:
        find_files([str(tmp_path)], "cache_train_*.npy")
    except ValueError as exc:
        assert "cache_train_0.npy" in str(exc)
    else:
        raise AssertionError("expected a ValueError for conflicting duplicate basenames")


def test_result_order_is_stable_and_sorted_by_basename(tmp_path):
    _touch(tmp_path / "cache_train_2.npy")
    _touch(tmp_path / "cache_train_10.npy")
    _touch(tmp_path / "cache_train_1.npy")

    found = find_files([str(tmp_path)], "cache_train_*.npy")
    assert [os.path.basename(p) for p in found] == [
        "cache_train_1.npy", "cache_train_10.npy", "cache_train_2.npy",
    ]
