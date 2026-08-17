"""`read_slot_multi` must be `read_slot`, run several times, for less money.

That equivalence is the whole safety argument for the mixed-grid submission:
members trained at six slices per slot and members trained at twelve are scored
in the same run, and the six-slice grid is fetched from a shared decode pool
rather than from its own pass over the study. If the pooled path differs from
the original by so much as the intensity normalisation, fifteen members get
scored on pixels they were never trained on and the run still finishes, still
writes a valid file, and still produces a believable number.

The submission notebook re-checks this on real test studies before scoring
anything, but that check runs once, remotely, on the hidden set. This one runs
here, on every commit, without DICOMs: pydicom and cv2 are stubbed so the test
exercises the index arithmetic, the decode-failure fallback and the per-grid
normalisation, which is where the divergence would actually live.
"""

import sys
import types

import numpy as np
import pytest

import src.data.cache as cache
from src.data.slots import IMG, N_SLOT, band_indices


class _Stub:
    """Stands in for a pydicom dataset. `n` seeds a distinct pixel pattern."""

    def __init__(self, n, spacing=0.4):
        self.n = n
        self.PixelSpacing = [spacing, spacing]

    @property
    def pixel_array(self):
        # A per-slice gradient, not noise: any index mix-up shows up as a
        # frame in the wrong place rather than as a plausible-looking blur.
        base = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
        return (base * 100.0 + self.n * 37.0).astype(np.float32)


@pytest.fixture
def stub_decoders(monkeypatch):
    """Stub pydicom/cv2 and make headers resolve from the file NAME."""
    fake_pydicom = types.ModuleType("pydicom")
    fake_pydicom.dcmread = lambda path, **_: _Stub(int(path.split("_")[-1]))
    fake_pydicom.config = types.SimpleNamespace(pixel_data_handlers=[])

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.INTER_AREA = 3

    def resize(pixels, size, interpolation=None):
        out = np.zeros(size, dtype=np.float32)
        out[: min(size[0], pixels.shape[0]), : min(size[1], pixels.shape[1])] = (
            pixels[: size[0], : size[1]]
        )
        return out

    fake_cv2.resize = resize

    monkeypatch.setitem(sys.modules, "pydicom", fake_pydicom)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(cache, "read_header",
                        lambda path: _Stub(int(path.split("_")[-1])))
    monkeypatch.setattr(cache, "slice_key", lambda ds: float(ds.n))
    monkeypatch.setattr(cache, "physical_crop", lambda px, mm: (px, True))
    return fake_pydicom


@pytest.mark.parametrize("n_files", [9, 17, 30, 41])
@pytest.mark.parametrize("plane", ["Sagittal", "Coronal"])
def test_multi_matches_single_grid_for_grid(stub_decoders, monkeypatch,
                                            n_files, plane):
    files = [f"s_{i}" for i in range(n_files)]
    takes = (6, 12)

    got = cache.read_slot_multi(files, plane, "L", takes)
    for k in takes:
        monkeypatch.setattr(cache, "N_SLICE", k)
        want_volume, want_cropped = cache.read_slot(files, plane, "L")
        assert got[k][0].shape == (k, IMG, IMG)
        assert np.array_equal(got[k][0], want_volume), (
            f"{k}-slice grid diverges from read_slot at n_files={n_files}"
        )
        assert got[k][1] == want_cropped


def test_grids_do_not_nest_so_subsampling_would_be_wrong():
    """The reason the pooled decode exists at all.

    If six evenly spaced points across the band were always a subset of twelve,
    the submission could build one cache and slice it. They are not: the points
    sit at k/5 and k/11 of the band and coincide only at the endpoints, so for
    most series lengths at least one six-slice index is absent from the
    twelve-slice grid. Pinned here because "just subsample" is the obvious
    shortcut and it is wrong in a way that shows up as a mediocre score.
    """
    disjoint = sum(
        not set(band_indices(n, 6).tolist()) <= set(band_indices(n, 12).tolist())
        for n in range(12, 60)
    )
    assert disjoint > len(range(12, 60)) // 2


def test_union_decodes_each_slice_once(stub_decoders, monkeypatch):
    """Two grids must cost far less than two passes, or this is pointless."""
    seen = []
    original = stub_decoders.dcmread
    monkeypatch.setattr(stub_decoders, "dcmread",
                        lambda path, **kw: (seen.append(path), original(path))[1])

    files = [f"s_{i}" for i in range(30)]
    cache.read_slot_multi(files, "Sagittal", "R", (6, 12))

    assert len(seen) == len(set(seen)), "a slice was decoded more than once"
    assert len(seen) < 18, f"union should be under 6+12 decodes, got {len(seen)}"


def test_undecodable_slice_falls_back_within_its_own_grid(stub_decoders,
                                                          monkeypatch):
    """A failed decode must not leak a black frame or a neighbouring grid's."""
    files = [f"s_{i}" for i in range(30)]
    bad = int(band_indices(30, 12)[4])

    original = stub_decoders.dcmread

    def flaky(path, **kw):
        if int(path.split("_")[-1]) == bad:
            raise RuntimeError("no decoder")
        return original(path)

    monkeypatch.setattr(stub_decoders, "dcmread", flaky)

    got = cache.read_slot_multi(files, "Coronal", None, (6, 12))
    for k in (6, 12):
        monkeypatch.setattr(cache, "N_SLICE", k)
        want, _ = cache.read_slot(files, "Coronal", None)
        assert np.array_equal(got[k][0], want)


def test_build_study_multi_shapes_and_mask(stub_decoders, monkeypatch):
    """One mask for every grid: a slot is present or it is not."""
    def fake_probe(series_dir, study_uid, plane_of):
        info = types.SimpleNamespace(uid="u0", plane="Sagittal")
        files = {"u0": [f"s_{i}" for i in range(20)]}
        return [info], files, "fp"

    monkeypatch.setattr(cache, "probe_study", fake_probe)
    monkeypatch.setattr(cache, "study_side", lambda infos: "R")
    monkeypatch.setattr(cache, "pick_slot_series",
                        lambda infos: [infos[0]] + [None] * (N_SLOT - 1))

    volumes, mask, row, _infos, (ok, total) = cache.build_study_multi(
        "dir", "study", {}, (6, 12)
    )

    assert set(volumes) == {6, 12}
    for k in (6, 12):
        assert volumes[k].shape == (N_SLOT, k, IMG, IMG)
        assert volumes[k][1:].max() == 0, "absent slots must stay zero"
    assert mask.tolist() == [1] + [0] * (N_SLOT - 1)
    assert (ok, total) == (1, 1), "crop counted once per slot, not once per grid"
    assert row["side"] == "R"
