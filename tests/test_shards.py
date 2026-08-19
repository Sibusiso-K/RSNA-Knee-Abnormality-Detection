"""Sharded cache reads: order is the contract, not just the bytes.

Labels and folds are joined to the cache BY ROW POSITION. A reader that
returns the right pixels in the wrong order trains every study against another
study's labels, runs to completion, and produces a plausible score. That has
already happened once here in a different form — a half-loaded cache scored
0.7956 and looked like "more slices hurt".
"""

import numpy as np
import pytest

from src.data.shards import ShardedCache


@pytest.fixture
def shards(tmp_path):
    """Two shards written the way kaggle_05_cache.py writes them.

    The builder takes `studies[k::n_shard]`, so shard 0 holds the even studies
    and shard 1 the odd ones, and the index CSVs are concatenated in shard
    order to match. Row r of the concatenated view is therefore NOT study r.
    """
    whole = np.arange(10 * 2 * 3, dtype=np.uint8).reshape(10, 2, 3)
    paths = []
    for k in range(2):
        p = tmp_path / f"cache_train_{k}.npy"
        np.save(p, whole[k::2])
        paths.append(str(p))
    return paths, whole


def test_shape_and_length_span_every_shard(shards):
    paths, whole = shards
    c = ShardedCache(paths)
    assert c.shape == whole.shape and len(c) == 10


def test_row_order_is_shard0_then_shard1(shards):
    paths, whole = shards
    c = ShardedCache(paths)
    expected = np.concatenate([whole[0::2], whole[1::2]])
    assert np.array_equal(c[np.arange(10)], expected)


def test_unsorted_rows_come_back_in_the_callers_order(shards):
    """A shuffled epoch produces unsorted batches spanning both shards.

    Grouping rows by shard for efficient reads is exactly where an ordering
    bug hides: the pixels are all present and correct, just against the wrong
    labels.
    """
    paths, _ = shards
    c = ShardedCache(paths)
    rows = np.array([7, 0, 9, 3, 4])
    got = c[rows]
    for i, r in enumerate(rows):
        assert np.array_equal(got[i], c[int(r)]), f"row {r} landed at {i}"


def test_single_row_and_boolean_mask(shards):
    paths, _ = shards
    c = ShardedCache(paths)
    assert np.array_equal(c[6], c[np.array([6])][0])
    mask = np.zeros(10, dtype=bool)
    mask[[1, 8]] = True
    assert np.array_equal(c[mask], c[np.array([1, 8])])


def test_nothing_is_materialised(shards):
    """The whole point: shards stay memmapped, never read into RAM."""
    paths, _ = shards
    c = ShardedCache(paths)
    assert all(isinstance(p, np.memmap) for p in c.parts)


def test_mismatched_shards_are_refused(shards, tmp_path):
    """Shards from different builds must fail loudly, not concatenate."""
    paths, _ = shards
    odd = tmp_path / "cache_train_2.npy"
    np.save(odd, np.zeros((3, 2, 4), dtype=np.uint8))    # 4 != 3
    with pytest.raises(ValueError, match="different builds"):
        ShardedCache(paths + [str(odd)])
