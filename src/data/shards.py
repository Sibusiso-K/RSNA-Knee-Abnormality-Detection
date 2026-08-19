"""Read a multi-shard cache without ever holding it in RAM.

`np.concatenate([np.load(p) for p in paths])` loads every shard AND allocates a
fresh array for the result, so peak memory is twice the cache. That is fine at
8.96 GB and fatal at 29.7 GB: the 448 px x 6 slice cache is 29.7 GB against the
29.9 GB a Kaggle worker reports available, and the run died with a bare
`Killed` after ten minutes of loading — no traceback, because the OOM killer
does not leave one.

A cache is only ever read as `cache[rows]` for a batch of ~4-8 study indices,
so it does not need to be contiguous or resident. This maps each global row to
(shard, local row) and reads through per-shard memmaps.

**Concatenation order is the contract.** The builder writes shard `k` as
`studies[k::n_shard]`, so shard 0 holds studies 0, 2, 4... and shard 1 holds
1, 3, 5... Neither is in original order, and the index CSVs are concatenated
shard-0-then-shard-1 to match. This class must reproduce exactly that order —
labels and folds are joined by row position, so an ordering that disagrees
would train every study against another study's labels and still run.
"""

from __future__ import annotations

import numpy as np


class ShardedCache:
    """Read-only, memmap-backed view over cache shards concatenated in order."""

    def __init__(self, paths: list[str]):
        if not paths:
            raise ValueError("no cache shards")
        self.parts = [np.load(p, mmap_mode="r") for p in paths]
        tail = self.parts[0].shape[1:]
        for path, part in zip(paths, self.parts):
            if part.shape[1:] != tail:
                raise ValueError(
                    f"shard {path} has per-study shape {part.shape[1:]}, "
                    f"expected {tail} — shards from different builds"
                )
        self.lengths = [len(p) for p in self.parts]
        self.offsets = np.cumsum([0] + self.lengths)
        self.shape = (int(self.offsets[-1]), *tail)
        self.dtype = self.parts[0].dtype
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, rows):
        """`cache[rows]` for an array of global row indices -> one real array.

        Rows are grouped by shard and written back in the caller's order, so
        an unsorted batch (which is what a shuffled epoch produces) comes back
        correctly ordered rather than silently permuted.
        """
        if isinstance(rows, (int, np.integer)):
            shard = int(np.searchsorted(self.offsets, rows, side="right") - 1)
            return np.asarray(self.parts[shard][rows - self.offsets[shard]])

        rows = np.asarray(rows)
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        out = np.empty((len(rows), *self.shape[1:]), dtype=self.dtype)
        shard_of = np.searchsorted(self.offsets, rows, side="right") - 1
        for shard in np.unique(shard_of):
            where = np.flatnonzero(shard_of == shard)
            local = rows[where] - self.offsets[shard]
            # np.take on a memmap reads only the requested rows; sorting the
            # local indices keeps the reads roughly sequential on disk.
            order = np.argsort(local)
            out[where[order]] = self.parts[shard][local[order]]
        return out
