"""Directory search that prunes before it descends, not after.

Every Kaggle notebook in this project needs to find files (a cache shard, a
member checkpoint) somewhere under a handful of candidate roots, one of which
is `/kaggle/input` — and `/kaggle/input` can contain `train_series/` or
`test_series/`, hundreds of thousands of DICOM files across tens of thousands
of nested directories. A recursive glob (`glob.glob(f"{root}/**/{pattern}",
recursive=True)`) walks the ENTIRE tree, including those, before any filtering
happens — filtering the result list afterward does not avoid that cost, it
just deletes the entries after paying for them. Measured once already on a
different search in this project: ~1,100s per call, most of a TPU quota,
before a single training step (docs/00-state.md, session log). The fix is to
prune the skip set from `os.walk`'s own `dirs` list before it descends, the
same way `find_dir` in every notebook here already locates a directory by
content.
"""

from __future__ import annotations

import fnmatch
import os

#: Directories never descended into while searching by content or pattern.
SKIP_DIRS = {"train_series", "test_series", ".git", "__pycache__", "pkg"}


def find_files(roots, patterns, skip_dirs=SKIP_DIRS):
    """Every file under `roots` matching any of `patterns`.

    `roots` may repeat or nest (e.g. `[INPUT, ".", ".."]`, where `.` can sit
    inside `INPUT` depending on the working directory) — each root is walked
    exactly once, and a file reachable through more than one root is only
    counted once. `patterns` is a single glob-style pattern or an iterable of
    them, matched against the basename only, so a caller collecting several
    naming conventions at once (our own checkpoints plus two public ones, say)
    does it in one pass instead of one walk per pattern.

    Returned in a fixed order (sorted by basename) for reproducibility — not
    the same order a shell glob would give, and callers must not depend on
    matching that.

    Two DIFFERENT files with the same basename raise, rather than silently
    keeping whichever was found first. That silent-first-wins behaviour is
    exactly the shape of an earlier real bug here: a sharded cache is mounted
    as one directory per shard, and a search that quietly kept only the first
    `cache_train_0.npy` it saw (out of two, from different mounts) trained on
    half the data and looked like "more slices hurt" instead of "half the
    data" (docs/00-state.md). Two paths that resolve to the SAME underlying
    file (reachable through more than one root) are not a conflict and are
    simply deduplicated.
    """
    if isinstance(patterns, str):
        patterns = (patterns,)

    found: dict[str, str] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for directory, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for name in files:
                if not any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                    continue
                path = os.path.join(directory, name)
                existing = found.get(name)
                if existing is not None and os.path.realpath(existing) != os.path.realpath(path):
                    raise ValueError(
                        f"conflicting sources for {name!r}: "
                        f"{existing!r} vs {path!r} are different files sharing a name"
                    )
                found.setdefault(name, path)

    return [found[name] for name in sorted(found)]
