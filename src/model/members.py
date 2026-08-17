"""What a checkpoint says it was trained on, across three publishers.

Ensembling other people's weights with ours means reading configuration
written by people who never agreed on a schema. Ours records `slices_per_slot`
at the top level; the twelve-slice public members record `config.slices` and a
`config.band` LIST; the champ members record a `fingerprint` dict with
`group` x `n_group` and a `window` STRING.

That last difference is why this lives in a module instead of inside the
submission notebook. A string is iterable, so parsing `"0.35,0.65"` with the
same code that handles `[0.2, 0.8]` walks characters and dies on `float('.')`
— which it did, after loading all thirty-five members, ~10 minutes into a GPU
run, leaving a file of 0.5s. The logic is small, it is guessy by nature, and
every checkpoint we might ever blend is sitting on disk: it should be tested
against them locally, not discovered on the accelerator.

Nothing here infers anything from tensor shapes. Every one of these
configurations produces identically-shaped tensors and differs only in which
anatomy is inside them, so a member that fails to declare itself is refused
rather than defaulted.
"""

from __future__ import annotations

import os
import re

#: What an unparseable band becomes. It compares unequal to any real band, so
#: a member carrying one is refused rather than quietly exempted from the check.
BAD_BAND = ("unparseable",)


def as_band(value):
    """A declared slice band -> tuple of floats, whatever shape it arrived in.

    `None` means "not declared" and skips the check. `BAD_BAND` means "declared
    something we could not read" and fails it.
    """
    if value is None:
        return None
    parts = (re.split(r"[,\s]+", value.strip()) if isinstance(value, str)
             else list(value))
    try:
        return tuple(round(float(p), 3) for p in parts if p != "")
    except (TypeError, ValueError):
        return BAD_BAND


def member_fingerprint(path: str, blob: dict, slices_by_dir: dict | None = None):
    """(slices_per_slot, img, band) as this checkpoint declares them.

    Any element may be `None`, meaning the checkpoint is silent about it.
    `slices_by_dir` supplies a fallback keyed by mounted dataset directory, for
    our own early members that predate the field.
    """
    cfg = blob.get("config") or {}
    fp = blob.get("fingerprint") or {}

    slices = (cfg.get("slices") or blob.get("slices_per_slot")
              or blob.get("n_slice"))
    if slices is None and fp.get("group") and fp.get("n_group"):
        slices = int(fp["group"]) * int(fp["n_group"])
    if slices is None and slices_by_dir:
        parts = path.replace(os.sep, "/").split("/")
        for name, value in slices_by_dir.items():
            if name in parts:
                slices = value
                break

    img = cfg.get("img") or fp.get("img") or blob.get("size")

    band = as_band(cfg.get("band"))
    if band is None:
        band = as_band(fp.get("window"))

    return (None if slices is None else int(slices),
            None if img is None else int(img),
            band)


def refuse_reason(slices, img, band, want_img, want_band, group):
    """Why this member must not be scored on our cache, or None if it may be.

    Slice count is not the only thing that has to match. A member trained at
    224 px over band 0.35-0.65 loads cleanly into the same class and runs
    without complaint on 336 px data — DINOv2 interpolates its position
    embeddings — and simply reads the wrong scale. Five such members sat inside
    a scored submission before this check existed.
    """
    if slices is None:
        return "declares no slices/slot and no directory default"
    if slices % group:
        return f"{slices} slices/slot is not a multiple of {group}"
    if img is not None and int(img) != want_img:
        return f"trained at {img}px, cache is {want_img}px"
    if band is not None and tuple(band) != tuple(want_band):
        return f"trained on band {band}, cache uses {tuple(want_band)}"
    return None
