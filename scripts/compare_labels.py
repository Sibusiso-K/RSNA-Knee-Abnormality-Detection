"""Score every available label set against the 58 gold studies.

Why this script exists
----------------------
Label quality is the ceiling on this competition: only 58 of 4,407 training
studies carry real per-condition annotations, so everyone trains against labels
they manufactured from the report text. Our own ensemble measures 0.8234 against
gold. Several public sets claim 0.87-0.89. If that holds, the cheapest available
improvement is to stop using ours.

"If that holds" is the point. A claimed number computed by its own author on its
own alignment is not a number we can spend GPU hours on, so every candidate is
re-scored here on the same 58 studies, through the same code, with the same
missing-value handling.

Honest limits, which the 58-study ruler imposes on every row it prints:

- Differences below ~0.02 macro are not resolvable on 58 studies. Treat a small
  gap as unknown, not as zero.
- Gold prevalence is the annotator's sampling, not disease prevalence: every
  gold study has at least one positive finding, mean 4.14.
- A better key is not automatically a better model. This ranks label sources; it
  does not predict AUC.

Usage:
    python scripts/compare_labels.py --dir <folder of candidate CSVs>
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from src.labels import TARGETS

ID = "StudyInstanceUID"


def macro_auc_vs_gold(gold: pd.DataFrame, pred: pd.DataFrame):
    """Macro AUC of `pred` against `gold`, plus the per-label breakdown.

    Only studies present in both frames are scored, and only labels with both
    classes present in gold — a label whose gold column is constant has no AUC,
    and averaging a NaN into the macro would silently understate everything.
    """
    from sklearn.metrics import roc_auc_score

    joined = gold.merge(pred, on=ID, suffixes=("_gold", "_pred"), how="inner")
    if joined.empty:
        return float("nan"), {}, 0

    per_label: dict[str, float] = {}
    for target in TARGETS:
        y = joined[f"{target}_gold"].values.astype(float)
        p = joined[f"{target}_pred"].values.astype(float)
        keep = ~(np.isnan(y) | np.isnan(p))
        if keep.sum() < 2 or len(set(y[keep])) < 2:
            continue
        per_label[target] = float(roc_auc_score(y[keep], p[keep]))

    macro = float(np.mean(list(per_label.values()))) if per_label else float("nan")
    return macro, per_label, len(joined)


def audit(gold: pd.DataFrame, frame: pd.DataFrame) -> tuple[int, bool]:
    """(studies usable for training, reproduces gold exactly).

    Two failure modes this catches, both of which score *perfectly* and are
    worth nothing:

    1. **Circularity.** A file that carries `train.csv`'s own label columns
       reproduces gold on all 58 studies and scores 1.0000. It is not a
       labeller; it is the answer key. Blending it in would then drag the
       selection toward itself and quietly define the whole label strategy.
       Measured: `barun2104/train_folds.csv` is exactly this — 696/696 cells
       identical, NaN everywhere else.
    2. **No training coverage.** Labels only on the 58 gold studies cannot
       supervise the other 4,349, which is the entire job.

    Neither is dishonesty by the publisher — `train_folds.csv` is published as a
    *fold assignment*, and the labels ride along. It is only a trap for a
    harness that globs a directory and ranks by score, which is what this is.
    """
    gold_ids = set(gold[ID])
    outside = frame[~frame[ID].isin(gold_ids)]
    usable = int(outside[list(TARGETS)].notna().any(axis=1).sum())

    merged = gold.merge(frame, on=ID, suffixes=("_g", "_p"))
    if merged.empty:
        return usable, False
    cells = matches = 0
    for target in TARGETS:
        g = merged[f"{target}_g"].values.astype(float)
        p = merged[f"{target}_p"].values.astype(float)
        keep = ~(np.isnan(g) | np.isnan(p))
        cells += int(keep.sum())
        matches += int((g[keep] == p[keep]).sum())
    return usable, cells > 0 and matches == cells


def load_candidate(path: str) -> pd.DataFrame | None:
    """Read a candidate CSV down to ID + the twelve target columns."""
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if ID not in frame.columns:
        return None
    missing = [t for t in TARGETS if t not in frame.columns]
    if missing:
        return None
    out = frame[[ID] + list(TARGETS)].copy()
    for target in TARGETS:
        out[target] = pd.to_numeric(out[target], errors="coerce")
    return out.drop_duplicates(subset=[ID])


def rank_normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Map each target column to [0,1] by rank.

    Blending raw scores across sources assumes they share a scale, and they do
    not: one emits calibrated probabilities, another 0/1, another 0.5 for "not
    addressed". AUC reads order only, so ranking first makes a mean meaningful.
    """
    out = frame.copy()
    for target in TARGETS:
        out[target] = out[target].rank(pct=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="folder searched for CSVs")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--ours", default="data/labels_v1.csv")
    parser.add_argument("--out", default=None, help="write the blend here")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    gold_mask = train[TARGETS].notna().any(axis=1)
    gold = train.loc[gold_mask, [ID] + list(TARGETS)].reset_index(drop=True)
    print(f"gold studies: {len(gold)}")
    prevalence = {t: float(gold[t].mean(skipna=True)) for t in TARGETS}
    print("gold prevalence: "
          + "  ".join(f"{t}:{v:.2f}" for t, v in prevalence.items()))
    print()

    candidates: dict[str, pd.DataFrame] = {}
    paths = sorted(glob.glob(os.path.join(args.dir, "**", "*.csv"), recursive=True))
    if os.path.exists(args.ours):
        paths.append(args.ours)

    for path in paths:
        frame = load_candidate(path)
        if frame is None:
            continue
        name = os.path.relpath(path, args.dir) if path.startswith(args.dir) else path
        candidates[name] = frame

    rows = []
    for name, frame in candidates.items():
        macro, per_label, n = macro_auc_vs_gold(gold, frame)
        usable, is_gold_copy = audit(gold, frame)
        rows.append({"source": name, "macro": macro, "n_gold": n,
                     "trainable": usable, "n_labels": len(per_label),
                     "verdict": "GOLD COPY - excluded" if is_gold_copy
                     else ("no coverage - excluded" if usable < 100 else "ok")})

    table = pd.DataFrame(rows).sort_values("macro", ascending=False)
    print("=== individual label sets, scored on the same 58 studies ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()

    usable_table = table[table["verdict"] == "ok"]
    excluded = table[table["verdict"] != "ok"]
    if len(excluded):
        print("EXCLUDED (a perfect score here means circular, not good):")
        for _, r in excluded.iterrows():
            print(f"  {r['source']}  macro {r['macro']:.4f}  -> {r['verdict']}")
        print()

    # Blend the strongest few. Rank-mean rather than mean: see rank_normalise.
    top = [r["source"] for _, r in usable_table.head(3).iterrows()
           if r["macro"] == r["macro"]]
    if len(top) >= 2:
        ids = set(candidates[top[0]][ID])
        for name in top[1:]:
            ids &= set(candidates[name][ID])
        ids = sorted(ids)
        stack = []
        for name in top:
            frame = candidates[name].set_index(ID).loc[ids].reset_index()
            stack.append(rank_normalise(frame)[TARGETS].values)
        blend = pd.DataFrame(np.mean(stack, axis=0), columns=list(TARGETS))
        blend.insert(0, ID, ids)
        macro, per_label, n = macro_auc_vs_gold(gold, blend)
        print(f"=== rank-mean blend of top {len(top)} ===")
        for name in top:
            print(f"  + {name}")
        print(f"  macro {macro:.4f} on {n} gold studies, {len(blend)} studies covered")
        print("  per label: "
              + "  ".join(f"{k}:{v:.3f}" for k, v in sorted(per_label.items())))
        if args.out:
            blend.to_csv(args.out, index=False)
            print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
