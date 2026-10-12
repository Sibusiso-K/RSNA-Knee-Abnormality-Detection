"""Offline check of ONE predeclared candidate from saved TTA predictions.

Candidate (fixed before computing, written here first):
    0.25 * TTA(S) + 0.25 * TTA(E24) + 0.5 * TTA(B)
i.e. average the two independently trained five-fold small/xattn runs
(deployed S and the snapshot run's epoch-24 checkpoint), keep the 50/50
small/base family split. Anchor: 0.5 * TTA(S) + 0.5 * TTA(B) (the 0.893
submission's per-fold formula, mean-fold OOF 0.8599910527).

Post hoc relative to docs/21: these folds and both checkpoints were already
inspected there, and that branch's two predeclared candidates were rejected.
This is a different blend (adds E24 alongside S instead of replacing it), but
the evidence is exploratory, not a fresh validation.

Usage: python scripts/eval_small_seed_average.py
"""
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.labels import TARGETS  # noqa: E402

UNDECIDED = 0.05
FOLDS = range(5)
N_BOOT = 1000
EXPECTED_ANCHOR_MEAN = 0.8599910526581681


def macro_auc(y_true, y_pred):
    """Same rule as kaggle_06's macro_auc(drop_undecided=True)."""
    per = {}
    for j, t in enumerate(TARGETS):
        y, p = y_true[:, j], y_pred[:, j]
        keep = ~np.isnan(y) & (np.abs(y - 0.5) >= UNDECIDED)
        yb = (y[keep] > 0.5).astype(int)
        if keep.sum() < 10 or len(set(yb)) < 2:
            continue
        per[t] = float(roc_auc_score(yb, p[keep]))
    return float(np.mean(list(per.values()))), per


def main():
    d = np.load(REPO / "kaggle/evaluate-snap-tta/results/predictions.npz", allow_pickle=True)
    y = {f: d[f"target_fold{f}"] for f in FOLDS}
    g = {f: d[f"groups_fold{f}"] for f in FOLDS}
    S = {f: d[f"S_tta_prob_fold{f}"] for f in FOLDS}
    E24 = {f: d[f"E24_tta_prob_fold{f}"] for f in FOLDS}
    B = {f: d[f"B_tta_prob_fold{f}"] for f in FOLDS}

    anchor = {f: 0.5 * S[f] + 0.5 * B[f] for f in FOLDS}
    cand = {f: 0.25 * S[f] + 0.25 * E24[f] + 0.5 * B[f] for f in FOLDS}

    a_sc, a_pt, c_sc, c_pt = {}, {}, {}, {}
    for f in FOLDS:
        a_sc[f], a_pt[f] = macro_auc(y[f], anchor[f])
        c_sc[f], c_pt[f] = macro_auc(y[f], cand[f])
    mean_anchor = float(np.mean(list(a_sc.values())))
    assert abs(mean_anchor - EXPECTED_ANCHOR_MEAN) < 1e-9, mean_anchor

    rng = np.random.default_rng(0)
    rows = {f: [np.flatnonzero(g[f] == k) for k in np.unique(g[f])] for f in FOLDS}
    boot = {f: [] for f in FOLDS}
    for _ in range(N_BOOT):
        for f in FOLDS:
            pick = rng.integers(0, len(rows[f]), len(rows[f]))
            idx = np.concatenate([rows[f][i] for i in pick])
            sa, _ = macro_auc(y[f][idx], anchor[f][idx])
            sb, _ = macro_auc(y[f][idx], cand[f][idx])
            boot[f].append(sb - sa)
    mean_boot = np.mean([boot[f] for f in FOLDS], axis=0)

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    deltas = {f: c_sc[f] - a_sc[f] for f in FOLDS}
    out = {
        "candidate": "0.25*S + 0.25*E24 + 0.5*B",
        "anchor": "0.5*S + 0.5*B",
        "mean_anchor": mean_anchor,
        "mean_candidate": float(np.mean(list(c_sc.values()))),
        "mean_delta": float(np.mean(list(deltas.values()))),
        "mean_delta_ci95": ci(mean_boot),
        "boot_frac_positive": float((mean_boot > 0).mean()),
        "folds_nonnegative": int(sum(v >= 0 for v in deltas.values())),
        "per_fold": {f: {"anchor": a_sc[f], "candidate": c_sc[f], "delta": deltas[f],
                         "delta_ci95": ci(boot[f])} for f in FOLDS},
        "per_target_delta_mean": {
            t: float(np.mean([c_pt[f][t] - a_pt[f][t] for f in FOLDS]))
            for t in a_pt[0]},
        "n_boot": N_BOOT,
        "status": "post hoc exploratory (folds and checkpoints inspected in docs/21)",
    }
    dest = REPO / "kaggle/evaluate-snap-tta/results/small_seed_average.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_target_delta_mean"}, indent=2))
    print("per-target mean delta:", {k: round(v, 4) for k, v in out["per_target_delta_mean"].items()})


if __name__ == "__main__":
    main()
