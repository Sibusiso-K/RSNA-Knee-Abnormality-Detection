"""Tests for the imaging pipeline.

These exist because the alternative is discovering a shape bug on a Kaggle GPU
notebook 40 minutes into a run. `timm` is stubbed so the wiring can be checked
without downloading ImageNet weights — the real backbone only changes the
feature width, which the model reads from `num_features` anyway.

The GroupKFold leakage test is the important one in this file. Site leakage
inflates CV by ~0.053 macro AUC (docs/04-method.md); if that test ever goes
green-to-red unnoticed, every score the project reports becomes fiction.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _stub_timm() -> None:
    if "timm" in sys.modules and not hasattr(sys.modules["timm"], "_is_stub"):
        return  # real timm present (Kaggle) — use it
    stub = types.ModuleType("timm")

    class _Backbone(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.num_features = 32
            self.conv = nn.Conv2d(3, 32, 3, padding=1)

        def forward(self, x):
            return self.conv(x).mean((2, 3))

    stub.create_model = lambda *a, **k: _Backbone()
    stub._is_stub = True
    sys.modules["timm"] = stub


_stub_timm()

from src.model.net import AttentionPool, KneeNet, positive_weights  # noqa: E402
from src.model.validation import grouped_folds, macro_auc  # noqa: E402


# -- 2.5D construction ---------------------------------------------------

def test_25d_shape_and_centre_channel():
    volume = torch.arange(2 * 4 * 3 * 3, dtype=torch.float32).reshape(2, 4, 3, 3)
    out = KneeNet._to_25d(volume)
    assert out.shape == (8, 3, 3, 3)
    assert torch.equal(out.reshape(2, 4, 3, 3, 3)[:, :, 1], volume)


def test_25d_replicates_at_edges_rather_than_wrapping():
    """Slice 0's "previous" must be slice 0, not the far end of the knee."""
    volume = torch.arange(1 * 4 * 2 * 2, dtype=torch.float32).reshape(1, 4, 2, 2)
    r = KneeNet._to_25d(volume).reshape(1, 4, 3, 2, 2)
    assert torch.equal(r[:, 0, 0], r[:, 0, 1])      # first: prev == self
    assert torch.equal(r[:, -1, 2], r[:, -1, 1])    # last: next == self


# -- attention pooling ---------------------------------------------------

def test_attention_pool_shapes_and_normalisation():
    pooled, weights = AttentionPool(16)(torch.randn(2, 5, 16))
    assert pooled.shape == (2, 16)
    assert weights.shape == (2, 5)
    assert torch.allclose(weights.sum(1), torch.ones(2), atol=1e-5)


# -- full model ----------------------------------------------------------

def test_forward_returns_twelve_logits():
    model = KneeNet(pretrained=False)
    out = model(torch.randn(2, 3, 8, 64, 64))
    assert out.shape == (2, 12)


def test_forward_attention_covers_every_plane_and_slice():
    model = KneeNet(pretrained=False)
    _, attn = model(torch.randn(2, 3, 8, 64, 64), return_attn=True)
    assert attn.shape == (2, 3, 8)


# -- class imbalance -----------------------------------------------------

def test_positive_weights_favour_rare_labels_and_stay_capped():
    labels = np.zeros((100, 12))
    labels[:50, 7] = 1   # common, like Effusion
    labels[:7, 11] = 1   # rare, like Fracture
    weights = positive_weights(labels, cap=10.0)
    assert weights[11] > weights[7]
    assert float(weights.max()) <= 10.0


def test_positive_weights_handle_all_zero_label():
    """A fold can contain zero positives for a rare label — must not divide by
    zero or emit inf, which would poison the loss."""
    weights = positive_weights(np.zeros((20, 12)))
    assert torch.isfinite(weights).all()


# -- validation ----------------------------------------------------------

def test_grouped_folds_never_split_a_group():
    groups = np.array(["A"] * 30 + ["B"] * 30 + ["C"] * 20 + ["D"] * 20)
    folds = grouped_folds(groups, n_splits=4)
    for group in set(groups):
        assert len(set(folds[groups == group])) == 1, f"{group} leaked across folds"


def test_grouped_folds_assign_every_row():
    groups = np.array([f"s{i % 9}" for i in range(60)])
    assert (grouped_folds(groups, n_splits=3) >= 0).all()


def test_macro_auc_skips_single_class_labels():
    truth = np.zeros((40, 12))
    truth[:20, 0] = 1                      # only label 0 has both classes
    pred = np.random.rand(40, 12)
    pred[:20, 0] += 1.0
    macro, per_label = macro_auc(truth, pred)
    assert list(per_label) == ["ACL"]
    assert macro == pytest.approx(1.0)


# -- regressions from the first real Kaggle training run -----------------

def test_macro_auc_accepts_soft_labels():
    """Training targets are soft (0.6 for a hedged finding). sklearn rejects
    continuous y_true outright -- this crashed fold 0 after a full epoch."""
    truth = np.zeros((40, 12))
    truth[:20, 0] = 0.92          # affirmed-in-findings score, not 1.0
    truth[20:25, 0] = 0.6         # hedged
    pred = np.random.rand(40, 12)
    pred[:20, 0] += 1.0
    macro, per_label = macro_auc(truth, pred)
    assert "ACL" in per_label
    assert 0.0 <= macro <= 1.0


def test_macro_auc_threshold_splits_hedged_labels():
    truth = np.array([[0.6] * 12, [0.0] * 12, [0.6] * 12, [0.0] * 12])
    pred = np.array([[0.9] * 12, [0.1] * 12, [0.8] * 12, [0.2] * 12])
    _, per_low = macro_auc(truth, pred, threshold=0.5)
    assert per_low["ACL"] == pytest.approx(1.0)   # 0.6 counts as positive
    _, per_high = macro_auc(truth, pred, threshold=0.7)
    assert per_high == {}                          # 0.6 now negative -> one class


def test_check_grouping_flags_near_unique_key(capsys):
    """The ImagingFrequency bug: 3,229 groups over 4,349 studies made
    GroupKFold equivalent to random KFold while still looking rigorous."""
    from src.model.validation import check_grouping

    stats = check_grouping([f"g{i}" for i in range(100)])
    assert stats["ratio"] == 1.0
    assert "WARNING" in capsys.readouterr().out


def test_check_grouping_quiet_for_coarse_key(capsys):
    from src.model.validation import check_grouping

    stats = check_grouping([f"s{i % 5}" for i in range(100)])
    assert stats["n_groups"] == 5
    assert "WARNING" not in capsys.readouterr().out


def test_clean_tag_collapses_imaging_frequency_drift():
    """Same magnet, different sessions -> same group."""
    from src.model.validation import _clean_tag

    a = _clean_tag("ImagingFrequency", "63.8721")
    b = _clean_tag("ImagingFrequency", "63.8934")
    c = _clean_tag("ImagingFrequency", "127.7312")
    assert a == b            # session drift collapsed
    assert a != c            # 1.5T vs 3T still distinguished
    assert _clean_tag("ImagingFrequency", None) == ""
    assert _clean_tag("ImagingFrequency", "n/a") == ""


def test_clean_tag_handles_multivalued_software_versions():
    from src.model.validation import _clean_tag

    assert _clean_tag("SoftwareVersions", ["syngo MR", "E11"]) == "syngo MR,E11"
