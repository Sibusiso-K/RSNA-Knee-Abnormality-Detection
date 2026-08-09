"""2.5D CNN + attention-MIL model for the twelve knee findings.

Architecture follows the pattern that won RSNA 2022 (cervical spine), RSNA 2023
(abdominal trauma) and RSNA 2024 (lumbar spine) — three consecutive years, so
this is the field-tested default, not a guess:

    per-slice 2D ImageNet backbone -> sequence/attention pooling across slices
    -> multi-label head

Why not a 3D CNN: with report-derived (noisy) labels and a free-tier GPU budget,
initialisation matters more than capacity. A 2D ImageNet backbone starts from
vastly better weights than any 3D alternative, and it keeps us inside the 9-hour
inference cap that the Efficiency track rewards. See docs/04-method.md.

Three planes (sagittal/coronal/axial) share one backbone rather than getting one
each. Reason: with ~4,400 studies and noisy labels, three separate backbones
would triple parameters against the same supervision and overfit. A shared
backbone with per-plane attention pooling keeps plane-specific reasoning in the
cheap part of the network.
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_PLANES = 3


class AttentionPool(nn.Module):
    """Gated attention MIL pooling over the slice axis (Ilse et al., 2018).

    Mean-pooling would dilute a finding that occupies 3 slices out of 16 — the
    meniscal-tear case that docs/02-domain-primer.md flags as the hardest label.
    Attention lets the model concentrate on the slices that matter, and the
    weights are inspectable, which is how you catch "it's reading the wrong
    plane" without guessing.
    """

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid())
        self.score = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, S, D)
        weights = self.score(self.attn(x) * self.gate(x))  # (B, S, 1)
        weights = torch.softmax(weights, dim=1)
        return (x * weights).sum(dim=1), weights.squeeze(-1)


class KneeNet(nn.Module):
    """(B, 3, S, H, W) -> (B, 12) logits."""

    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s.in21k_ft_in1k",
        n_classes: int = 12,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, in_chans=3
        )
        dim = self.backbone.num_features

        # One pooling head per plane: a sagittal stack and an axial stack want
        # different slices attended to, and that is cheap to specialise.
        self.pools = nn.ModuleList([AttentionPool(dim) for _ in range(N_PLANES)])
        self.head = nn.Sequential(
            nn.LayerNorm(dim * N_PLANES),
            nn.Dropout(dropout),
            nn.Linear(dim * N_PLANES, n_classes),
        )

    @staticmethod
    def _to_25d(volume: torch.Tensor) -> torch.Tensor:
        """(B, S, H, W) -> (B*S, 3, H, W) using adjacent-slice triplets.

        Each slice is fed as (previous, current, next) in the RGB channels, so
        the backbone sees local through-plane context for free — the 2.5D trick
        every recent RSNA winner uses. Edges replicate rather than wrap: slice 0
        must not be given the far end of the knee as its "previous".
        """
        before = torch.cat([volume[:, :1], volume[:, :-1]], dim=1)
        after = torch.cat([volume[:, 1:], volume[:, -1:]], dim=1)
        stacked = torch.stack([before, volume, after], dim=2)  # (B, S, 3, H, W)
        b, s, c, h, w = stacked.shape
        return stacked.reshape(b * s, c, h, w)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # x: (B, planes, S, H, W)
        b, p, s, h, w = x.shape
        features, attentions = [], []
        for plane in range(p):
            flat = self._to_25d(x[:, plane])          # (B*S, 3, H, W)
            encoded = self.backbone(flat)              # (B*S, D)
            encoded = encoded.reshape(b, s, -1)        # (B, S, D)
            pooled, attn = self.pools[plane](encoded)
            features.append(pooled)
            attentions.append(attn)

        logits = self.head(torch.cat(features, dim=1))
        if return_attn:
            return logits, torch.stack(attentions, dim=1)
        return logits


def positive_weights(labels, cap: float = 10.0) -> torch.Tensor:
    """Per-label pos_weight for BCEWithLogitsLoss.

    The metric is *macro* AUC, so Fracture (6.9% prevalence) counts exactly as
    much as Effusion (54.9%). Without weighting, the loss is dominated by the
    common labels and the rare ones — precisely the ones dragging the macro
    score down — get undertrained.

    Capped because raw pos/neg ratios on the rarest labels produce enormous
    weights that destabilise early training more than they help.
    """
    import numpy as np

    arr = np.asarray(labels, dtype=np.float32)
    pos = arr.sum(axis=0)
    neg = len(arr) - pos
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return torch.tensor(np.clip(weights, 1.0, cap), dtype=torch.float32)
