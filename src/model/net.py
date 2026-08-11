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

import os

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


class DinoEncoder(nn.Module):
    """A DINOv2 ViT wrapped to the same contract as a timm backbone.

    Maps (N, 3, H, W) -> (N, D) by taking the CLS token, so it drops straight
    into KneeNet in place of `timm.create_model(...)` and every downstream part
    (AttentionPool, the head) is unchanged.

    Why bother: the public 0.899 baseline is not a better-tuned version of this
    pipeline, it is a different one — a self-supervised ViT *adapted* with a
    near-frozen encoder, rather than an ImageNet CNN retrained end to end. Our
    EfficientNetV2-S recipe measures 0.7767. That gap is architectural, and it
    is the only difference identified so far that is plausibly worth ~0.1
    rather than the ~0.002 that preprocessing tweaks have been returning.

    `path` must be a local directory (Kaggle mounts the model at
    /kaggle/input/dinov2/pytorch/base/1). Submission notebooks run with the
    internet OFF, so downloading weights by name at runtime cannot work.
    """

    #: DINOv2 is patch-14. Any input side must be a multiple of 14 or the
    #: patch embedding silently drops a partial patch and position embeddings
    #: stop lining up. 224 = 14x16 and 336 = 14x24; **256 is not** (18.29), so
    #: the 256px config written for the CNN is invalid for this encoder.
    PATCH = 14

    def __init__(self, path: str, unfreeze_last: int = 4) -> None:
        super().__init__()
        from transformers import AutoModel

        self.vit = AutoModel.from_pretrained(path)
        self.num_features = int(self.vit.config.hidden_size)

        # Adapt, do not retrain. Full fine-tuning of a self-supervised ViT on
        # ~3.5k noisy-labelled studies destroys the pretrained features; the
        # baseline opens only the last few blocks and pairs that with a tiny
        # backbone LR. Everything else stays frozen, which also cuts the
        # activation memory a 15 GB T4 has to hold.
        for param in self.vit.parameters():
            param.requires_grad = False
        blocks = self.vit.encoder.layer
        for block in blocks[max(0, len(blocks) - unfreeze_last):]:
            for param in block.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self.PATCH or x.shape[-2] % self.PATCH:
            raise ValueError(
                f"DINOv2 is patch-{self.PATCH}: input side must be a multiple of "
                f"{self.PATCH}, got {tuple(x.shape[-2:])}. Use 224 or 336, not 256."
            )
        return self.vit(pixel_values=x).last_hidden_state[:, 0]  # CLS token


class KneeNet(nn.Module):
    """(B, 3, S, H, W) -> (B, 12) logits."""

    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s.in21k_ft_in1k",
        n_classes: int = 12,
        pretrained: bool = True,
        dropout: float = 0.3,
        unfreeze_last: int = 4,
    ) -> None:
        super().__init__()

        # A path rather than a timm name selects the DINOv2 encoder. Kept as one
        # argument so checkpoints, the submission notebook and the training
        # script all keep working without a second code path to drift.
        if "/" in backbone or os.path.isdir(backbone):
            self.backbone = DinoEncoder(backbone, unfreeze_last=unfreeze_last)
        else:
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
