"""DINOv2 encoder + per-diagnosis attention over six acquisition slots.

Why this replaces `net.py`'s KneeNet
------------------------------------
KneeNet retrains an ImageNet CNN end to end over three planes x sixteen slices.
Two things are wrong with that here, and neither is a tuning problem:

- **48 encoder passes per study.** Three planes x sixteen slices. The slot
  representation needs six, one per acquisition, because the slice axis is
  folded into the RGB channels as a 2.5D triplet. That is an ~8x cut in encoder
  work for the same study, which is what makes a ViT affordable on free-tier
  hardware at all.
- **The initialisation.** `docs/00-state.md` records that eight epochs did not
  beat four and that loss fell 61% while AUC declined — memorisation, on ~3.5k
  studies with noisy labels. A self-supervised ViT *adapted* with a near-frozen
  encoder starts from features that already separate tissue, so the supervision
  is spent on the twelve decisions rather than on relearning edges.

The head is deliberately small. With a study-level label there is nothing
telling the model which slice inside a slot carries the evidence, so parameters
below the slot level have nothing to learn from and would fit noise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.slots import SLOTS
from src.labels import TARGETS

N_SLOT = len(SLOTS)

#: Which slots each diagnosis is read on, as indices into `SLOTS`:
#:   0 SAG_FLUID_FS  1 COR_FLUID_FS  2 AX_FLUID_FS
#:   3 SAG_FLUID_NOFS  4 COR_T1  5 SAG_T1
#:
#: Straight from the anatomy in docs/02-domain-primer.md: cruciates are read
#: sagittally, the collateral ligaments and the tibiofemoral compartments
#: coronally, patellar cartilage axially, and the fluid-sensitive sequences
#: carry effusion, contusion, Baker's cyst and marrow oedema.
#:
#: This is a fixed tilt on the attention logits, not a mask and not a learned
#: parameter. Nothing is excluded — a preferred slot gets exp(0.55) ~ 1.73x the
#: weight of an unpreferred one, so the model can still overrule the anatomy
#: where the data says otherwise. It is a prior because 4,349 noisily labelled
#: studies are not enough to rediscover which sequence shows a Baker's cyst.
SLOT_PRIOR_TABLE: dict[str, tuple[int, ...]] = {
    "ACL": (0, 3, 5),
    "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4),
    "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5),
    "Lateral OA": (1, 4, 5),
    "PF OA": (0, 2, 5),
    "Effusion": (0, 2),
    "Synovitis": (0, 2),
    "Baker's": (0,),
    "Contusion": (0, 1, 2),
    "Fracture": (0, 1, 2, 4, 5),
}
SLOT_PRIOR_STRENGTH = 0.55


class SlotHead(nn.Module):
    """(B, S, D) + presence mask -> (B, n_out) logits.

    One attention distribution per diagnosis over the slots. Pooling the slots
    identically would dilute the one acquisition carrying the evidence with five
    that do not — an ACL tear is a sagittal finding, and averaging it with the
    axial stack is most of the signal thrown away.
    """

    def __init__(self, dim: int, n_slot: int = N_SLOT, n_out: int = len(TARGETS),
                 hidden: int = 256, dropout: float = 0.2, prior: bool = True) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        self.use_prior = prior

        tilt = torch.zeros(n_out, n_slot)
        if prior:
            for target, slots in SLOT_PRIOR_TABLE.items():
                if target in TARGETS:
                    tilt[TARGETS.index(target), list(slots)] = SLOT_PRIOR_STRENGTH
        # Registered as a buffer so it travels in the state dict: a checkpoint
        # trained with the tilt and loaded without it is silently a different
        # model that still loads cleanly.
        self.register_buffer("slot_prior", tilt)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.slot_emb                      # (B, S, H)
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        if self.use_prior:
            att = att + self.slot_prior.unsqueeze(0)
        # Absent slots are masked out of the softmax rather than fed as zeros.
        # A zero image is not "no information", it is a black image, and the
        # encoder maps it to a perfectly confident feature vector.
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))  # (B, O, H)
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


#: Findings that share anatomy, and therefore share evidence. Grouping lets the
#: twelve target queries start from a shared prior per family instead of twelve
#: unrelated random vectors: a medial and a lateral meniscal tear are read on the
#: same sequences with the same appearance, and 4,349 noisily-labelled studies is
#: not much supervision for each query to discover that alone.
TARGET_GROUPS: dict[str, tuple[str, ...]] = {
    "ligament": ("ACL", "MCL"),
    "meniscus": ("Medial Meniscus", "Lateral Meniscus"),
    "oa": ("Medial OA", "Lateral OA", "PF OA"),
    "inflammation": ("Effusion", "Synovitis", "Baker's"),
    "bone": ("Contusion", "Fracture"),
}
GROUP_NAMES = tuple(TARGET_GROUPS)
GROUP_OF_TARGET = [
    next(i for i, g in enumerate(GROUP_NAMES) if t in TARGET_GROUPS[g])
    for t in TARGETS
]


class XAttnHead(nn.Module):
    """(B, S, T, D) tokens + (B, S) mask -> (B, 12) logits.

    **This is the fix for the bottleneck that made four experiments null.**

    `SlotHead` pools each slot to a single vector before the head runs, so twelve
    target queries attend over six numbers. Everything spatial is gone by then —
    which is why a bigger encoder (base, -0.0005) and finer pixels (448, -0.0014)
    could not help, and why the focal findings score worst while the diffuse ones
    are fine:

        Baker's .845  Medial OA .832  Effusion .831   (diffuse — survive pooling)
        Lat Meniscus .754  MCL .747  PF OA .711       (focal — destroyed by it)

    Here the queries cross-attend the **patch tokens themselves**, so a query for
    PF OA can look at the patches over the patella rather than at a study-level
    average.

    Deliberately NOT a full copy of the published RTAHMIL design. That one also
    runs self-attention across the whole token sequence, which is O(n^2) over
    6 x 577 = 3,462 tokens — ~12M pairs per study, expensive and untested here.
    Cross-attention from twelve queries is 12 x 3,462, which is trivial. Restore
    spatial access first, measure, and only then buy the quadratic part.

    The pooled path is kept and concatenated rather than replaced: it is what
    the current 0.850 submission is built on, and discarding a working component
    to test an addition confounds the measurement.
    """

    def __init__(self, dim: int, pooled_dim: int, n_slot: int = N_SLOT,
                 n_out: int = len(TARGETS), hidden: int = 256, heads: int = 8,
                 dropout: float = 0.2, prior: bool = True) -> None:
        """`dim` is the encoder width (tokens); `pooled_dim` is dim * pool parts.

        Two widths, because the two paths see different things: the tokens
        arrive at the encoder's own width, while the pooled vector is the
        concatenation of CLS, mean and the focal tail. Passing one where the
        other belongs raises immediately in LayerNorm rather than training
        something subtly wrong — which is how this was caught.
        """
        super().__init__()
        self.n_out = n_out
        self.token_proj = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU()
        )
        # Which slot a token came from is information the encoder never sees —
        # it processes each slot independently — so it is added here.
        self.slot_emb = nn.Embedding(n_slot, hidden)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.group_emb = nn.Embedding(len(GROUP_NAMES), hidden)
        self.register_buffer(
            "group_of_target", torch.tensor(GROUP_OF_TARGET, dtype=torch.long)
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(hidden)

        # The pooled path, unchanged from SlotHead, so the two can be compared
        # and so nothing that already works is thrown away.
        self.pooled = SlotHead(pooled_dim, n_slot=n_slot, n_out=n_out,
                               hidden=hidden, dropout=dropout, prior=prior)

        self.drop = nn.Dropout(dropout)
        # One small MLP per target rather than a shared linear layer. The
        # findings do not share a decision boundary, and this is cheap.
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
            )
            for _ in range(n_out)
        ])

    def forward(self, tokens: torch.Tensor, pooled: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        b, s, t, _ = tokens.shape
        x = self.token_proj(tokens)                                # (B,S,T,H)
        x = x + self.slot_emb.weight.view(1, s, 1, -1)
        x = x.reshape(b, s * t, -1)                                # (B,S*T,H)

        # Absent slots are masked out of the attention, never fed as zeros: a
        # zero image is a black image, and the encoder maps it to a confident
        # feature like any other input.
        key_pad = (mask < 0.5).unsqueeze(-1).expand(b, s, t).reshape(b, s * t)

        q = self.query + self.group_emb(self.group_of_target)      # (O,H)
        q = self.norm_q(q).unsqueeze(0).expand(b, -1, -1)          # (B,O,H)
        ctx, _ = self.cross_attn(q, x, x, key_padding_mask=key_pad,
                                 need_weights=False)
        ctx = self.drop(ctx)

        # Per-target logits from the cross-attended context, plus the pooled
        # path's own logit for that target.
        spatial = torch.cat(
            [self.heads[i](ctx[:, i]) for i in range(self.n_out)], dim=1
        )
        return spatial + self.pooled(pooled, mask)


class SlotNet(nn.Module):
    """(B, S, 3, H, W) uint8 + (B, S) mask -> (B, 12) logits."""

    #: DINOv2 is patch-14: the input side must be a multiple of 14 or the patch
    #: embedding drops a partial patch and the position embeddings stop lining
    #: up. 336 = 14x24 and 224 = 14x16; **256 is not** (18.29).
    PATCH = 14

    def __init__(self, source: str, unfreeze_last: int = 6, pool: str = "cls_mean_focal",
                 dropout: float = 0.2, prior: bool = True, head: str = "slot") -> None:
        super().__init__()
        from transformers import AutoModel

        # `head` defaults to "slot" so every existing checkpoint keeps loading
        # unchanged. The 0.850 submission is five of those, and an ensemble that
        # silently stops loading its members is worse than one that never gained
        # a new architecture.
        self.head_type = head
        self.vit = AutoModel.from_pretrained(source)
        self.pool = pool
        parts = {"cls_mean": 2, "cls_mean_focal": 3}[pool]

        # Adapt, do not retrain. Full fine-tuning of a self-supervised ViT on
        # this much noisy supervision destroys the pretrained features; opening
        # the last few blocks also cuts the activation memory a 15 GB T4 holds.
        for param in self.vit.parameters():
            param.requires_grad = False
        blocks = self.vit.encoder.layer
        for block in blocks[max(0, len(blocks) - unfreeze_last):]:
            for param in block.parameters():
                param.requires_grad = True
        # The final norm is opened too: it is 2*D parameters and it is what
        # rescales features for a head that did not exist during pretraining.
        for param in self.vit.layernorm.parameters():
            param.requires_grad = True

        dim = int(self.vit.config.hidden_size)
        if head == "xattn":
            # Tokens go in at the encoder's own width; the pooled path keeps the
            # concatenated width it has always had.
            self.head = XAttnHead(dim, pooled_dim=dim * parts,
                                  dropout=dropout, prior=prior)
        elif head == "slot":
            self.head = SlotHead(dim * parts, dropout=dropout, prior=prior)
        else:
            raise ValueError(f"unknown head {head!r}; use 'slot' or 'xattn'")
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs: torch.Tensor, mask: torch.Tensor,
                img_size: int | None = None) -> torch.Tensor:
        b, s = imgs.shape[:2]
        x = imgs.reshape(b * s, *imgs.shape[2:]).float().div_(255.0)
        if img_size is not None and img_size != x.shape[-1]:
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear",
                              align_corners=False)
        if x.shape[-1] % self.PATCH or x.shape[-2] % self.PATCH:
            raise ValueError(
                f"DINOv2 is patch-{self.PATCH}: side must be a multiple of "
                f"{self.PATCH}, got {tuple(x.shape[-2:])}. Use 224 or 336, not 256."
            )
        x = (x - self.mean) / self.std

        out = self.vit(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        parts = [out[:, 0], patch.mean(1)]
        if self.pool == "cls_mean_focal":
            # The upper tail of each channel across the patch grid, taken per
            # channel rather than by picking whole patches. A finding occupies a
            # small part of the field, so a plain mean over 576 patches dilutes
            # it by orders of magnitude; this keeps the top eighth of each
            # channel's responses, which is where a small bright lesion lives.
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        feat = torch.cat(parts, dim=1).reshape(b, s, -1)

        if self.head_type == "xattn":
            # Hand the head the token grid as well. This is the whole point:
            # `feat` has already thrown away where in the image anything was,
            # and that is what the four null experiments were fighting.
            tokens = out.reshape(b, s, out.shape[1], out.shape[2])
            return self.head(tokens, feat, mask)
        return self.head(feat, mask)

    def param_groups(self, lr_head: float, lr_backbone: float):
        """Discriminative learning rates, as two optimiser groups.

        A single rate across a self-supervised ViT defeats the entire approach:
        at 1e-3 the encoder is retrained rather than adapted, and the features
        that justify using it are gone within an epoch.

        **OneCycleLR must then be given `max_lr` as a per-group list.** With a
        scalar it overwrites every group's rate and silently restores the bug
        this method exists to avoid — the run completes, and is worthless.
        """
        backbone = [p for p in self.vit.parameters() if p.requires_grad]
        return [
            {"params": backbone, "lr": lr_backbone},
            {"params": list(self.head.parameters()), "lr": lr_head},
        ]
