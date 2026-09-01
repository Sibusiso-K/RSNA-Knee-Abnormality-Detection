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


class GroupAttnHead(nn.Module):
    """(B, S, G, T, D) tokens + (B, S) mask -> (B, 12) logits.

    **The fix for slice saturation.**

    `XAttnHead` restored spatial access within a triplet and paid: it is what
    the 0.864 submission runs. But the SLICE axis is still outside the model.
    Training draws one random group of three slices per step and inference
    averages the resulting logits:

        for g in range(N_GROUPS): acc += net(take_group(x, g), m)

    So a study with six slices per slot is two independent predictions averaged,
    not one prediction with twice the evidence. That is exactly the shape of
    what we measured — 3 -> 6 slices gained **+0.0236**, concentrated on focal
    findings, and 6 -> 12 gave **-0.005**. A fracture visible in one group gets
    averaged with a group where it is not, and diluted; doubling the groups
    doubles the dilution as fast as it adds evidence, so the curve flattens.

    Here every group is a token source in ONE attention. A query for Fracture
    attends across all S x G x T tokens at once, so it can concentrate on the
    slice group where the cortical break actually is and ignore the rest —
    selection instead of averaging. Nothing is thrown away and nothing is
    diluted, and the through-plane position becomes something the model can
    attend to rather than something the harness averages over.

    Cost is one attention over S*G*T keys: at 6 slots x 2 groups x 577 tokens
    that is 6,924 keys for 12 queries, ~83k pairs. Negligible next to the
    encoder, which does the same S*G passes either way.

    The queries also attend to EACH OTHER before reading the tokens. The twelve
    findings co-occur strongly — effusion with synovitis, medial with lateral
    OA — and a query that knows what the others found is better placed than
    twelve queries deciding in isolation. It is 12x12, so it is free.
    """

    def __init__(self, dim: int, pooled_dim: int, n_slot: int = N_SLOT,
                 n_out: int = len(TARGETS), hidden: int = 256, heads: int = 8,
                 dropout: float = 0.2, prior: bool = True,
                 max_groups: int = 8, depth: int = 2) -> None:
        super().__init__()
        self.n_out = n_out
        self.depth = depth
        self.token_proj = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU()
        )
        # Which slot AND which slice group a token came from. The encoder sees
        # neither - it processes every (slot, group) triplet independently - so
        # both are added here. `max_groups` is fixed so a model trained at two
        # groups can be scored at four without a shape error; unused rows just
        # never receive gradient.
        self.slot_emb = nn.Embedding(n_slot, hidden)
        self.group_emb = nn.Embedding(max_groups, hidden)
        self.max_groups = max_groups

        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.target_group_emb = nn.Embedding(len(GROUP_NAMES), hidden)
        self.register_buffer(
            "group_of_target", torch.tensor(GROUP_OF_TARGET, dtype=torch.long)
        )

        # Two rounds of (queries talk -> queries read -> queries think). One
        # round is a single static look; iterating lets the second look be
        # informed by what the first found, which is the whole argument for
        # attention over pooling.
        self.self_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
            for _ in range(depth)
        ])
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
            for _ in range(depth)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))
            for _ in range(depth)
        ])
        self.norm_self = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.norm_cross = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.norm_ffn = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])

        # The pooled path is kept, exactly as XAttnHead keeps it: it is what
        # the working submission is built on, and discarding a component that
        # works to test an addition confounds the measurement.
        self.pooled = SlotHead(pooled_dim, n_slot=n_slot, n_out=n_out,
                               hidden=hidden, dropout=dropout, prior=prior)

        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
            )
            for _ in range(n_out)
        ])

    def forward(self, tokens: torch.Tensor, pooled: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        b, s, g, t, _ = tokens.shape
        if g > self.max_groups:
            raise ValueError(
                f"{g} slice groups exceeds max_groups={self.max_groups}; the "
                f"group embedding has no row for the extra ones"
            )
        x = self.token_proj(tokens)                                # (B,S,G,T,H)
        x = x + self.slot_emb.weight.view(1, s, 1, 1, -1)
        x = x + self.group_emb.weight[:g].view(1, 1, g, 1, -1)
        x = x.reshape(b, s * g * t, -1)

        # An absent slot is absent in every one of its groups. Masking rather
        # than feeding zeros: a zero image is a black image, and the encoder
        # maps it to a confident feature like any other input.
        key_pad = (mask < 0.5).unsqueeze(-1).unsqueeze(-1)
        key_pad = key_pad.expand(b, s, g, t).reshape(b, s * g * t)

        q = self.query + self.target_group_emb(self.group_of_target)
        q = q.unsqueeze(0).expand(b, -1, -1)                        # (B,O,H)

        for i in range(self.depth):
            h = self.norm_self[i](q)
            q = q + self.self_attn[i](h, h, h, need_weights=False)[0]
            h = self.norm_cross[i](q)
            q = q + self.cross_attn[i](h, x, x, key_padding_mask=key_pad,
                                       need_weights=False)[0]
            q = q + self.ffn[i](self.norm_ffn[i](q))

        q = self.drop(q)
        spatial = torch.cat(
            [self.heads[i](q[:, i]) for i in range(self.n_out)], dim=1
        )
        return spatial + self.pooled(pooled, mask)


class SlotNet(nn.Module):
    """(B, S, C, H, W) uint8 + (B, S) mask -> (B, 12) logits.

    `C` is 3 for the `slot` and `xattn` heads — one 2.5D triplet per slot. For
    `gattn` it is `3 * G`: every slice group the cache holds, handed to the
    model together instead of sampled one at a time by the training loop.
    """

    #: DINOv2 is patch-14: the input side must be a multiple of 14 or the patch
    #: embedding drops a partial patch and the position embeddings stop lining
    #: up. 336 = 14x24 and 224 = 14x16; **256 is not** (18.29).
    PATCH = 14

    def __init__(self, source, unfreeze_last: int = 6, pool: str = "cls_mean_focal",
                 dropout: float = 0.2, prior: bool = True, head: str = "slot") -> None:
        super().__init__()
        from transformers import AutoModel

        # `head` defaults to "slot" so every existing checkpoint keeps loading
        # unchanged. The 0.850 submission is five of those, and an ensemble that
        # silently stops loading its members is worse than one that never gained
        # a new architecture.
        self.head_type = head
        if isinstance(source, str):
            self.vit = AutoModel.from_pretrained(source)
        else:
            # An already-loaded backbone, not a path — passed in to avoid
            # calling from_pretrained() again. That call reads the checkpoint
            # off Kaggle's mounted input storage, and a 5-fold training loop
            # that calls it once per fold hit the SAME transient stall, at
            # the SAME point (a new fold's model construction), four times
            # across two independent scripts and both accelerators (GPU and
            # TPU) in one session — see docs/12-handover.md. Deepcopy so each
            # fold still gets an independently-trainable module; sharing one
            # instance across folds would let one fold's gradient updates
            # corrupt the "fresh" initialisation the next fold expects.
            import copy
            self.vit = copy.deepcopy(source)
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
        if head == "gattn":
            self.head = GroupAttnHead(dim, pooled_dim=dim * parts,
                                      dropout=dropout, prior=prior)
        elif head == "xattn":
            # Tokens go in at the encoder's own width; the pooled path keeps the
            # concatenated width it has always had.
            self.head = XAttnHead(dim, pooled_dim=dim * parts,
                                  dropout=dropout, prior=prior)
        elif head == "slot":
            self.head = SlotHead(dim * parts, dropout=dropout, prior=prior)
        else:
            raise ValueError(
                f"unknown head {head!r}; use 'slot', 'xattn' or 'gattn'"
            )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs: torch.Tensor, mask: torch.Tensor,
                img_size: int | None = None) -> torch.Tensor:
        b, s = imgs.shape[:2]

        # The group axis is the model's business, not the training loop's.
        #
        # For `slot` and `xattn` the caller hands over one triplet per slot and
        # averages logits across groups outside. `gattn` exists precisely to
        # stop that averaging, so it takes every group at once: C = 3*G is split
        # here and folded into the batch, which means the encoder does the same
        # S*G passes either way and only the head sees a difference.
        groups = 1
        if self.head_type == "gattn":
            channels = imgs.shape[2]
            if channels % 3:
                raise ValueError(
                    f"gattn takes 3*G channels per slot, got {channels}"
                )
            groups = channels // 3
            imgs = imgs.reshape(b, s * groups, 3, *imgs.shape[3:])
        elif imgs.shape[2] != 3:
            raise ValueError(
                f"head {self.head_type!r} takes one 2.5D triplet per slot "
                f"(3 channels), got {imgs.shape[2]}. Sample a group first, or "
                f"use head='gattn' to hand the model all of them."
            )

        x = imgs.reshape(b * imgs.shape[1], *imgs.shape[2:]).float().div_(255.0)
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
        feat = torch.cat(parts, dim=1)
        n_token, width = out.shape[1], out.shape[2]

        if self.head_type == "gattn":
            tokens = out.reshape(b, s, groups, n_token, width)
            # The pooled path stays per-SLOT, so its groups are averaged here.
            # That is deliberate: averaging is the thing the token path exists
            # to avoid, and leaving the pooled baseline exactly as it was in the
            # working model keeps the comparison to XAttnHead honest. Only the
            # spatial path gets the new axis.
            pooled = feat.reshape(b, s, groups, -1).mean(2)
            return self.head(tokens, pooled, mask)

        feat = feat.reshape(b, s, -1)
        if self.head_type == "xattn":
            # Hand the head the token grid as well. This is the whole point:
            # `feat` has already thrown away where in the image anything was,
            # and that is what the four null experiments were fighting.
            tokens = out.reshape(b, s, n_token, width)
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
