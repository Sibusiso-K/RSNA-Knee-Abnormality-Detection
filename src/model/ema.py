"""Exponential moving average of model weights (docs/20 job 2, the remaining
untested candidate from docs/15).

Deliberately minimal: one predeclared decay, no schedule, no warmup ramp. The
question this pilot answers is whether EMA helps at all on this architecture
and data, not what the best decay is - a decay search would be exactly the
kind of after-the-fact tuning docs/15's acceptance protocol rules out.
"""
from __future__ import annotations

import torch


class EMA:
    """Shadow copy of a model's float parameters and buffers, updated by

        shadow <- decay * shadow + (1 - decay) * current

    after each real optimizer step.

    Construction copies the model's CURRENT weights as the shadow's starting
    point ("initialize it from the current model at the declared start") -
    call `EMA(model, decay)` once, right after the model is built and before
    any training step, not partway through training. Non-float buffers
    (e.g. an integer running-count) are copied verbatim on every update
    rather than averaged, since a fractional buffer of that kind is
    meaningless.

    Does not read or write the source model's `.grad` or optimizer state,
    and every update is wrapped in `torch.no_grad()`, so it cannot affect
    backprop through the ORDINARY model. It also never mutates the ordinary
    model's tensors: `state_dict()` values are cloned into new leaf tensors
    at construction, and `update()` only ever writes into those shadow
    tensors, never into the source model's own parameters/buffers.

    XLA note: `update()` uses only elementwise in-place ops (`mul_`, `add_`,
    `copy_`) on already-materialised tensors, the same primitives the
    training loop already relies on before its own `xm.mark_step()`; no
    control flow depends on a `.item()`/host sync, so it traces the same way
    on CPU/CUDA and XLA.
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.num_updates = 0
        with torch.no_grad():
            self.shadow = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Call once per REAL optimizer step (not per micro-batch, if the
        caller ever accumulates gradients) with the model as it stands right
        after that step."""
        current = model.state_dict()
        if current.keys() != self.shadow.keys():
            raise ValueError("EMA shadow and model state_dict keys differ")
        self.num_updates += 1
        for k, v in current.items():
            s = self.shadow[k]
            if torch.is_floating_point(s):
                # s <- decay*s + (1-decay)*v, in place, without ever
                # constructing a tensor that aliases the source model's own
                # storage.
                s.mul_(self.decay).add_(v.to(s.device, s.dtype), alpha=1.0 - self.decay)
            else:
                s.copy_(v)

    def state_dict(self) -> dict:
        """Shadow weights on CPU, ready to `torch.save`.

        Always a fresh copy: `.cpu()` is a no-op (returns the SAME tensor,
        not a copy) when the source is already on CPU, so without the
        explicit `.clone()` a caller who saved this dict and then called
        `update()` again would see the "saved" tensors change under them.
        """
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}

    def to(self, device) -> "EMA":
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self
