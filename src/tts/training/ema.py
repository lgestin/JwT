"""Exponential moving average of model weights for the TTS trainer."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch.nn import Module


@dataclass
class EMAConfig:
    # EMA almost always improves flow-matching sample quality at near-zero
    # training cost, so it is on by default.
    enabled: bool = True
    decay: float = 0.9995


class EMA:
    """Exponential moving average of a model's trainable parameters.

    Holds a shadow fp32 copy of every parameter with ``requires_grad`` and
    nudges it toward the live weights after each optimizer step. A decay-warmup
    keeps early-init weights from being locked in. Evaluation borrows the
    shadow weights via the ``swapped`` context manager, which copies in place
    so a compiled forward graph is left untouched.
    """

    def __init__(self, model: Module, decay: float = 0.9995):
        self.decay = decay
        self._shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone().float()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def _effective_decay(self, step: int) -> float:
        """Decay ramped up over early steps so init weights aren't locked in."""
        return min(self.decay, (1 + step) / (10 + step))

    @torch.no_grad()
    def update(self, model: Module, step: int) -> None:
        """Nudge every shadow weight toward the live weight after an opt step."""
        decay = self._effective_decay(step)
        params = dict(model.named_parameters())
        shadow = list(self._shadow.values())
        live = [params[name].detach().float() for name in self._shadow]
        torch._foreach_lerp_(shadow, live, 1.0 - decay)

    @contextmanager
    def swapped(self, model: Module) -> Iterator[None]:
        """Install the EMA weights into ``model`` for the duration of the block.

        The swap is an in-place ``copy_``, so parameter identities are
        preserved and a compiled forward graph does not recompile. The
        original weights are restored on exit, including on exception.
        """
        params = dict(model.named_parameters())
        backup: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, shadow in self._shadow.items():
                param = params[name]
                backup[name] = param.detach().clone()
                param.copy_(shadow)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, original in backup.items():
                    params[name].copy_(original)

    def state_dict(self) -> dict[str, Any]:
        """Serializable EMA state for checkpointing."""
        return {"decay": self.decay, "shadow": self._shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore EMA state saved by ``state_dict``, in place."""
        self.decay = state["decay"]
        loaded = state["shadow"]
        with torch.no_grad():
            for name, tensor in self._shadow.items():
                tensor.copy_(loaded[name])
