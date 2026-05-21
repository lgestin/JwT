"""Optimizer config, builder, and LR warmup for the TTS trainer."""

from dataclasses import dataclass

from torch.nn import Module
from torch.optim import AdamW, Optimizer


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-2  # AdamW default
    betas: tuple[float, float] = (0.9, 0.999)
    # Linear LR warmup over this many steps, then constant. 0 disables it.
    warmup_steps: int = 0


def build_optimizer(model: Module, config: OptimizerConfig) -> AdamW:
    """Build an AdamW optimizer from `config`."""
    return AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )


class LinearWarmup:
    """Linear LR warmup to the base LR, then hold constant.

    A pure function of the training step — no scheduler object, no state to
    checkpoint, so `--resume` rejoins the ramp at the right point for free.
    The base LR is captured per param group at construction.
    """

    def __init__(self, optimizer: Optimizer, warmup_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self._base_lrs = [group["lr"] for group in optimizer.param_groups]

    def apply(self, step: int) -> float:
        """Set the LR for `step` on every param group; return the LR in effect."""
        if self.warmup_steps <= 0:
            return self.optimizer.param_groups[0]["lr"]
        scale = min(1.0, (step + 1) / self.warmup_steps)
        for group, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            group["lr"] = base_lr * scale
        return self.optimizer.param_groups[0]["lr"]
