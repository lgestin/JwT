"""Optimizer config and parameter grouping for the TTS trainer."""

from dataclasses import dataclass
from enum import StrEnum

import torch.nn as nn
from torch.optim import AdamW, Optimizer

from jwt.training.muon import NorMuonScheduleFree


class Optimizers(StrEnum):
    ADAMW = "adamw"
    SF_NORMUON = "sf_normuon"


@dataclass
class OptimizerConfig:
    name: Optimizers = Optimizers.SF_NORMUON
    # Shared: the whole model under ADAMW, the Adam group under SF_NORMUON.
    lr: float = 3e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 1000  # linear LR warmup, applied by the trainer
    # SF_NORMUON only: the hidden-matrix group.
    muon_lr: float = 5e-3
    muon_momentum: float = 0.8
    muon_weight_decay: float = 0.0

    def build(self, model: nn.Module, head: nn.Module | None = None) -> Optimizer:
        """Build the optimizer over `model`; groups carry `initial_lr`.

        Under SF_NORMUON every Linear weight takes the spectral step except the
        output `head`; embeddings, gains and biases stay on Adam (Muon papers).
        """
        if self.name is Optimizers.ADAMW:
            opt: Optimizer = AdamW(
                model.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        else:
            spectral = {
                id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)
            }
            if head is not None:
                spectral -= {id(p) for p in head.parameters()}
            hidden = [p for p in model.parameters() if id(p) in spectral]
            rest = [p for p in model.parameters() if id(p) not in spectral]
            opt = NorMuonScheduleFree(
                [
                    dict(
                        params=hidden,
                        use_muon=True,
                        lr=self.muon_lr,
                        betas=self.betas,
                        momentum=self.muon_momentum,
                        weight_decay=self.muon_weight_decay,
                    ),
                    dict(
                        params=rest,
                        use_muon=False,
                        lr=self.lr,
                        betas=self.betas,
                        weight_decay=self.weight_decay,
                    ),
                ]
            )
        for group in opt.param_groups:
            group.setdefault("initial_lr", group["lr"])
        return opt


def warmup_scale(step: int, warmup_steps: int) -> float:
    """LR multiplier at `step`: linear ramp to 1 over `warmup_steps`."""
    return min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
