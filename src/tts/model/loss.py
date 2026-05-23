from collections.abc import Callable
from enum import StrEnum
from functools import partial

import torch
import torch.nn.functional as F

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class LossFns(StrEnum):
    """Per-element loss functions, selectable from config.

    Each variant resolves to the corresponding ``F.*_loss`` with
    ``reduction="none"`` — the elementwise error tensor, same shape as the
    inputs. Reducing (over the feature dim, then the masked time axis) is the
    caller's job.
    """

    MSE = "mse"
    L1 = "l1"

    @property
    def fn(self) -> LossFn:
        match self:
            case LossFns.MSE:
                return partial(F.mse_loss, reduction="none")
            case LossFns.L1:
                return partial(F.l1_loss, reduction="none")
