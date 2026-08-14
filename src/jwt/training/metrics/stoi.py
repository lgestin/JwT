import torch
import torch.nn as nn
from torchmetrics.functional.audio import (
    short_time_objective_intelligibility,
)

from jwt.training.metrics.metric import ComparativeMetric


class STOI(nn.Module, ComparativeMetric):
    @torch.inference_mode()
    def score(
        self,
        pred: torch.Tensor,
        trgt: torch.Tensor,
        mask: torch.Tensor | None = None,
        sample_rate: int = 16_000,
    ) -> dict[str, torch.Tensor]:
        if mask is not None:
            raise NotImplementedError(
                "masked scoring not supported; pass full-length waveforms"
            )
        stoi = short_time_objective_intelligibility(
            preds=pred,
            target=trgt,
            fs=sample_rate,
            extended=True,
        )
        return {"stoi": torch.as_tensor(stoi)}
