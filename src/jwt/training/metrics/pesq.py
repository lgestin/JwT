import torch
import torch.nn as nn
from torchmetrics.functional.audio import (
    perceptual_evaluation_speech_quality,
)

from jwt.training.metrics.metric import ComparativeMetric
from jwt.training.metrics.utils import _to_16khz_mono


class PESQ(nn.Module, ComparativeMetric):
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
        pred = _to_16khz_mono(pred.detach(), sample_rate)
        trgt = _to_16khz_mono(trgt.detach(), sample_rate)
        pesq = perceptual_evaluation_speech_quality(
            preds=pred,
            target=trgt,
            fs=16_000,
            mode="wb",
        )
        return {"pesq": pesq}
