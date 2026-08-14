import torch
import torch.nn as nn
from torchmetrics.functional.audio.nisqa import (
    non_intrusive_speech_quality_assessment,
)

from jwt.training.metrics.metric import AbsoluteMetric
from jwt.training.metrics.utils import _to_16khz_mono


class NISQA(nn.Module, AbsoluteMetric):
    DIMS = ("mos", "noi", "dis", "col", "loud")

    @torch.inference_mode()
    def forward(
        self,
        waveforms: torch.Tensor,
        mask: torch.Tensor | None = None,
        sample_rate: int = 16_000,
    ) -> dict[str, torch.Tensor]:
        return self.score(waveforms, mask, sample_rate)

    def score(
        self,
        waveforms: torch.Tensor,
        mask: torch.Tensor | None = None,
        sample_rate: int = 16_000,
    ) -> dict[str, torch.Tensor]:
        """Per-dimension NISQA scores, each a (B,) tensor keyed `nisqa_<dim>`."""
        if mask is not None:
            raise NotImplementedError(
                "masked scoring not supported; pass full-length waveforms"
            )

        waveforms = _to_16khz_mono(waveforms.detach(), sample_rate)
        scores = non_intrusive_speech_quality_assessment(waveforms, 16_000)
        return {
            f"nisqa_{dim}": score
            for (dim, score) in zip(self.DIMS, scores.T, strict=True)
        }
