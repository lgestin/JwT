import torch
import torch.nn as nn

from jwt.training.metrics.metric import AbsoluteMetric
from jwt.training.metrics.utils import _to_16khz_mono


class UTMOS(nn.Module, AbsoluteMetric):
    def __init__(self):
        super().__init__()
        model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        )
        model = model.eval()
        self.model = model

    @property
    def device(self):
        return next(self.parameters()).device

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
        if mask is not None:
            raise NotImplementedError(
                "masked scoring not supported; pass full-length waveforms"
            )
        waveforms = _to_16khz_mono(waveforms.detach(), sample_rate)
        scores = self.model(waveforms, 16_000)
        return {"utmos": torch.as_tensor(scores)}
