import torch
from pesq import PesqError
from pesq import pesq as pesq_score
from torch import nn

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
        pred_np = _to_16khz_mono(pred.detach(), sample_rate).cpu().numpy()
        trgt_np = _to_16khz_mono(trgt.detach(), sample_rate).cpu().numpy()
        scores = torch.tensor(
            [
                pesq_score(16_000, t, p, "wb", on_error=PesqError.RETURN_VALUES)
                for p, t in zip(pred_np, trgt_np, strict=True)
            ],
            dtype=torch.float32,
        )
        # PESQ signals refusal as NaN or a negative code; wide-band scores are
        # >= 1.04. Mark them NaN rather than dropping them so the row stays
        # aligned with the batch and per-sample records keep it — the reduction
        # is nan-aware instead.
        scored = torch.isfinite(scores) & (scores > 0)
        return {
            "pesq": scores.where(scored, torch.nan),
            "pesq_scored": torch.tensor([float(scored.sum())]),
        }
