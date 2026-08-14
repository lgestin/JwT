"""Training loss helpers that are not owned by the model or the codec."""

import torch


def masked_mean_reduction(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-row mean of `x` over the positions where `mask` is set.

    An all-zero row yields 0 rather than NaN — this feeds the training loss,
    so a degenerate mask must not poison the batch.
    """
    mean = (x * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    return mean
