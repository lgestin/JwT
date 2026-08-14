"""Shared metric helpers.

The hot-loop diagnostics (binned_loss_stats, per_pos_l1_error,
masked_mean_std) are deliberately sync-free: every function returns GPU
tensors and never calls `.item()`/`.cpu()`. The trainer accumulates the
results on-device and performs a single batched host transfer per logging
window.
"""

import torch
import torchaudio


def _to_16khz_mono(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if wav.dim() == 1:
        wav = wav[None]
    wav = wav.float()
    if sample_rate != 16_000:
        wav = torchaudio.functional.resample(wav, sample_rate, 16_000)
    return wav


def binned_loss_stats(
    per_pos_loss: torch.Tensor,  # (B, T)
    t: torch.Tensor,  # (B, T), values in [0, 1]
    v_mask: torch.Tensor,  # (B, T), bool supervision mask
    n_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-bin (sum, count) of the flow-matching loss bucketed by timestep ``t``.

    Pure GPU ops — returns two ``(n_bins,)`` tensors on the input device with no
    host sync. The caller accumulates ``sums``/``counts`` across the logging
    window and divides once at log time, which keeps the per-bin means exact
    (not a biased mean-of-means).

    ``t`` is clamped to exactly ``[0, 1]`` at the rolling-window edges, so the
    bin index is clamped to ``[0, n_bins - 1]``: ``t == 1.0`` folds into the
    last bin rather than spilling to index ``n_bins``, and ``t == 0.0`` into the
    first.
    """
    loss = per_pos_loss.detach().float().reshape(-1)
    w = v_mask.reshape(-1).to(loss.dtype)
    idx = torch.clamp((t.reshape(-1) * n_bins).long(), 0, n_bins - 1)

    sums = torch.zeros(n_bins, device=loss.device, dtype=loss.dtype)
    counts = torch.zeros(n_bins, device=loss.device, dtype=loss.dtype)
    sums.scatter_add_(0, idx, loss * w)
    counts.scatter_add_(0, idx, w)
    return sums, counts


def per_pos_l1_error(
    pred: torch.Tensor,  # (B, T, D)
    target: torch.Tensor,  # (B, T, D)
) -> torch.Tensor:
    """Per-position mean-absolute error, averaged over the feature dim.

    Returns a ``(B, T)`` tensor — the *un-reweighted* x_1-prediction error.
    Binned by timestep it disentangles genuine denoising difficulty from any
    parametrization-induced loss reweighting (e.g. JWT's ``1/(1-t)`` factor,
    which amplifies the FM loss near ``t == 1`` regardless of accuracy).

    Sync-free and detached: a hot-loop diagnostic that never retains the graph.
    """
    return (pred.detach().float() - target.detach().float()).abs().mean(-1)


def masked_mean_std(
    values: torch.Tensor,  # (B, D, T)
    frame_mask: torch.Tensor,  # (B, T), bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std of ``values`` over the frames selected by ``frame_mask``.

    Sync-free: uses masked sums rather than boolean indexing (``values[mask]``
    would force a device sync to size the output). Returns two 0-dim GPU
    tensors; an all-False mask yields NaN.
    """
    vals = values.detach().float()
    m = frame_mask.unsqueeze(1).to(vals.dtype)  # (B, 1, T) — broadcasts over D
    count = m.sum() * vals.shape[1]
    mean = (vals * m).sum() / count
    var = ((vals - mean) ** 2 * m).sum() / count
    return mean, var.clamp(min=0).sqrt()
