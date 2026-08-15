"""Masked time-domain SNR metrics — GPU-native and sync-free, safe for the
training hot loop."""

import torch

from jwt.data.audio.stft import STFT


def _masked_zero_mean(
    x: torch.Tensor,  # (B, S)
    m: torch.Tensor,  # (B, S), float 0/1
) -> torch.Tensor:
    count = m.sum(dim=-1, keepdim=True).clamp(min=1)
    mean = (x * m).sum(dim=-1, keepdim=True) / count
    return (x - mean) * m


@torch.inference_mode()
def si_snr(
    pred: torch.Tensor,  # (B, S)
    target: torch.Tensor,  # (B, S)
    mask: torch.Tensor,  # (B, S), bool
    zero_mean: bool = True,
) -> torch.Tensor:
    """Masked scale-invariant SNR in dB, per utterance — returns (B,)."""
    p = pred.float()
    t = target.float()
    m = mask.to(p.dtype)
    if zero_mean:
        p = _masked_zero_mean(p, m)
        t = _masked_zero_mean(t, m)
    else:
        p = p * m
        t = t * m
    scale = (p * t).sum(-1, keepdim=True) / (t * t).sum(-1, keepdim=True).clamp(
        min=1e-8
    )
    s = scale * t
    e = p - s
    ratio = (s * s).sum(-1) / (e * e).sum(-1).clamp(min=1e-8)
    return 10 * torch.log10(ratio.clamp(min=1e-8))


@torch.inference_mode()
def mag_snr(
    pred: torch.Tensor,  # (B, S)
    target: torch.Tensor,  # (B, S)
    mask: torch.Tensor,  # (B, S), bool
    stft: STFT,
) -> torch.Tensor:
    """Masked STFT-magnitude SNR in dB, per utterance — returns (B,).

    Phase-blind counterpart of `snr`: discarding phase can only shrink the
    error (reverse triangle inequality), so `mag_snr - snr` isolates the
    phase-attributable part of the waveform error.
    """
    p = pred.float()
    t = target.float()
    m = mask.to(p.dtype)
    p_mag = stft.magnitudes(p * m)
    t_mag = stft.magnitudes(t * m)
    e = t_mag - p_mag
    ratio = (t_mag * t_mag).sum((-2, -1)) / (e * e).sum((-2, -1)).clamp(min=1e-8)
    return 10 * torch.log10(ratio.clamp(min=1e-8))


@torch.inference_mode()
def snr(
    pred: torch.Tensor,  # (B, S)
    target: torch.Tensor,  # (B, S)
    mask: torch.Tensor,  # (B, S), bool
) -> torch.Tensor:
    """Masked plain SNR in dB, per utterance — returns (B,). Scale-sensitive."""
    p = pred.float()
    t = target.float()
    m = mask.to(p.dtype)
    e = (t - p) * m
    t = t * m
    ratio = (t * t).sum(-1) / (e * e).sum(-1).clamp(min=1e-8)
    return 10 * torch.log10(ratio.clamp(min=1e-8))
