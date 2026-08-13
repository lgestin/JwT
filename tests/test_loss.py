import pytest
import torch

from jwt.training.loss import MelAuxLoss

HOP = 256
SAMPLE_RATE = 24000


def _wav(batch: int, n_frames: int, *, seed: int) -> torch.Tensor:
    """A (batch, n_frames * HOP) waveform — decodes to exactly n_frames mel frames."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, n_frames * HOP, generator=generator)


def test_mel_aux_loss_is_zero_for_identical_waveforms() -> None:
    loss = MelAuxLoss(sample_rate=SAMPLE_RATE, hop_length=HOP)
    wav = _wav(2, 16, seed=0)
    out, _ = loss(wav, wav.clone(), torch.ones(2, 16))
    assert out.item() == 0.0


def test_mel_aux_loss_time_axis_matches_v_mask() -> None:
    """A waveform of n_frames * hop samples mels to exactly n_frames frames,
    so a v_mask in acoustic time aligns frame-for-frame."""
    loss = MelAuxLoss(sample_rate=SAMPLE_RATE, hop_length=HOP)
    pred, target = _wav(2, 13, seed=1), _wav(2, 13, seed=2)
    out, _ = loss(pred, target, torch.ones(2, 13))
    assert torch.isfinite(out)


def test_mel_aux_loss_raises_when_time_axis_misaligned() -> None:
    loss = MelAuxLoss(sample_rate=SAMPLE_RATE, hop_length=HOP)
    pred, target = _wav(2, 13, seed=1), _wav(2, 13, seed=2)
    with pytest.raises(AssertionError):
        loss(pred, target, torch.ones(2, 14))


def test_mel_aux_loss_gradient_flows_to_prediction() -> None:
    loss = MelAuxLoss(sample_rate=SAMPLE_RATE, hop_length=HOP)
    pred = _wav(2, 16, seed=1).requires_grad_(True)
    target = _wav(2, 16, seed=2)
    loss(pred, target, torch.ones(2, 16))[0].backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_mel_aux_loss_excludes_masked_frames() -> None:
    """Frames zeroed in v_mask do not contribute. pred and target share their
    first half exactly, so masking to that region yields zero loss."""
    loss = MelAuxLoss(sample_rate=SAMPLE_RATE, hop_length=HOP)
    shared = _wav(2, 8, seed=3)
    pred = torch.cat([shared, _wav(2, 8, seed=4)], dim=-1)
    target = torch.cat([shared, _wav(2, 8, seed=5)], dim=-1)

    full, _ = loss(pred, target, torch.ones(2, 16))
    assert full.item() > 0  # the second half genuinely differs

    first_quarter = torch.zeros(2, 16)
    first_quarter[:, :4] = 1.0
    assert loss(pred, target, first_quarter)[0].item() == 0.0


def test_mel_aux_loss_zero_mask_is_zero_without_nan() -> None:
    loss = MelAuxLoss(sample_rate=SAMPLE_RATE, hop_length=HOP)
    pred, target = _wav(2, 16, seed=1), _wav(2, 16, seed=2)
    out, _ = loss(pred, target, torch.zeros(2, 16))
    assert out.item() == 0.0
    assert torch.isfinite(out)


def test_mel_aux_loss_returns_per_position_diff() -> None:
    """The second return value is a detached (B, T) per-frame log-mel L1 whose
    v_mask-weighted mean is the scalar loss — the trainer bins it by timestep
    into the `logmel_l1_by_t` curve."""
    loss = MelAuxLoss(sample_rate=SAMPLE_RATE, hop_length=HOP)
    pred, target = _wav(2, 13, seed=1), _wav(2, 13, seed=2)
    v_mask = torch.ones(2, 13)

    scalar, per_pos = loss(pred, target, v_mask)

    assert per_pos.shape == (2, 13)
    assert not per_pos.requires_grad
    weighted_mean = (per_pos * v_mask).sum() / v_mask.sum()
    assert weighted_mean.item() == pytest.approx(scalar.item(), abs=1e-5)
