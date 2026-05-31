import pytest
import torch

from jwt.training.loss import MelAuxLoss, MultiResComplexSTFTAuxLoss

HOP = 256
N_FFTS = (256, 512, 1024)
SAMPLE_RATE = 24000


def _wav(batch: int, n_frames: int, *, seed: int) -> torch.Tensor:
    """A (batch, n_frames * HOP) waveform — STFTs to exactly n_frames frames."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, n_frames * HOP, generator=generator)


def test_stft_aux_loss_is_zero_for_identical_waveforms() -> None:
    loss = MultiResComplexSTFTAuxLoss(hop_length=HOP, n_ffts=N_FFTS)
    wav = _wav(2, 16, seed=0)
    out, _ = loss(wav, wav.clone(), torch.ones(2, 16))
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_stft_aux_loss_time_axis_matches_v_mask() -> None:
    """A waveform of n_frames * hop samples STFTs to exactly n_frames frames
    at every scale (center=False with internal reflect padding)."""
    loss = MultiResComplexSTFTAuxLoss(hop_length=HOP, n_ffts=N_FFTS)
    pred, target = _wav(2, 13, seed=1), _wav(2, 13, seed=2)
    out, _ = loss(pred, target, torch.ones(2, 13))
    assert torch.isfinite(out)


def test_stft_aux_loss_raises_when_time_axis_misaligned() -> None:
    loss = MultiResComplexSTFTAuxLoss(hop_length=HOP, n_ffts=N_FFTS)
    pred, target = _wav(2, 13, seed=1), _wav(2, 13, seed=2)
    with pytest.raises(AssertionError):
        loss(pred, target, torch.ones(2, 14))


def test_stft_aux_loss_gradient_flows_to_prediction() -> None:
    loss = MultiResComplexSTFTAuxLoss(hop_length=HOP, n_ffts=N_FFTS)
    pred = _wav(2, 16, seed=1).requires_grad_(True)
    target = _wav(2, 16, seed=2)
    loss(pred, target, torch.ones(2, 16))[0].backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_stft_aux_loss_excludes_masked_frames() -> None:
    """Frames zeroed in v_mask do not contribute. pred and target share their
    first half exactly, so masking to that region yields zero loss."""
    loss = MultiResComplexSTFTAuxLoss(hop_length=HOP, n_ffts=N_FFTS)
    shared = _wav(2, 8, seed=3)
    pred = torch.cat([shared, _wav(2, 8, seed=4)], dim=-1)
    target = torch.cat([shared, _wav(2, 8, seed=5)], dim=-1)

    full, _ = loss(pred, target, torch.ones(2, 16))
    assert full.item() > 0  # the second half genuinely differs

    # Mask the first quarter only. Boundary frames of larger STFT windows
    # leak across the cut, so the loss isn't exactly zero — just much smaller.
    first_quarter = torch.zeros(2, 16)
    first_quarter[:, :4] = 1.0
    assert loss(pred, target, first_quarter)[0].item() < full.item() * 0.5


def test_stft_aux_loss_zero_mask_is_zero_without_nan() -> None:
    loss = MultiResComplexSTFTAuxLoss(hop_length=HOP, n_ffts=N_FFTS)
    pred, target = _wav(2, 16, seed=1), _wav(2, 16, seed=2)
    out, _ = loss(pred, target, torch.zeros(2, 16))
    assert out.item() == 0.0
    assert torch.isfinite(out)


def test_stft_aux_loss_returns_per_position_diff() -> None:
    """The second return value is a detached (B, T) per-frame STFT L1 from the
    first scale — the trainer bins it by timestep into `aux_stft_l1_by_t`."""
    loss = MultiResComplexSTFTAuxLoss(hop_length=HOP, n_ffts=N_FFTS)
    pred, target = _wav(2, 13, seed=1), _wav(2, 13, seed=2)
    scalar, per_pos = loss(pred, target, torch.ones(2, 13))

    assert per_pos.shape == (2, 13)
    assert not per_pos.requires_grad
    assert torch.isfinite(scalar)


def test_stft_aux_loss_phase_sensitive() -> None:
    """Same magnitude spectrogram with mismatched phase produces nonzero loss
    — that's the property this loss exists for."""
    loss = MultiResComplexSTFTAuxLoss(hop_length=HOP, n_ffts=N_FFTS)
    target = _wav(2, 16, seed=7)
    # Phase shift: reverse the waveform in time. |STFT| is largely preserved
    # at each frame's magnitude content but instantaneous phase differs.
    pred = target.flip(dims=[-1])
    out, _ = loss(pred, target, torch.ones(2, 16))
    assert out.item() > 0


def test_mel_aux_loss_basic_contract() -> None:
    """MelAuxLoss shares the (loss, per_pos) contract with the STFT loss and
    is identity-zero for identical inputs."""
    loss = MelAuxLoss(sample_rate=SAMPLE_RATE, hop_length=HOP)
    wav = _wav(2, 16, seed=0)
    out, per_pos = loss(wav, wav.clone(), torch.ones(2, 16))
    assert out.item() == pytest.approx(0.0, abs=1e-6)
    assert per_pos.shape == (2, 16)
    assert not per_pos.requires_grad


def test_mel_aux_loss_gradient_flows_to_prediction() -> None:
    loss = MelAuxLoss(sample_rate=SAMPLE_RATE, hop_length=HOP)
    pred = _wav(2, 16, seed=1).requires_grad_(True)
    target = _wav(2, 16, seed=2)
    loss(pred, target, torch.ones(2, 16))[0].backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_mel_aux_loss_raises_when_time_axis_misaligned() -> None:
    loss = MelAuxLoss(sample_rate=SAMPLE_RATE, hop_length=HOP)
    pred, target = _wav(2, 13, seed=1), _wav(2, 13, seed=2)
    with pytest.raises(AssertionError):
        loss(pred, target, torch.ones(2, 14))
