import math

import pytest
import torch

from tts.data.audio.stft import MelSpectrogram
from tts.model.neural_speaker import (
    MaskedTensor,
    RollingFlowConfig,
    RollingFlowSpeaker,
)
from tts.model.transformer import TransformerConfig

B = 2
T_TEXT = 4
N_MELS = 16
T_MEL_MAX = 16
MAX_MEL_LEN = T_MEL_MAX


@pytest.fixture
def model() -> RollingFlowSpeaker:
    torch.manual_seed(0)
    cfg = RollingFlowConfig(
        transformer_config=TransformerConfig(dim=32, num_heads=4, num_layers=2),
        vocabulary_size=20,
        mel_dim=N_MELS,
        n_denoising_steps=4,
        max_mel_len=MAX_MEL_LEN,
    )
    return RollingFlowSpeaker(cfg).eval()


@pytest.fixture
def text(model: RollingFlowSpeaker) -> MaskedTensor:
    return MaskedTensor(
        values=torch.randint(0, model.cfg.vocabulary_size, (B, 1, T_TEXT)),
        mask=torch.ones(B, T_TEXT, dtype=torch.bool),
    )


@pytest.fixture
def mels(model: RollingFlowSpeaker, audio_from_file) -> MaskedTensor:
    """Mels from the most-energetic T_MEL_MAX-frame window of a real audio asset."""
    waveform = audio_from_file.waveform  # (channels, T_audio)
    mono = waveform.mean(dim=0, keepdim=True)  # (1, T_audio)
    batch_wave = mono.repeat(B, 1)  # (B, T_audio)
    mel_spec = MelSpectrogram(
        n_fft=1024,
        hop_length=256,
        n_mels=model.cfg.mel_dim,
        sample_rate=audio_from_file.sample_rate,
    )
    full = mel_spec(batch_wave)  # (B, mel_dim, T_mel_full)
    # Slice the highest-energy window so the test data isn't a leading-silence flat patch.
    energy = full[0].mean(dim=0)  # (T_mel_full,)
    start = int(energy.unfold(0, T_MEL_MAX, 1).mean(dim=-1).argmax().item())
    values = full[..., start : start + T_MEL_MAX]
    return MaskedTensor(values=values, mask=torch.ones(B, T_MEL_MAX, dtype=torch.bool))


def test_speak_shape_and_mask(model: RollingFlowSpeaker, text: MaskedTensor) -> None:
    out = model.speak(text)
    assert out.values.ndim == 3
    assert out.values.shape[0] == B
    assert out.values.shape[1] == N_MELS
    assert out.values.shape[2] <= model.cfg.max_mel_len
    assert out.mask.shape == (B, out.values.shape[2])
    assert (out.mask.sum(-1) >= 1).all()
    assert (out.mask.sum(-1) <= model.cfg.max_mel_len).all()


def test_speak_outputs_are_finite(
    model: RollingFlowSpeaker, text: MaskedTensor
) -> None:
    out = model.speak(text)
    assert torch.isfinite(out.values).all()


def test_speak_is_deterministic_with_pinned_x0(
    model: RollingFlowSpeaker, text: MaskedTensor
) -> None:
    x_0 = torch.randn(B, N_MELS, model.cfg.max_mel_len)
    a = model.speak(text, x_0=x_0)
    b = model.speak(text, x_0=x_0)
    assert torch.equal(a.values, b.values)
    assert torch.equal(a.mask, b.mask)


def test_speak_actually_updates_positions(
    model: RollingFlowSpeaker, text: MaskedTensor
) -> None:
    """Output should differ from the initial noise — speak must integrate."""
    x_0 = torch.randn(B, N_MELS, model.cfg.max_mel_len)
    out = model.speak(text, x_0=x_0)
    # Compare each sample's output to the matching x_0 prefix (still normalized,
    # so denormalize x_0 the same way speak() does its output).
    mean = model.mel_mean
    std = model.mel_std
    for i in range(B):
        L = int(out.mask[i].sum().item())
        x_0_denorm = x_0[i, :, :L] * std + mean
        assert not torch.allclose(out.values[i, :, :L], x_0_denorm)


def test_speak_per_sample_lengths_can_differ(
    model: RollingFlowSpeaker, text: MaskedTensor
) -> None:
    """API path for per-sample autonomous stopping must work (untrained head)."""
    out = model.speak(text)
    lens = out.mask.sum(-1)
    assert (lens >= 1).all()
    assert (lens <= model.cfg.max_mel_len).all()
    # Mask is exactly mel_idx < lens[b]
    mel_idx = torch.arange(out.values.shape[-1]).expand(B, -1)
    expected_mask = mel_idx < lens.unsqueeze(1)
    assert torch.equal(out.mask, expected_mask)


def test_training_step_shapes(
    model: RollingFlowSpeaker, text: MaskedTensor, mels: MaskedTensor
) -> None:
    T_mel = mels.values.shape[-1]
    # Pin mel_front so the (front, front+n-1] window stays inside mel_lens.
    mel_front = torch.tensor([2, 3], dtype=torch.long)
    v_pred, v_target, v_mask, r_pred, r_target, r_mask, t = model.training_step(
        text, mels, mel_front=mel_front
    )
    assert v_pred.shape == (B, T_mel, N_MELS)
    assert v_target.shape == (B, T_mel, N_MELS)
    assert v_mask.shape == (B, T_mel)
    # Window is now n-1 positions (ramp + first t=0, dropping the second t=0).
    assert (v_mask.sum(-1) == model.cfg.n_denoising_steps - 1).all()
    assert r_pred.shape == (B, T_mel)
    assert r_target.shape == (B, T_mel)
    assert r_mask.shape == (B, T_mel)
    assert torch.isfinite(r_pred).all()
    assert t.shape == (B, T_mel)


def test_training_step_is_deterministic(
    model: RollingFlowSpeaker, text: MaskedTensor, mels: MaskedTensor
) -> None:
    T_mel = mels.values.shape[-1]
    mel_front = torch.tensor([2, 3], dtype=torch.long)
    x_0 = torch.randn(B, T_mel, N_MELS)
    a = model.training_step(text, mels, mel_front=mel_front, x_0=x_0)
    b = model.training_step(text, mels, mel_front=mel_front, x_0=x_0)
    for ta, tb in zip(a, b):
        assert torch.equal(ta, tb)


def test_remaining_target_correctness(model: RollingFlowSpeaker) -> None:
    """r_target[b, i] = log1p(max(L_b - i - 1, 0)) on real frames; r_mask = real & t=1."""
    T_mel = 8
    # Two samples with different real lengths: 5 and 8.
    mask = torch.tensor(
        [
            [True, True, True, True, True, False, False, False],
            [True, True, True, True, True, True, True, True],
        ]
    )
    mel_values = torch.randn(B, N_MELS, T_mel)
    mels = MaskedTensor(values=mel_values, mask=mask)
    text = MaskedTensor(
        values=torch.randint(0, model.cfg.vocabulary_size, (B, 1, T_TEXT)),
        mask=torch.ones(B, T_TEXT, dtype=torch.bool),
    )
    mel_front = torch.tensor([1, 2], dtype=torch.long)
    _, _, _, _, r_target, r_mask, _ = model.training_step(
        text, mels, mel_front=mel_front
    )

    # r_mask supervises only positions with t==1 (clean tail up to mel_front).
    expected_r_mask = torch.tensor(
        [
            [True, True, False, False, False, False, False, False],
            [True, True, True, False, False, False, False, False],
        ]
    )
    assert torch.equal(r_mask, expected_r_mask)
    # Real-frame targets match log1p(L - i - 1).
    lens = [5, 8]
    for b in range(B):
        for i in range(lens[b]):
            expected = math.log1p(max(lens[b] - i - 1, 0))
            assert r_target[b, i].item() == pytest.approx(expected)
    # Padding positions: target is log1p(<= 0) clamped to log1p(0) = 0.
    assert r_target[0, 5].item() == pytest.approx(0.0)
    assert r_target[0, 7].item() == pytest.approx(0.0)
