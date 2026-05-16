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


@pytest.fixture
def model() -> RollingFlowSpeaker:
    torch.manual_seed(0)
    cfg = RollingFlowConfig(
        transformer_config=TransformerConfig(dim=32, num_heads=4, num_layers=2),
        vocabulary_size=20,
        mel_dim=N_MELS,
        n_denoising_steps=4,
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
    mel_lens = torch.tensor([5, 8], dtype=torch.long)
    out = model.speak(text, mel_lens)
    assert out.values.shape == (B, N_MELS, 8)
    assert out.mask.shape == (B, 8)
    assert out.mask.sum(-1).tolist() == [5, 8]


def test_speak_outputs_are_finite(
    model: RollingFlowSpeaker, text: MaskedTensor
) -> None:
    out = model.speak(text, torch.tensor([6, 6]))
    assert torch.isfinite(out.values).all()


def test_speak_is_deterministic_with_pinned_x0(
    model: RollingFlowSpeaker, text: MaskedTensor
) -> None:
    mel_lens = torch.tensor([6, 6])
    x_0 = torch.randn(B, N_MELS, 6)
    a = model.speak(text, mel_lens, x_0=x_0)
    b = model.speak(text, mel_lens, x_0=x_0)
    assert torch.equal(a.values, b.values)


def test_speak_actually_updates_positions(
    model: RollingFlowSpeaker, text: MaskedTensor
) -> None:
    """Output should differ from the initial noise — speak must integrate."""
    mel_lens = torch.tensor([6, 6])
    x_0 = torch.randn(B, N_MELS, 6)
    out = model.speak(text, mel_lens, x_0=x_0)
    assert not torch.allclose(out.values, x_0)


def test_training_step_shapes(
    model: RollingFlowSpeaker, text: MaskedTensor, mels: MaskedTensor
) -> None:
    T_mel = mels.values.shape[-1]
    # Pin mel_front so the (front, front+n] window stays inside mel_lens.
    mel_front = torch.tensor([2, 3], dtype=torch.long)
    v_pred, target, loss_mask = model.training_step(text, mels, mel_front=mel_front)
    assert v_pred.shape == (B, T_mel, N_MELS)
    assert target.shape == (B, T_mel, N_MELS)
    assert loss_mask.shape == (B, T_mel)
    assert (loss_mask.sum(-1) == model.cfg.n_denoising_steps).all()


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
