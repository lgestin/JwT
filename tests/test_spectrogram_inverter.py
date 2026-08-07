import pytest
import torch

from jwt.data.audio.codecs import Codec, RawAudioPatcher
from jwt.data.audio.stft import MelSpectrogram
from jwt.model.neural_speaker import MaskedTensor
from jwt.model.spectrogram_inverter import (
    SpectrogramInverter,
    SpectrogramInverterCodec,
    SpectrogramInverterConfig,
)
from jwt.model.transformer import TransformerConfig

B = 2
ACOUSTIC_DIM = 16
N_MELS = 8
T = 12
N_STEPS = 4


@pytest.fixture
def model() -> SpectrogramInverter:
    torch.manual_seed(0)
    cfg = SpectrogramInverterConfig(
        transformer_config=TransformerConfig(dim=32, num_heads=4, num_layers=2),
        acoustic_dim=ACOUSTIC_DIM,
        n_denoising_steps=N_STEPS,
        n_mels=N_MELS,
        mel_n_fft=ACOUSTIC_DIM * 4,
        mel_hop_length=ACOUSTIC_DIM,
        sample_rate=22050,
    )
    return SpectrogramInverter(cfg)


def masked(values: torch.Tensor) -> MaskedTensor:
    return MaskedTensor(
        values=values,
        mask=torch.ones(values.shape[0], values.shape[-1], dtype=torch.bool),  # ty: ignore[invalid-argument-type]
    )


def test_mel_patch_length_alignment():
    """With hop == patch size and center=False, mel frames == patches exactly."""
    torch.manual_seed(0)
    n_frames = 7
    wav = torch.randn(1, 256 * n_frames) * 0.05
    patches = RawAudioPatcher(patch_size=256).encode(wav)
    mel = MelSpectrogram(
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        sample_rate=22050,
        window="hann",
        center=False,
        mel_scale="slaney",
    )(wav)
    assert patches.shape[-1] == n_frames
    assert mel.shape[-1] == n_frames


def test_forward_and_training_step_shapes(model: SpectrogramInverter):
    acoustic = masked(torch.randn(B, ACOUSTIC_DIM, T))
    mel = masked(torch.randn(B, N_MELS, T))

    t = torch.rand(B, T)
    pred = model.forward(acoustic, mel, t)
    assert pred.shape == (B, T, ACOUSTIC_DIM)

    out = model.training_step(mel, acoustic)
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert out.x_pred.shape == (B, T, ACOUSTIC_DIM)
    assert out.v_mask.dtype == torch.bool
    assert torch.equal(out.v_mask, acoustic.mask)
    # Pure FM: the timestep is constant across positions within a sample.
    assert out.t.shape == (B, T)
    assert torch.equal(out.t, out.t[:, :1].expand(B, T))

    out.loss.backward()


def test_invert_matches_mel_length(model: SpectrogramInverter):
    mel = masked(torch.randn(B, N_MELS, T))
    pred = model.invert(mel)
    assert pred.values.shape == (B, ACOUSTIC_DIM, T)
    assert torch.equal(pred.mask, mel.mask)
    assert torch.isfinite(pred.values).all()


def test_codec_roundtrip_shapes(model: SpectrogramInverter):
    codec = SpectrogramInverterCodec(model)
    assert isinstance(codec, Codec)

    n_frames = 5
    wav = torch.randn(B, ACOUSTIC_DIM * n_frames) * 0.05
    logmel = codec.encode(wav)
    assert logmel.shape == (B, N_MELS, n_frames)
    # normalize(encode(...)) must match the model's training-time conditioning.
    assert torch.allclose(codec.normalize(logmel), model.encode_mel(wav))

    decoded = codec.decode(logmel)
    assert decoded.shape == (B, ACOUSTIC_DIM * n_frames)
