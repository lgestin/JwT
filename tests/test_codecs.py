import pytest
import torch

from jwt.data.audio import AudioFile
from jwt.data.audio.codecs import (
    BigVGAN,
    BigVGANVersions,
    Codec,
    Codecs,
    RawAudioPatcher,
    check_sample_rate,
)


@pytest.fixture(scope="module")
def codec() -> BigVGAN:
    return BigVGAN(version=BigVGANVersions.V2_24KHz_100MEL_256X)


def test_bigvgan_encode_shape(codec: BigVGAN) -> None:
    # 1 second of synthetic audio at 24 kHz, mono, batched
    sr = codec.decoder.h.sampling_rate
    n_mels = codec.decoder.h.num_mels
    hop = codec.decoder.h.hop_size
    waveform = torch.randn(1, sr)
    mel = codec.encode(waveform)
    assert mel.dim() == 3
    assert mel.shape[1] == n_mels
    # T_mel ≈ T_wav / hop (centered STFT gives ceil-style framing)
    assert abs(mel.shape[2] - waveform.shape[-1] // hop) <= 2
    assert torch.isfinite(mel).all()


def test_bigvgan_decode_shape(codec: BigVGAN) -> None:
    sr = codec.decoder.h.sampling_rate
    n_mels = codec.decoder.h.num_mels
    hop = codec.decoder.h.hop_size
    t_mel = sr // hop  # ~1 second worth of frames
    mel = torch.randn(1, n_mels, t_mel)
    waveform = codec.decode(mel)
    assert waveform.dim() in (2, 3)
    # Output waveform length is approximately t_mel * hop
    assert abs(waveform.shape[-1] - t_mel * hop) <= hop
    assert torch.isfinite(waveform).all()


def test_bigvgan_roundtrip_shape(codec: BigVGAN) -> None:
    sr = codec.decoder.h.sampling_rate
    waveform = torch.randn(1, sr)
    reconstructed = codec.reconstruct(waveform)
    assert torch.isfinite(reconstructed).all()
    # Reconstructed length within one hop of the input
    hop = codec.decoder.h.hop_size
    assert abs(reconstructed.shape[-1] - waveform.shape[-1]) <= 2 * hop


# --- RawAudioPatcher (no HF download required) ------------------------------


def test_rawaudio_protocol_conformance() -> None:
    assert isinstance(RawAudioPatcher(), Codec)


@pytest.mark.parametrize("patch_size", [128, 256])
def test_rawaudio_roundtrip_exact_multiple(patch_size: int) -> None:
    codec = RawAudioPatcher(patch_size=patch_size)
    # Length already a multiple of patch_size — round-trip must be exact.
    waveform = torch.randn(2, patch_size * 10)
    z = codec.encode(waveform)
    assert z.shape == (2, patch_size, 10)
    out = codec.decode(z)
    assert out.shape == waveform.shape
    assert torch.equal(out, waveform)


def test_rawaudio_pads_non_multiple_length() -> None:
    codec = RawAudioPatcher(patch_size=256)
    # Non-multiple length: encode pads to the next patch boundary.
    waveform = torch.randn(2, 256 * 3 + 17)
    z = codec.encode(waveform)
    assert z.shape == (2, 256, 4)
    # Decoded length is the padded length, not the original.
    out = codec.decode(z)
    assert out.shape == (2, 256 * 4)
    # The original prefix must match.
    assert torch.equal(out[:, : waveform.shape[-1]], waveform)


def test_rawaudio_eos_detection() -> None:
    codec = RawAudioPatcher(patch_size=256)
    zero_frame = codec.eos_frames(1).T  # (1, patch_size), unnormalized
    assert codec.is_eos(zero_frame).all()
    loud_frame = torch.full((1, 256), 0.5)
    assert not codec.is_eos(loud_frame).any()


def test_rawaudio_normalize_scales_by_wav_std() -> None:
    """normalize divides by wav_std (waveforms sit near -24 dBFS upstream);
    unnormalize is its inverse, so the round-trip is identity."""
    codec = RawAudioPatcher(patch_size=256)
    x = torch.randn(2, 256, 8)
    assert torch.allclose(codec.normalize(x), x / codec.wav_std)
    assert torch.allclose(codec.unnormalize(x), x * codec.wav_std)
    assert torch.allclose(codec.unnormalize(codec.normalize(x)), x)


def test_rawaudio_decode_is_differentiable() -> None:
    """decode must propagate gradients so the auxiliary mel loss, which decodes
    predicted features to a waveform, can backprop through it."""
    codec = RawAudioPatcher(patch_size=256)
    z = torch.randn(2, 256, 8, requires_grad=True)
    out = codec.decode(z)
    assert out.requires_grad
    out.sum().backward()
    assert z.grad is not None


def test_rawaudio_required_sample_rate_is_none() -> None:
    """RawAudioPatcher patches raw samples, so it works at any sample rate."""
    assert RawAudioPatcher.required_sample_rate is None
    assert RawAudioPatcher().required_sample_rate is None


def test_check_sample_rate_accepts_any_rate_for_unconstrained_codec() -> None:
    check_sample_rate(RawAudioPatcher(), 16000)
    check_sample_rate(RawAudioPatcher(), 44100)


# --- Codecs enum: RawAudio variants -----------------------------------------


@pytest.mark.parametrize("patch_size", [32, 64, 128, 256])
def test_rawaudio_variant_builds_patcher_at_its_patch_size(patch_size: int) -> None:
    """Each RawAudio<N> enum variant resolves to a RawAudioPatcher patched at N."""
    codec = Codecs[f"RAWAUDIO_{patch_size}"].codec
    assert isinstance(codec, RawAudioPatcher)
    assert codec.patch_size == patch_size
    assert codec.acoustic_dim == patch_size


@pytest.mark.parametrize("patch_size", [32, 64, 128, 256])
def test_rawaudio_variant_codec_class_is_patcher(patch_size: int) -> None:
    assert Codecs[f"RAWAUDIO_{patch_size}"].codec_class is RawAudioPatcher


def test_rawaudio_variant_str_drives_arrow_column_name() -> None:
    """str(variant).lower() is the codec_name used for acoustic_{name} columns
    by create_arrow_ljspeech.py and ArrowTTSSource."""
    assert str(Codecs["RAWAUDIO_256"]).lower() == "rawaudio256"


def test_bigvgan_codec_class() -> None:
    assert Codecs.BIGVGAN.codec_class is BigVGAN


def test_bigvgan_required_sample_rate_is_24khz() -> None:
    """BigVGAN's vocoder is locked to 24 kHz, exposed as a class attribute so
    it can be checked without downloading the model."""
    assert BigVGAN.required_sample_rate == 24000


# --- BigVGAN (requires HF download) -----------------------------------------


def test_check_sample_rate_rejects_mismatch_for_bigvgan(codec: BigVGAN) -> None:
    with pytest.raises(ValueError, match="24000"):
        check_sample_rate(codec, 16000)


def test_check_sample_rate_accepts_match_for_bigvgan(codec: BigVGAN) -> None:
    check_sample_rate(codec, 24000)


def test_bigvgan_encode_matches_reference_mel(
    codec: BigVGAN, audio_from_file: AudioFile
) -> None:
    """Our encode() must produce numerically the same mel as bigvgan's own
    mel_spectrogram so the vocoder sees mels in its training space."""
    from bigvgan.meldataset import mel_spectrogram as reference_mel

    h = codec.decoder.h
    audio = audio_from_file.mono().resample(h.sampling_rate)
    waveform = audio.waveform

    our_mel = codec.encode(waveform)
    ref_mel = reference_mel(
        waveform,
        n_fft=h.n_fft,
        num_mels=h.num_mels,
        sampling_rate=h.sampling_rate,
        hop_size=h.hop_size,
        win_size=h.win_size,
        fmin=h.fmin,
        fmax=h.fmax,
    )

    assert our_mel.shape == ref_mel.shape
    torch.testing.assert_close(our_mel, ref_mel, atol=1e-4, rtol=1e-4)
