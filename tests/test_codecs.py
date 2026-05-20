import pytest
import torch

from tts.data.audio import AudioFile
from tts.data.audio.codecs import BigVGAN, BigVGANVersions, Codec, RawAudioPatcher


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


def test_rawaudio_to_logmel_preserves_time_axis() -> None:
    """to_logmel's time axis must align with the acoustic time axis so the
    trainer can reuse v_mask without resampling."""
    codec = RawAudioPatcher(patch_size=256)
    T_acoustic = 12
    features = torch.randn(2, codec.acoustic_dim, T_acoustic)
    lm = codec.to_logmel(features)
    assert lm.shape[0] == 2
    assert lm.shape[-1] == T_acoustic, (lm.shape, T_acoustic)
    assert torch.isfinite(lm).all()


# --- BigVGAN (requires HF download) -----------------------------------------


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
