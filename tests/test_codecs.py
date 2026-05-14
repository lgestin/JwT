import pytest
import torch

from tts.data.audio import AudioFile
from tts.data.audio.codecs import BigVGAN, BigVGANVersions


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
