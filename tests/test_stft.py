import pytest
import torch

from tts.data.audio import Audio
from tts.data.stft import STFT, MelSpectrogram

n_ffts = [2**i for i in range(5, 11)]
hop_length_ratios = [0.1, 0.25, 0.5]


@pytest.mark.parametrize("n_fft", n_ffts)
@pytest.mark.parametrize("hop_length_ratio", hop_length_ratios)
def test_stft_roundtrip_with_files(
    audio_from_file: Audio, n_fft: int, hop_length_ratio: float
) -> None:
    hop_length = int(n_fft * hop_length_ratio)
    stft = STFT(n_fft=n_fft, hop_length=hop_length)

    x_stft = stft.stft(audio_from_file.waveform)
    x_istft = stft.istft(x_stft, length=audio_from_file.waveform.shape[-1])
    assert torch.is_tensor(x_stft)
    assert torch.is_tensor(x_istft)
    assert x_istft.shape == audio_from_file.waveform.shape
    torch.testing.assert_close(
        x_istft,
        audio_from_file.waveform,
        atol=3e-6,
        rtol=1e-5,
    )


@pytest.mark.parametrize("n_fft", n_ffts)
@pytest.mark.parametrize("hop_length_ratio", hop_length_ratios)
def test_stft_roundtrip_with_random(
    random_audio: Audio, n_fft: int, hop_length_ratio: float
) -> None:
    hop_length = int(n_fft * hop_length_ratio)
    stft = STFT(n_fft=n_fft, hop_length=hop_length)

    x_stft = stft.stft(random_audio.waveform)
    x_istft = stft.istft(x_stft, length=random_audio.waveform.shape[-1])
    assert torch.is_tensor(x_stft)
    assert torch.is_tensor(x_istft)
    assert x_istft.shape == random_audio.waveform.shape
    # Trim boundary samples: white-noise input has full energy at the edges,
    # which the windowed istft cannot perfectly reconstruct.
    trim = n_fft
    torch.testing.assert_close(
        x_istft[..., trim:-trim],
        random_audio.waveform[..., trim:-trim],
        atol=3e-6,
        rtol=1e-5,
    )


def test_stft_magnitudes_shape(random_audio: Audio) -> None:
    n_fft, hop_length = 256, 64
    stft = STFT(n_fft=n_fft, hop_length=hop_length)
    mags = stft.magnitudes(random_audio.waveform)
    assert mags.shape[-2] == n_fft // 2 + 1
    assert (mags >= 0).all()


def test_melspectrogram_shape(random_audio: Audio) -> None:
    n_fft, hop_length, n_mels = 512, 128, 80
    mel = MelSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        sample_rate=random_audio.sample_rate,
    )
    out = mel(random_audio.waveform)
    assert out.shape[-2] == n_mels
    assert (out >= 0).all()
