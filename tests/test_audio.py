import math

import pytest
import torch

from jwt.data.audio import AudioFile, AudioInfo


def test_audio_from_file_loads_waveform(audio_from_file: AudioFile) -> None:
    waveform = audio_from_file.waveform
    assert torch.is_tensor(waveform)
    assert waveform.dtype == torch.float32
    assert waveform.ndim == 2
    assert waveform.shape[-1] > 0
    assert audio_from_file.sample_rate > 0


def test_random_audio_basic_properties(random_audio: AudioFile) -> None:
    assert torch.is_tensor(random_audio.waveform)
    assert random_audio.n_frames == random_audio.waveform.shape[-1]
    assert math.isclose(
        random_audio.duration_s,
        random_audio.n_frames / random_audio.sample_rate,
    )


def test_audio_info_roundtrip(audio_from_file: AudioFile) -> None:
    info = audio_from_file.info
    assert isinstance(info, AudioInfo)
    rebuilt = AudioFile.from_audioinfo(info)
    assert rebuilt.filepath == audio_from_file.filepath
    assert rebuilt.sample_rate == audio_from_file.sample_rate
    assert rebuilt.loudness == audio_from_file.loudness


def test_loudness_is_finite(audio_from_file: AudioFile) -> None:
    loudness = audio_from_file.loudness
    assert isinstance(loudness, float)
    assert not math.isnan(loudness)
    assert math.isfinite(loudness)


def test_mono_collapses_channels() -> None:
    waveform = torch.randn(2, 16000)
    audio = AudioFile(waveform=waveform, sample_rate=16000).mono()
    assert audio.waveform.shape[0] == 1
    assert audio.waveform.shape[-1] == 16000


def test_resample_changes_rate_and_length() -> None:
    waveform = torch.randn(1, 16000)
    audio = AudioFile(waveform=waveform, sample_rate=16000)
    audio.resample(8000)
    assert audio.sample_rate == 8000
    assert audio.waveform.shape[-1] == pytest.approx(8000, abs=2)


def test_normalize_sets_loudness() -> None:
    waveform = 0.1 * torch.randn(1, 16000, generator=torch.Generator().manual_seed(0))
    audio = AudioFile(waveform=waveform, sample_rate=16000)
    audio.normalize(db=-20.0)
    assert audio.loudness == pytest.approx(-20.0, abs=1e-3)


def test_excerpt_returns_expected_duration() -> None:
    waveform = torch.randn(1, 32000)
    audio = AudioFile(waveform=waveform, sample_rate=16000)
    excerpt = audio.excerpt(offset_s=0.5, duration_s=1.0)
    assert excerpt.waveform.shape[-1] == 16000
    assert excerpt.sample_rate == 16000


def test_random_excerpt_returns_expected_duration() -> None:
    waveform = torch.randn(1, 48000)
    audio = AudioFile(waveform=waveform, sample_rate=16000)
    generator = torch.Generator().manual_seed(0)
    excerpt = audio.random_excerpt(duration_s=1.0, generator=generator)
    assert excerpt.waveform.shape[-1] == 16000


def test_salient_excerpt_returns_expected_duration(audio_from_file: AudioFile) -> None:
    duration_s = min(0.5, audio_from_file.duration_s / 2)
    generator = torch.Generator().manual_seed(0)
    excerpt = audio_from_file.salient_excerpt(
        duration_s=duration_s, generator=generator
    )
    expected = int(duration_s * excerpt.sample_rate)
    assert excerpt.waveform.shape[-1] == expected


def test_audio_requires_filepath_or_waveform() -> None:
    with pytest.raises(AssertionError):
        AudioFile()


def test_audio_from_waveform_requires_sample_rate() -> None:
    with pytest.raises(AssertionError):
        AudioFile(waveform=torch.zeros(1, 16))


def test_device_property(random_audio: AudioFile) -> None:
    assert random_audio.device == random_audio.waveform.device


def test_stft_property_shape(random_audio: AudioFile) -> None:
    stft = random_audio.stft
    assert torch.is_tensor(stft)
    assert stft.is_complex()


def test_resample_same_rate_is_noop() -> None:
    waveform = torch.randn(1, 16000)
    audio = AudioFile(waveform=waveform, sample_rate=16000)
    assert audio.resample(16000).waveform is waveform


def test_lazy_resample_at_load_matches_eager(audio_from_file: AudioFile) -> None:
    native_sr = audio_from_file.sample_rate
    target_sr = native_sr // 2
    lazy = AudioFile(audio_from_file.filepath, sample_rate=target_sr)
    eager = AudioFile(audio_from_file.filepath).resample(target_sr)
    assert lazy.sample_rate == target_sr
    torch.testing.assert_close(lazy.waveform, eager.waveform)
    assert lazy.sample_rate == target_sr
    assert lazy.resample(target_sr).waveform is lazy.waveform


def test_lazy_excerpt_offsets_use_native_rate(audio_from_file: AudioFile) -> None:
    native_sr = audio_from_file.sample_rate
    target_sr = native_sr // 2
    lazy = AudioFile(
        audio_from_file.filepath, sample_rate=target_sr, start_s=0.1, end_s=0.6
    )
    assert lazy.waveform.shape[-1] == pytest.approx(0.5 * target_sr, abs=2)
