from pathlib import Path

import pytest
import torch

from tts.data.audio import AudioFile

TEST_SEED = 42
ASSETS_FOLDER = Path(__file__).parent / "assets"
AUDIO_TEST_FILES = [fpath.as_posix() for fpath in sorted(ASSETS_FOLDER.glob("*.wav"))]


@pytest.fixture(params=AUDIO_TEST_FILES)
def audio_from_file(request) -> AudioFile:
    return AudioFile(request.param)


@pytest.fixture(
    params=[
        (1, 16000, 16000),
        (2, 24000, 24000),
        (1, 48000, 96000),
        (2, 22050, 44100),
        (1, 8000, 4000),
    ]
)
def random_audio(request) -> AudioFile:
    channels, sample_rate, num_samples = request.param
    generator = torch.Generator().manual_seed(TEST_SEED)
    waveform = torch.randn(channels, num_samples, generator=generator)
    return AudioFile(waveform=waveform, sample_rate=sample_rate)
