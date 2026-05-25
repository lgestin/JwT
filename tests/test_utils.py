from pathlib import Path

import numpy as np
import pytest

from jwt.data.audio.utils import load_waveform, resample

TEST_FILE_PATH = Path(__file__).parent / "assets" / "physicsworks.wav"


def test_load_waveform_dtype_and_shape() -> None:
    loaded, sr = load_waveform(TEST_FILE_PATH.as_posix())
    assert loaded.dtype == np.int16
    assert isinstance(loaded, np.ndarray)
    assert loaded.ndim == 2
    assert isinstance(sr, int)
    assert sr > 0


def test_load_waveform_with_target_sample_rate() -> None:
    _, native_sr = load_waveform(TEST_FILE_PATH.as_posix())
    target_sr = native_sr // 2
    resampled, sr = load_waveform(TEST_FILE_PATH.as_posix(), sample_rate=target_sr)
    assert resampled.dtype == np.int16
    assert sr == native_sr  # load_waveform returns the source sr, not the target


def test_load_waveform_start_end() -> None:
    full, sr = load_waveform(TEST_FILE_PATH.as_posix())
    sliced, _ = load_waveform(TEST_FILE_PATH.as_posix(), start=0, end=sr)
    assert sliced.shape[-1] == sr
    assert sliced.shape[0] == full.shape[0]


srs = [4000, 8000, 11025, 16000, 22050, 24000, 44100, 48000]


@pytest.mark.parametrize("orig_sr", srs)
@pytest.mark.parametrize("targ_sr", srs)
def test_resample(orig_sr: int, targ_sr: int) -> None:
    rng = np.random.default_rng(0)
    t_s = 2
    waveform = rng.integers(-(2**15), 2**15, size=(1, t_s * orig_sr), dtype=np.int16)
    resampled = resample(waveform, orig_sr=orig_sr, targ_sr=targ_sr)
    assert resampled.shape[-1] == t_s * targ_sr
    assert resampled.dtype == waveform.dtype
    assert resampled.shape[0] == waveform.shape[0]


def test_resample_identity_returns_input() -> None:
    waveform = np.zeros((1, 1024), dtype=np.int16)
    out = resample(waveform, orig_sr=16000, targ_sr=16000)
    assert out is waveform
