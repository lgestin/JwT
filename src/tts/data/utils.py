import numpy as np
import soundfile as sf
import torch
from av.audio.frame import AudioFrame
from av.audio.resampler import AudioResampler


def load_waveform(
    path: str,
    sample_rate: int | None = None,
    start: int = 0,
    end: int | None = None,
):
    num_frames = -1 if end is None else end - start
    with open(path, "rb") as audio_file:
        waveform, sr = sf.read(
            audio_file,
            start=start,
            frames=num_frames,
            dtype="int16",
            always_2d=True,
        )
        waveform = np.transpose(waveform)
    if sample_rate:
        waveform = resample(waveform, orig_sr=sr, targ_sr=sample_rate)
    return waveform, int(sr)


def av_resampler_layout_from_waveform(waveform: np.ndarray):
    assert 1 <= waveform.shape[0] <= 2
    if waveform.shape[0] == 1:
        layout = "mono"
    elif waveform.shape[0] == 2:
        layout = "stereo"
    else:
        raise ValueError
    return layout


def resample(waveform: np.ndarray, orig_sr: int, targ_sr: int):
    assert isinstance(waveform, np.ndarray)
    assert waveform.dtype == np.int16
    assert waveform.ndim == 2

    if orig_sr == targ_sr:
        return waveform

    layout = av_resampler_layout_from_waveform(waveform)
    resampler = AudioResampler(format="s16", layout=layout, rate=targ_sr)
    frame = AudioFrame.from_ndarray(waveform, layout=layout)
    frame.rate = orig_sr
    frame = resampler.resample(frame)
    flush = resampler.resample(None)
    resampled = np.concat([frame[0].to_ndarray(), flush[0].to_ndarray()], axis=-1)
    return resampled


def truncated_normal(
    size: tuple,
    min_val: float,
    max_val: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    approximation of a truncated normal distribution between min and max
    """
    normal = torch.randn(size, generator=generator)
    trunc_normal = torch.fmod(normal, 2)
    trunc_normal = (trunc_normal / 2 + 1) / 2
    trunc_normal = trunc_normal * (max_val - min_val)
    trunc_normal = trunc_normal + min_val
    return trunc_normal
