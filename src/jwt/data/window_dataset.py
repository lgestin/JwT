"""Fixed-length audio windows for spectrogram-inversion training.

No text, no padding: every sample is a random ``n_frames * patch_size``-sample
crop of a clip's waveform, so batches are uniform tensors and collate is a
plain stack. Clips shorter than the window are excluded up front rather than
padded.
"""

from dataclasses import dataclass, fields

import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch.utils.data import Dataset


class WindowedAudioSource:
    """Reads waveforms from an arrow file produced by
    ``scripts/data/create_arrow_ljspeech.py``.

    Only the ``waveform_i16`` column is used (already resampled and
    loudness-normalized); the per-codec ``acoustic_*`` columns are ignored, so
    an arrow built for any codec works as a waveform source.
    """

    def __init__(self, arrow_path: str):
        source = pa.memory_map(arrow_path, "r")
        reader = pa.ipc.open_file(source)
        self._table = reader.read_all()

    def __len__(self) -> int:
        return self._table.num_rows

    @property
    def sample_rate(self) -> int:
        """Sample rate of the stored audio. create_arrow_ljspeech.py resamples
        every clip to one rate, so the first row's value holds for the file."""
        if self._table.num_rows == 0:
            raise ValueError("arrow file is empty; cannot read its sample rate")
        return self._table.column("sample_rate")[0].as_py()

    def waveform_lengths(self) -> list[int]:
        """Per-row waveform length in samples, without decoding the audio."""
        # binary_length exists at runtime; ty's pyarrow stubs don't know it.
        n_bytes = pc.binary_length(  # ty: ignore[unresolved-attribute]
            self._table.column("waveform_i16")
        )
        return [n // 2 for n in n_bytes.to_pylist()]  # int16 = 2 bytes/sample

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Waveform ``(S,)`` float. Same int16 scale as ``ArrowTTSSource``."""
        raw = self._table.column("waveform_i16")[idx].as_py()
        waveform_i16 = torch.frombuffer(bytearray(raw), dtype=torch.int16)
        return waveform_i16.float() / 32678.0


@dataclass
class WindowSample:
    idx: int
    waveform: torch.FloatTensor  # (window,)


@dataclass
class WindowBatch:
    idxs: list[int]
    waveform: torch.FloatTensor  # (B, window)

    def to(self, device: str | torch.device, non_blocking: bool = False):
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                value = value.to(device, non_blocking=non_blocking)
                setattr(self, field.name, value)
        return self

    def pin_memory(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                setattr(self, field.name, value.pin_memory())
        return self


def collate_windows(samples: list[WindowSample]) -> WindowBatch:
    return WindowBatch(
        idxs=[s.idx for s in samples],
        waveform=torch.stack([s.waveform for s in samples]),  # ty: ignore[invalid-argument-type]
    )


class WindowedAudioDataset(Dataset):
    """Fixed-length windows over the clips long enough to supply one.

    ``deterministic=True`` (validation/sample splits) always crops from the
    start of the clip so metrics stay comparable across steps; otherwise the
    crop position is uniform over the clip.
    """

    def __init__(
        self,
        source: WindowedAudioSource,
        n_frames: int = 256,
        patch_size: int = 256,
        deterministic: bool = False,
    ):
        self.source = source
        self.window = n_frames * patch_size
        self.deterministic = deterministic
        self.idxs = [
            i for i, n in enumerate(source.waveform_lengths()) if n >= self.window
        ]

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, index: int) -> WindowSample:
        idx = self.idxs[index]
        waveform = self.source[idx]
        margin = waveform.shape[-1] - self.window
        start = 0 if self.deterministic else int(torch.randint(margin + 1, ()))
        return WindowSample(
            idx=idx,
            waveform=waveform[start : start + self.window],  # ty: ignore[invalid-argument-type]
        )
