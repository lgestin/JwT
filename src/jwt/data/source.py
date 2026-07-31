import csv
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import torch

from jwt.data.audio import Audio, AudioFile
from jwt.data.text.text import Text
from jwt.data.text.tokenizer import Tokenizer


class TTSSource(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[Audio | AudioFile, Text]: ...


class LJTTSSource(TTSSource):
    def __init__(self, folder_path: str, tokenizer: Tokenizer | None = None):
        folder = Path(folder_path)
        items: list[tuple[Path, str]] = []
        with open(folder / "metadata.csv", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
            for row in reader:
                if not row:
                    continue
                audio_id, _, normalized = row[0], row[1], row[2]
                items.append((folder / "wavs" / f"{audio_id}.wav", normalized))
        self.items = items
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[AudioFile, Text]:
        audio_path, text = self.items[idx]
        return AudioFile(filepath=str(audio_path)), Text(
            text=text, tokenizer=self.tokenizer
        )


class ArrowTTSSource(TTSSource):
    """Reads a pre-processed PyArrow IPC file produced by
    ``scripts/data/create_arrow_ljspeech.py``.

    Each row carries the waveform (int16 PCM, already resampled and
    loudness-normalized), text, phonemes, tokenizer ids, and the codec-encoded
    acoustic features — so ``__getitem__`` is pure decode + tensor construction.

    ``codec_name`` selects which ``acoustic_{codec_name}`` column to read, so
    one arrow file per codec keeps schemas explicit on disk.
    """

    def __init__(self, arrow_path: str, tokenizer: Tokenizer, codec_name: str):
        source = pa.memory_map(arrow_path, "r")
        reader = pa.ipc.open_file(source)
        self._table = reader.read_all()
        self.tokenizer = tokenizer
        self._codec_name = codec_name.lower()
        self._acoustic_field = f"acoustic_{self._codec_name}"

    def __len__(self) -> int:
        return self._table.num_rows

    @property
    def sample_rate(self) -> int:
        """Sample rate of the stored audio. create_arrow_ljspeech.py resamples
        every clip to one rate, so the first row's value holds for the file."""
        if self._table.num_rows == 0:
            raise ValueError("arrow file is empty; cannot read its sample rate")
        return self._table.column("sample_rate")[0].as_py()

    def __getitem__(self, idx: int) -> tuple[Audio, Text]:
        row = {name: self._table.column(name)[idx] for name in self._table.column_names}

        sample_rate = row["sample_rate"].as_py()
        loudness = row["loudness"].as_py()
        acoustic_dim = row["acoustic_dim"].as_py()
        n_frames = row["n_frames"].as_py()

        waveform_i16 = torch.frombuffer(
            bytearray(row["waveform_i16"].as_py()), dtype=torch.int16
        )
        waveform = waveform_i16.view(1, -1).float() / 32678.0

        acoustic = torch.frombuffer(
            bytearray(row[self._acoustic_field].as_py()), dtype=torch.float
        )
        acoustic = acoustic.view(1, acoustic_dim, n_frames).float()

        audio = Audio(
            waveform=waveform,  # ty: ignore[invalid-argument-type]
            sample_rate=sample_rate,
            loudness=loudness,
            acoustic=acoustic,  # ty: ignore[invalid-argument-type]
        )
        text = Text(text=row["text"].as_py(), tokenizer=self.tokenizer)
        return audio, text
