import csv
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import torch

from tts.data.audio import Audio, AudioFile
from tts.data.text.text import Text
from tts.data.text.tokenizer import Tokenizer


class TTSSource(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[Audio | AudioFile, Text]: ...


class LJTTSSource(TTSSource):
    def __init__(self, folder_path: str, tokenizer: Tokenizer):
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
    mel — so ``__getitem__`` is pure decode + tensor construction.
    """

    def __init__(self, arrow_path: str, tokenizer: Tokenizer):
        source = pa.memory_map(arrow_path, "r")
        reader = pa.ipc.open_file(source)
        self._table = reader.read_all()
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return self._table.num_rows

    def __getitem__(self, idx: int) -> tuple[Audio, Text]:
        row = {name: self._table.column(name)[idx] for name in self._table.column_names}

        sample_rate = row["sample_rate"].as_py()
        loudness = row["loudness"].as_py()
        n_mels = row["n_mels"].as_py()
        n_frames = row["n_frames"].as_py()

        waveform_i16 = torch.frombuffer(
            bytearray(row["waveform_i16"].as_py()), dtype=torch.int16
        )
        waveform = waveform_i16.view(1, -1).float() / 32678.0

        mels = torch.frombuffer(bytearray(row["mel"].as_py()), dtype=torch.float)
        mels = mels.view(1, n_mels, n_frames).float()

        audio = Audio(
            waveform=waveform,
            sample_rate=sample_rate,
            loudness=loudness,
            mels=mels,
        )
        text = Text(text=row["text"].as_py(), tokenizer=self.tokenizer)
        return audio, text
