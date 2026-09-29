import bisect
import csv
import itertools
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import pyarrow as pa
import torch

from jwt.data.audio import Audio, AudioFile
from jwt.data.audio.codecs import RawAudioPatcher
from jwt.data.prepared import read_meta, shard_paths
from jwt.data.text.text import Text
from jwt.data.text.tokenizer import Tokenizer


class TTSSource(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[Audio | AudioFile, Text]: ...


class LJTTSSource(TTSSource):
    def __init__(
        self,
        folder_path: str,
        tokenizer: Tokenizer | None = None,
        sample_rate: int | None = None,
    ) -> None:
        """sample_rate: resample on load (int16, see AudioFile); None keeps native."""
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
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[AudioFile, Text]:
        audio_path, text = self.items[idx]
        audio = AudioFile(filepath=str(audio_path), sample_rate=self.sample_rate)
        return audio, Text(text=text, tokenizer=self.tokenizer)


class Word(NamedTuple):
    """One aligned word: times in seconds, offsets into the text and phonemes."""

    start: float
    end: float
    text_start: int
    phoneme_start: int


class ArrowTTSSource(TTSSource):
    """Reads a prepared dataset directory (see ``jwt.data.prepared``).

    Shards are memory-mapped; light columns stay in memory for splits and audio
    prompt selection, audio is decoded per item and reshaped into `patch_size`
    raw-audio patches. Items are located by bisect over record batches, so
    access stays O(log n) across many shards.
    """

    def __init__(
        self, folder: str, tokenizer: Tokenizer | None, patch_size: int
    ) -> None:
        self.meta = read_meta(Path(folder))
        self.tokenizer = tokenizer
        self.patcher = RawAudioPatcher(patch_size)
        self.batches: list[pa.RecordBatch] = []
        for path in shard_paths(Path(folder)):
            reader = pa.ipc.open_file(pa.memory_map(str(path), "r"))
            self.batches += [
                reader.get_batch(i) for i in range(reader.num_record_batches)
            ]
        if not self.batches:
            raise ValueError(f"no arrow shards in {folder}")
        self.batch_starts = list(
            itertools.accumulate((b.num_rows for b in self.batches), initial=0)
        )
        light = pa.Table.from_batches(self.batches)
        self.utt_ids: list[str] = light["utt_id"].to_pylist()
        self.speakers: list[str] = light["speaker"].to_pylist()
        self.sessions: list[str | None] = light["session"].to_pylist()
        self.session_idxs: list[int | None] = light["session_idx"].to_pylist()
        self.durations: list[float] = light["duration"].to_pylist()

    def __len__(self) -> int:
        return self.batch_starts[-1]

    @property
    def sample_rate(self) -> int:
        return self.meta.sample_rate

    @property
    def hop_length(self) -> int:
        return self.patcher.hop_length

    def cell(self, idx: int, name: str) -> Any:
        b = bisect.bisect_right(self.batch_starts, idx) - 1
        return self.batches[b].column(name)[idx - self.batch_starts[b]].as_py()

    def words(self, idx: int) -> list[Word]:
        """The aligned words of item `idx`, rebuilt from the per-word columns."""
        columns = [self.cell(idx, f"word_{field}") for field in Word._fields]
        return [Word(*values) for values in zip(*columns, strict=True)]

    def __getitem__(self, idx: int) -> tuple[Audio, Text]:
        waveform = torch.frombuffer(
            bytearray(self.cell(idx, "waveform_i16")), dtype=torch.int16
        )
        waveform = waveform.view(1, -1).float() / 32768.0
        audio = Audio(
            waveform=waveform,
            sample_rate=self.cell(idx, "sample_rate"),
            loudness=self.cell(idx, "loudness"),
            acoustic=self.patcher.encode(waveform),
        )
        text = Text(
            text=self.cell(idx, "text"),
            tokenizer=self.tokenizer,
            stored_phonemes=self.cell(idx, "phonemes"),
        )
        return audio, text


def check_same_sample_rate(sources: list[ArrowTTSSource], paths: list[str]) -> None:
    """Raise unless every source has the first source's sample rate."""
    sample_rate = sources[0].sample_rate
    for source, path in zip(sources, paths, strict=True):
        if source.sample_rate != sample_rate:
            raise ValueError(
                f"{path}: sample_rate={source.sample_rate}, expected {sample_rate}"
            )
