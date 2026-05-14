import csv
from pathlib import Path
from typing import Protocol

from tts.data.audio import Audio
from tts.data.text.text import Text
from tts.data.text.tokenizer import Tokenizer


class TTSSource(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[Audio, Text]: ...


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

    def __getitem__(self, idx: int) -> tuple[Audio, Text]:
        audio_path, text = self.items[idx]
        return Audio(filepath=str(audio_path)), Text(
            text=text, tokenizer=self.tokenizer
        )
