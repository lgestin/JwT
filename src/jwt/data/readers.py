"""Per-dataset readers yielding utterances for scripts/data/create_arrow.py."""

import http.client
import json
import shutil
import urllib.request
import warnings
import zlib
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from jwt.data.audio import AudioFile
from jwt.data.source import LJTTSSource


@dataclass
class Utterance:
    utt_id: str
    speaker: str  # namespaced "<dataset>/<id>"
    session: str | None
    session_idx: int | None
    audio: AudioFile
    text: str


def read_ljspeech(folder: str, sample_rate: int) -> Iterator[Utterance]:
    for path, text in LJTTSSource(folder).items:
        book, clip = path.stem.split("-")
        yield Utterance(
            utt_id=path.stem,
            speaker="ljspeech/lj",
            session=book,
            session_idx=int(clip),
            audio=AudioFile(filepath=str(path), sample_rate=sample_rate),
            text=text,
        )


def select_hifitts2_chapters(chapters_path: Path, hours: float | None) -> list[dict]:
    """Chapters in a stable pseudo-random order (crc32 of path) up to `hours`."""
    order = []
    with open(chapters_path) as f:
        for i, line in enumerate(f):
            chapter = json.loads(line)
            duration = sum(u["duration"] for u in chapter["utterances"])
            order.append(
                (zlib.crc32(chapter["chapter_filepath"].encode()), i, duration)
            )
    keep, total = set(), 0.0
    for _, i, duration in sorted(order):
        if hours is not None and total >= hours * 3600:
            break
        keep.add(i)
        total += duration
    with open(chapters_path) as f:
        return [json.loads(line) for i, line in enumerate(f) if i in keep]


def chapter_mp3(cache_dir: Path, chapter: dict) -> Path:
    """Where a chapter's MP3 is cached: its manifest path with an .mp3 suffix."""
    return Path(cache_dir) / Path(chapter["chapter_filepath"]).with_suffix(".mp3")


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    https = url.replace("http://", "https://", 1)
    with urllib.request.urlopen(https, timeout=60) as response, open(part, "wb") as f:
        shutil.copyfileobj(response, f)
    part.rename(path)


def read_hifitts2(
    manifest_path: Path,
    chapters_path: Path,
    cache_dir: Path,
    sample_rate: int,
    hours: float | None = None,
    drops: Counter[str] | None = None,
) -> Iterator[Utterance]:
    """Utterances of the selected chapters; undownloadable chapters are skipped
    with a warning and their utterances counted in `drops["download"]`."""
    chapters = select_hifitts2_chapters(chapters_path, hours)
    wanted = set()
    for chapter in chapters:
        wanted.update(u["audio_filepath"] for u in chapter["utterances"])
    rows = {}
    with open(manifest_path) as f:
        for line in f:
            row = json.loads(line)
            if row["audio_filepath"] in wanted:
                rows[row["audio_filepath"]] = row
    for chapter in chapters:
        mp3 = chapter_mp3(cache_dir, chapter)
        if not mp3.exists():
            try:
                download(chapter["url"], mp3)
            except (OSError, http.client.HTTPException) as e:  # flaky archive.org
                warnings.warn(f"skipping chapter {chapter['url']}: {e}", stacklevel=2)
                if drops is not None:
                    utts = chapter["utterances"]
                    drops["download"] += sum(u["audio_filepath"] in rows for u in utts)
                continue
        session = Path(chapter["chapter_filepath"]).stem
        for idx, utt in enumerate(chapter["utterances"]):
            row = rows.get(utt["audio_filepath"])
            if row is None:
                continue
            yield Utterance(
                utt_id=Path(utt["audio_filepath"]).stem,
                speaker=f"hifitts2/{row['speaker']}",
                session=session,
                session_idx=idx,
                audio=AudioFile(
                    filepath=str(mp3),
                    sample_rate=sample_rate,
                    start_s=utt["offset"],
                    end_s=utt["offset"] + utt["duration"],
                ),
                text=row["normalized_text"],
            )
