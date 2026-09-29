import http.client
import json
from collections import Counter
from pathlib import Path
from typing import Self

import pytest

from jwt.data import readers
from jwt.data.readers import read_hifitts2, read_ljspeech, select_hifitts2_chapters


def test_read_ljspeech_namespaces_speaker_and_sessions(tmp_path: Path) -> None:
    """LJSpeech is one speaker; the book is the session, the clip its position."""
    (tmp_path / "metadata.csv").write_text(
        "LJ001-0002|raw|Second.\nLJ002-0007|raw|Other book.\n", encoding="utf-8"
    )
    utts = list(read_ljspeech(str(tmp_path), sample_rate=24000))
    assert [u.utt_id for u in utts] == ["LJ001-0002", "LJ002-0007"]
    assert {u.speaker for u in utts} == {"ljspeech/lj"}
    assert [(u.session, u.session_idx) for u in utts] == [("LJ001", 2), ("LJ002", 7)]
    assert utts[0].text == "Second."


def utterance(path: str, offset: float, duration: float) -> dict:
    return {"audio_filepath": path, "offset": offset, "duration": duration}


def write_hifitts2(tmp_path: Path) -> tuple[Path, Path]:
    chapters = [
        {
            "url": "http://x/a.mp3",
            "chapter_filepath": "1/9/a.flac",
            "utterances": [
                utterance("1/9/a_0.flac", 1.0, 2.0),
                utterance("1/9/a_1.flac", 3.0, 4.0),
            ],
        },
        {
            "url": "http://x/b.mp3",
            "chapter_filepath": "2/9/b.flac",
            "utterances": [utterance("2/9/b_0.flac", 0.0, 5.0)],
        },
    ]
    manifest = [
        {"audio_filepath": "1/9/a_0.flac", "speaker": "1", "normalized_text": "A0."},
        {"audio_filepath": "1/9/a_1.flac", "speaker": "1", "normalized_text": "A1."},
        {"audio_filepath": "2/9/b_0.flac", "speaker": "2", "normalized_text": "B0."},
    ]
    chapters_path, manifest_path = (
        tmp_path / "chapters.json",
        tmp_path / "manifest.json",
    )
    chapters_path.write_text("\n".join(json.dumps(c) for c in chapters))
    manifest_path.write_text("\n".join(json.dumps(m) for m in manifest))
    return manifest_path, chapters_path


def cache_mp3s(tmp_path: Path, *names: str) -> Path:
    cache = tmp_path / "cache"
    for name in names:
        (cache / name).parent.mkdir(parents=True, exist_ok=True)
        (cache / name).touch()
    return cache


def test_read_hifitts2_fields_from_cached_chapters(tmp_path: Path) -> None:
    """Utterances come from cached chapter MP3s at their manifest offsets."""
    manifest, chapters = write_hifitts2(tmp_path)
    cache = cache_mp3s(tmp_path, "1/9/a.mp3", "2/9/b.mp3")
    utts = {u.utt_id: u for u in read_hifitts2(manifest, chapters, cache, 24000)}
    a1 = utts["a_1"]
    assert a1.speaker == "hifitts2/1"
    assert (a1.session, a1.session_idx) == ("a", 1)
    assert (a1.audio.start_s, a1.audio.end_s) == (3.0, 7.0)
    assert a1.text == "A1."
    assert utts["b_0"].speaker == "hifitts2/2"


def test_select_hifitts2_chapters_stops_at_hours(tmp_path: Path) -> None:
    """Chapter selection stops once the requested hours are reached."""
    _, chapters = write_hifitts2(tmp_path)
    assert len(select_hifitts2_chapters(chapters, hours=None)) == 2
    assert len(select_hifitts2_chapters(chapters, hours=1e-6)) == 1


def test_read_hifitts2_skips_undownloadable_chapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed download warns and counts the chapter's utterances as dropped."""
    manifest, chapters = write_hifitts2(tmp_path)
    cache = cache_mp3s(tmp_path, "2/9/b.mp3")

    def fail(url: str, path: Path) -> None:
        raise OSError("404")

    monkeypatch.setattr(readers, "download", fail)
    drops: Counter[str] = Counter()
    with pytest.warns(UserWarning, match="a.mp3"):
        utts = list(read_hifitts2(manifest, chapters, cache, 24000, drops=drops))
    assert [u.utt_id for u in utts] == ["b_0"]
    assert drops == {"download": 2}  # both utterances of the skipped chapter


def test_download_failures_skip_the_chapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Downloads time out, and a truncated response skips the chapter, not the run."""
    manifest, chapters = write_hifitts2(tmp_path)
    cache = cache_mp3s(tmp_path, "2/9/b.mp3")
    timeouts = []

    class Truncated:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def read(self, *args: object) -> bytes:
            raise http.client.IncompleteRead(b"")

    def urlopen(url: str, timeout: float | None = None) -> Truncated:
        timeouts.append(timeout)
        return Truncated()

    monkeypatch.setattr(readers.urllib.request, "urlopen", urlopen)
    with pytest.warns(UserWarning):
        utts = list(read_hifitts2(manifest, chapters, cache, 24000))
    assert [u.utt_id for u in utts] == ["b_0"]
    assert timeouts and all(t is not None for t in timeouts)
