import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "data" / "download_hifitts2.py"
UTTERANCE = {"audio_filepath": "u.flac", "offset": 0.0, "duration": 5.0}
CHAPTERS = [
    {
        "url": "http://x/a.mp3",
        "chapter_filepath": "1/9/a.flac",
        "utterances": [UTTERANCE],
    },
    {
        "url": "http://x/b.mp3",
        "chapter_filepath": "2/9/b.flac",
        "utterances": [UTTERANCE],
    },
]


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("download_hifitts2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["download_hifitts2"] = module
    spec.loader.exec_module(module)
    return module


def fake_downloads(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, fail: str | None = None
) -> list[str]:
    """Replace the network: manifests get fixture content, MP3s a few bytes."""
    urls = []

    def download(url: str, path: Path) -> None:
        urls.append(url)
        if url == fail:
            raise OSError("404")
        path.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith("chapters_22khz.json"):
            path.write_text("\n".join(json.dumps(c) for c in CHAPTERS))
        else:
            path.write_bytes(b"data")

    monkeypatch.setattr(script, "download", download)
    return urls


def test_downloads_manifests_and_every_chapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both manifests land in output_dir, chapter MP3s where the reader looks."""
    script = load_script()
    urls = fake_downloads(script, monkeypatch)
    counts = script.main(script.Args(output_dir=tmp_path))
    assert (tmp_path / "manifest_22khz.json").exists()
    assert (tmp_path / "chapters_22khz.json").exists()
    assert (tmp_path / "mp3/1/9/a.mp3").exists() and (
        tmp_path / "mp3/2/9/b.mp3"
    ).exists()
    assert counts == {"downloaded": 2}
    assert len(urls) == 4  # 2 manifests + 2 chapters


def test_cached_files_are_not_downloaded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing manifests and MP3s are skipped, so reruns resume."""
    script = load_script()
    urls = fake_downloads(script, monkeypatch)
    script.main(script.Args(output_dir=tmp_path))
    (tmp_path / "mp3/2/9/b.mp3").unlink()
    urls.clear()
    counts = script.main(script.Args(output_dir=tmp_path))
    assert urls == ["http://x/b.mp3"]
    assert counts == {"cached": 1, "downloaded": 1}


def test_hours_selects_the_same_chapters_as_create_arrow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tiny --hours keeps one chapter, as `select_hifitts2_chapters` does."""
    script = load_script()
    fake_downloads(script, monkeypatch)
    counts = script.main(script.Args(output_dir=tmp_path, hours=1e-6))
    assert counts == {"downloaded": 1}


def test_failed_chapters_are_counted_and_warned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed chapter doesn't stop the others; the summary warns about it."""
    script = load_script()
    fake_downloads(script, monkeypatch, fail="http://x/a.mp3")
    with pytest.warns(UserWarning, match="1 chapter"):
        counts = script.main(script.Args(output_dir=tmp_path))
    assert counts == {"downloaded": 1, "failed": 1}


def test_unknown_rate_raises(tmp_path: Path) -> None:
    """Only the two rates HiFiTTS-2 publishes are accepted."""
    script = load_script()
    with pytest.raises(ValueError, match="48khz"):
        script.main(script.Args(output_dir=tmp_path, rate="48khz"))
