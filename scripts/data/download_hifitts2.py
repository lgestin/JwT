"""Download the HiFiTTS-2 manifests and the chapter MP3s of an `--hours` subset."""

import http.client
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from rich.progress import track
from simple_parsing import ArgumentParser

from jwt.data.readers import chapter_mp3, download, select_hifitts2_chapters

HF_URL = "https://huggingface.co/datasets/nvidia/hifitts-2/resolve/main"


@dataclass
class Args:
    output_dir: Path  # Manifests go here, chapter MP3s under output_dir/mp3.
    rate: str = "22khz"  # HiFiTTS-2 subset: "22khz" or "44khz".
    hours: float | None = None  # Same selection as create_arrow.py; None = all.
    n_workers: int = 8  # Parallel MP3 downloads; archive.org throttles more.

    def __post_init__(self) -> None:
        if self.rate not in ("22khz", "44khz"):
            raise ValueError(f"HiFiTTS-2 has 22khz and 44khz subsets, got {self.rate}")


def fetch(chapter: dict, mp3_dir: Path) -> str:
    """Download one chapter's MP3; "failed" when archive.org doesn't deliver it."""
    try:
        download(chapter["url"], chapter_mp3(mp3_dir, chapter))
    except (OSError, http.client.HTTPException):  # flaky archive.org: count, go on
        return "failed"
    return "downloaded"


def main(args: Args) -> Counter[str]:
    manifest, chapters_path = (
        args.output_dir / f"{name}_{args.rate}.json"
        for name in ("manifest", "chapters")
    )
    for path in (manifest, chapters_path):
        if not path.exists():
            print(f"Downloading {path.name}")
            download(f"{HF_URL}/{args.rate}/{path.name}", path)

    mp3_dir = args.output_dir / "mp3"
    chapters = select_hifitts2_chapters(chapters_path, args.hours)
    counts: Counter[str] = Counter()
    missing = []
    for chapter in chapters:
        if chapter_mp3(mp3_dir, chapter).exists():
            counts["cached"] += 1
        else:
            missing.append(chapter)
    with ThreadPoolExecutor(args.n_workers) as executor:
        results = executor.map(partial(fetch, mp3_dir=mp3_dir), missing)
        counts.update(track(results, total=len(missing), description="Downloading"))

    print(f"{len(chapters)} chapters: {dict(counts)}")
    if counts["failed"]:
        warnings.warn(
            f"{counts['failed']} chapter MP3s failed to download; rerun to retry",
            stacklevel=2,
        )
    print(
        "Prepare with: uv run python scripts/data/create_arrow.py --dataset hifitts2 "
        f"--manifest_path {manifest} --chapters_path {chapters_path} "
        f"--cache_dir {mp3_dir} --hours {args.hours} --output_dir <dir> "
        "--sample_rate <rate>"
    )
    return counts


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    main(parser.parse_args().args)
