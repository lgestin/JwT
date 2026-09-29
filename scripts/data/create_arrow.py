"""Prepare a dataset into aligned arrow shards (see `jwt.data.prepared`)."""

import warnings
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import batched
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from rich.progress import track
from simple_parsing import ArgumentParser

from jwt.data.align import Aligners, word_phoneme_starts, word_text_starts
from jwt.data.prepared import PreparedMeta, ShardWriter, build_schema, write_meta
from jwt.data.readers import Utterance, read_hifitts2, read_ljspeech
from jwt.data.text import Text


@dataclass
class Args:
    dataset: str  # "ljspeech" or "hifitts2".
    output_dir: Path  # Prepared dataset directory to create.
    sample_rate: int  # Target rate to resample every clip to (Hz).
    source_path: Path | None = None  # ljspeech: LJSpeech-1.1 folder.
    manifest_path: Path | None = None  # hifitts2: manifest_*.json (JSONL).
    chapters_path: Path | None = None  # hifitts2: chapters_*.json (JSONL).
    cache_dir: Path | None = None  # hifitts2: downloaded chapter MP3s.
    hours: float | None = None  # hifitts2: subset size.
    aligner: Aligners = Aligners.MMS_FA
    device: str = "cuda"  # for alignment
    align_batch_size: int = 16
    n_workers: int = 4  # G2P holds the GIL: more threads starve the aligner
    rows_per_shard: int = 10_000
    target_loudness: float = -24.0


def utterances(args: Args, drops: Counter[str]) -> Iterator[Utterance]:
    if args.dataset == "ljspeech":
        if args.source_path is None:
            raise ValueError("--dataset ljspeech needs --source_path")
        return read_ljspeech(str(args.source_path), args.sample_rate)
    if args.dataset == "hifitts2":
        if not (args.manifest_path and args.chapters_path and args.cache_dir):
            raise ValueError(
                "--dataset hifitts2 needs --manifest_path, --chapters_path and "
                "--cache_dir"
            )
        return read_hifitts2(
            args.manifest_path,
            args.chapters_path,
            args.cache_dir,
            args.sample_rate,
            args.hours,
            drops,
        )
    raise ValueError(f"unknown dataset {args.dataset!r}")


def bounded_map(
    executor: Executor, fn: Callable, items: Iterable, window: int
) -> Iterator[Future]:
    """Futures of `fn` over `items`, in order, with at most `window` in flight."""
    pending: deque[Future] = deque()
    for item in items:
        pending.append(executor.submit(fn, item))
        if len(pending) >= window:
            yield pending.popleft()
    while pending:
        yield pending.popleft()


def prepare(utt: Utterance, target_loudness: float) -> dict | None:
    """Load, normalize, and phonemize one utterance (runs in a thread); None if
    misaki's word tokens don't rebuild its text and phoneme string."""
    audio = utt.audio.mono().normalize(target_loudness)
    waveform_i16 = (audio.waveform * 32768.0).clamp(-32768, 32767).to(torch.int16)
    text = utt.text.strip()
    phonemes, tokens = Text(text).phonemes_tokens
    phoneme_starts = word_phoneme_starts(tokens, phonemes)
    text_starts = word_text_starts(tokens, text)
    if phoneme_starts is None or text_starts is None:
        return None
    return {
        "utt": utt,
        "waveform_i16": waveform_i16,
        "loudness": float(audio.loudness),
        "text": text,
        "phonemes": phonemes,
        "tokens": tokens,
        "word_phoneme_start": phoneme_starts,
        "word_text_start": text_starts,
    }


def prepared_items(futures: Iterable[Future], drops: Counter[str]) -> Iterator[dict]:
    """Results of `prepare`, counting the utterances that failed."""
    for future in futures:
        try:
            item = future.result()
        except sf.SoundFileError:  # a corrupt or missing file mustn't end a long run
            drops["load"] += 1
            continue
        if item is None:
            drops["tokens"] += 1
            continue
        yield item


def arrow_row(args: Args, item: dict, spans: list[tuple[float, float]]) -> dict:
    utt, waveform_i16 = item["utt"], item["waveform_i16"]
    return {
        "utt_id": utt.utt_id,
        "dataset": args.dataset,
        "speaker": utt.speaker,
        "session": utt.session,
        "session_idx": utt.session_idx,
        "duration": waveform_i16.shape[-1] / args.sample_rate,
        "text": item["text"],
        "phonemes": item["phonemes"],
        "word_start": [start for start, _ in spans],
        "word_end": [end for _, end in spans],
        "word_phoneme_start": item["word_phoneme_start"],
        "word_text_start": item["word_text_start"],
        "waveform_i16": np.ascontiguousarray(waveform_i16.numpy()).tobytes(),
        "num_samples": int(waveform_i16.numel()),
        "sample_rate": args.sample_rate,
        "loudness": item["loudness"],
    }


def main(args: Args) -> Counter[str]:
    schema = build_schema()
    aligner = args.aligner.aligner(args.device)
    drops: Counter[str] = Counter()

    with (
        ShardWriter(args.output_dir, schema, args.rows_per_shard) as writer,
        ThreadPoolExecutor(args.n_workers) as executor,
    ):
        prepare_one = partial(prepare, target_loudness=args.target_loudness)
        futures = bounded_map(
            executor, prepare_one, utterances(args, drops), window=4 * args.n_workers
        )
        items = prepared_items(track(futures, description="Preparing"), drops)
        for batch in batched(items, args.align_batch_size, strict=False):
            waveforms = [item["waveform_i16"].float() / 32768.0 for item in batch]
            tokens = [item["tokens"] for item in batch]
            all_spans = aligner.align(waveforms, args.sample_rate, tokens)
            for item, spans in zip(batch, all_spans, strict=True):
                if spans is None:
                    drops["align"] += 1
                else:
                    writer.write(arrow_row(args, item, spans))
    write_meta(
        args.output_dir,
        PreparedMeta(
            sample_rate=args.sample_rate,
            target_loudness=args.target_loudness,
            aligner=str(args.aligner),
            dropped=dict(drops),
        ),
    )
    print(f"Wrote {writer.n_rows} rows to {args.output_dir}")
    if drops:
        warnings.warn(
            f"prepared data is incomplete: dropped {dict(drops)}", stacklevel=2
        )
    return drops


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    main(parser.parse_args().args)
