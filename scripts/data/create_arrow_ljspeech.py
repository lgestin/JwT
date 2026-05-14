"""Convert an LJSpeech folder into a single PyArrow IPC file with waveforms,
text, phonemes, tokens, and codec-encoded mels."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from rich.progress import track
from simple_parsing import ArgumentParser

from tts.data.audio.codecs import Codecs
from tts.data.source import LJTTSSource
from tts.data.text import Tokenizer, Vocabulary


@dataclass
class Args:
    lj_folder: Path  # Root folder of the LJSpeech-1.1 dataset.
    vocab_path: Path  # JSON vocabulary produced by create_vocabulary.py.
    output_path: Path  # Destination .arrow file.
    codec: Codecs = Codecs.BIGVGAN
    device: str = "cuda"
    n_workers: int = 8
    chunk_size: int = 64
    target_loudness: float = -24.0


SCHEMA = pa.schema(
    [
        pa.field("audio_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("phonemes", pa.string()),
        pa.field("tokens", pa.list_(pa.int32())),
        pa.field("waveform_i16", pa.binary()),
        pa.field("num_samples", pa.int32()),
        pa.field("sample_rate", pa.int32()),
        pa.field("loudness", pa.float32()),
        pa.field("mel", pa.binary()),
        pa.field("n_mels", pa.int32()),
        pa.field("n_frames", pa.int32()),
    ]
)


def _prepare(idx: int, source: LJTTSSource, target_sr: int, target_loudness: float):
    audio_path, _ = source.items[idx]
    audio_id = audio_path.stem
    audio, text = source[idx]
    audio = audio.mono().resample(target_sr).normalize(target_loudness)
    waveform_i16 = (audio.waveform * 32678.0).clamp(-32768, 32767).to(torch.int16)
    phonemes = text.phonemes
    tokens = source.tokenizer.encode(phonemes)
    return {
        "audio_id": audio_id,
        "text": text.text,
        "phonemes": phonemes,
        "tokens": tokens,
        "waveform_i16": waveform_i16,
        "loudness": float(audio.loudness),
    }


def main(args: Args) -> None:
    tokenizer = Tokenizer(Vocabulary.from_json(str(args.vocab_path)))
    source = LJTTSSource(str(args.lj_folder), tokenizer=tokenizer)
    print(f"Loaded {len(source)} items from {args.lj_folder}")

    device = torch.device(args.device)
    codec = args.codec.codec
    codec = codec.eval().to(device)
    target_sr = codec.sample_rate

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    type_map = {f.name: f.type for f in SCHEMA}

    with (
        pa.OSFile(str(args.output_path), "wb") as sink,
        pa.ipc.new_file(sink, SCHEMA) as writer,
        ThreadPoolExecutor(args.n_workers) as executor,
    ):
        futures = [
            executor.submit(_prepare, i, source, target_sr, args.target_loudness)
            for i in range(len(source))
        ]

        buffer: dict[str, list] = defaultdict(list)
        for future in track(futures, description="Encoding"):
            item = future.result()
            waveform_i16: torch.Tensor = item["waveform_i16"]

            with torch.inference_mode():
                wav_f = waveform_i16.to(device, dtype=torch.float32).div_(32678.0)
                mel = codec.encode(wav_f[None]).squeeze(0).to(torch.float32).cpu()

            n_mels, n_frames = int(mel.shape[-2]), int(mel.shape[-1])
            buffer["audio_id"].append(item["audio_id"])
            buffer["text"].append(item["text"])
            buffer["phonemes"].append(item["phonemes"])
            buffer["tokens"].append(item["tokens"])
            buffer["waveform_i16"].append(
                np.ascontiguousarray(waveform_i16.numpy()).tobytes()
            )
            buffer["num_samples"].append(int(waveform_i16.numel()))
            buffer["sample_rate"].append(int(target_sr))
            buffer["loudness"].append(item["loudness"])
            buffer["mel"].append(np.ascontiguousarray(mel.numpy()).tobytes())
            buffer["n_mels"].append(n_mels)
            buffer["n_frames"].append(n_frames)

            if len(buffer["audio_id"]) >= args.chunk_size:
                _flush(writer, buffer, type_map)
                buffer = defaultdict(list)

        if buffer["audio_id"]:
            _flush(writer, buffer, type_map)

    print(f"Wrote {args.output_path}")


def _flush(writer, buffer: dict[str, list], type_map: dict) -> None:
    arrays = {k: pa.array(v, type=type_map[k]) for k, v in buffer.items()}
    writer.write_batch(pa.record_batch(arrays))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    main(parser.parse_args().args)
