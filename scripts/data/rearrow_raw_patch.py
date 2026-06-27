"""Rebuild a raw-audio arrow at a different patch size from an existing one.

Raw-audio "encoding" is a deterministic reshape of the waveform, so a new
`acoustic_rawaudio{P}` column can be derived from the `waveform_i16` already
stored in any raw-audio arrow — no re-download / re-resample needed. Every other
column is copied through unchanged.

Usage:
    uv run python scripts/data/rearrow_raw_patch.py \
        --input_path data/ljspeech_8khz.raw64.arrow \
        --output_path data/ljspeech_8khz.raw256.arrow \
        --patch_size 256
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from rich.progress import track
from simple_parsing import ArgumentParser

from jwt.data.audio.codecs import Codecs


@dataclass
class Args:
    input_path: Path  # Existing raw-audio arrow to source waveforms from.
    output_path: Path  # Destination arrow.
    patch_size: int = 256  # New raw-audio patch size.
    chunk_size: int = 64


_PATCH_TO_CODEC = {
    32: Codecs.RAWAUDIO_32,
    64: Codecs.RAWAUDIO_64,
    128: Codecs.RAWAUDIO_128,
    256: Codecs.RAWAUDIO_256,
}

# Columns copied verbatim from the source arrow.
_PASSTHROUGH = [
    "audio_id",
    "text",
    "phonemes",
    "tokens",
    "waveform_i16",
    "num_samples",
    "sample_rate",
    "loudness",
]


def _build_schema(codec_name: str) -> pa.Schema:
    return pa.schema(
        [
            pa.field("audio_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("phonemes", pa.string()),
            pa.field("tokens", pa.list_(pa.int32())),
            pa.field("waveform_i16", pa.binary()),
            pa.field("num_samples", pa.int32()),
            pa.field("sample_rate", pa.int32()),
            pa.field("loudness", pa.float32()),
            pa.field(f"acoustic_{codec_name}", pa.binary()),
            pa.field("acoustic_dim", pa.int32()),
            pa.field("n_frames", pa.int32()),
        ]
    )


def _flush(writer, buffer: dict[str, list], type_map: dict) -> None:
    arrays = {k: pa.array(v, type=type_map[k]) for k, v in buffer.items()}
    writer.write_batch(pa.record_batch(arrays))


def main(args: Args) -> None:
    codec = _PATCH_TO_CODEC[args.patch_size].codec
    codec_name = str(_PATCH_TO_CODEC[args.patch_size]).lower()
    acoustic_field = f"acoustic_{codec_name}"

    table = pa.ipc.open_file(pa.memory_map(str(args.input_path), "r")).read_all()
    print(f"Loaded {table.num_rows} rows from {args.input_path}")

    schema = _build_schema(codec_name)
    type_map = {f.name: f.type for f in schema}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    cols = {name: table.column(name) for name in _PASSTHROUGH}

    with (
        pa.OSFile(str(args.output_path), "wb") as sink,
        pa.ipc.new_file(sink, schema) as writer,
    ):
        buffer: dict[str, list] = defaultdict(list)
        for i in track(range(table.num_rows), description="Re-patching"):
            wav_i16 = torch.frombuffer(
                bytearray(cols["waveform_i16"][i].as_py()), dtype=torch.int16
            )
            wav_f = wav_i16.view(1, -1).float().div_(32678.0)
            with torch.inference_mode():
                acoustic = codec.encode(wav_f).squeeze(0).to(torch.float32)
            acoustic_dim, n_frames = int(acoustic.shape[-2]), int(acoustic.shape[-1])

            for name in _PASSTHROUGH:
                buffer[name].append(cols[name][i].as_py())
            buffer[acoustic_field].append(np.ascontiguousarray(acoustic.numpy()).tobytes())
            buffer["acoustic_dim"].append(acoustic_dim)
            buffer["n_frames"].append(n_frames)

            if len(buffer["audio_id"]) >= args.chunk_size:
                _flush(writer, buffer, type_map)
                buffer = defaultdict(list)

        if buffer["audio_id"]:
            _flush(writer, buffer, type_map)

    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    main(parser.parse_args().args)
