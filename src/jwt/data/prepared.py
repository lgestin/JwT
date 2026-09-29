"""Prepared dataset on disk: arrow shards of int16 waveforms + meta.json."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow as pa

META_FILE = "meta.json"


@dataclass
class PreparedMeta:
    sample_rate: int
    target_loudness: float
    aligner: str  # `Aligners` name that produced the word timings
    dropped: dict[str, int] = field(default_factory=dict)  # reason -> utterances


def write_meta(folder: Path, meta: PreparedMeta) -> None:
    (Path(folder) / META_FILE).write_text(json.dumps(asdict(meta), indent=2))


def read_meta(folder: Path) -> PreparedMeta:
    return PreparedMeta(**json.loads((Path(folder) / META_FILE).read_text()))


def shard_paths(folder: Path) -> list[Path]:
    return sorted(Path(folder).glob("*.arrow"))


def build_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("utt_id", pa.string()),
            pa.field("dataset", pa.string()),
            pa.field("speaker", pa.string()),
            pa.field("session", pa.string()),
            pa.field("session_idx", pa.int32()),
            pa.field("duration", pa.float32()),
            pa.field("text", pa.string()),
            pa.field("phonemes", pa.string()),
            pa.field("word_start", pa.list_(pa.float32())),
            pa.field("word_end", pa.list_(pa.float32())),
            pa.field("word_phoneme_start", pa.list_(pa.int32())),
            pa.field("word_text_start", pa.list_(pa.int32())),
            pa.field("waveform_i16", pa.binary()),
            pa.field("num_samples", pa.int32()),
            pa.field("sample_rate", pa.int32()),
            pa.field("loudness", pa.float32()),
        ]
    )


class ShardWriter:
    """Writes row dicts into numbered shards of at most `rows_per_shard` rows."""

    def __init__(
        self,
        folder: Path,
        schema: pa.Schema,
        rows_per_shard: int = 10_000,
        batch_size: int = 64,
    ):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.rows_per_shard = rows_per_shard
        self.batch_size = batch_size
        self.n_rows = 0
        self.buffer: list[dict] = []
        self.sink: pa.OSFile | None = None
        self.writer: pa.RecordBatchFileWriter | None = None
        self.n_shards = 0
        self.rows_in_shard = 0

    def write(self, row: dict) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        rows, self.buffer = self.buffer, []
        while rows:
            if self.writer is None or self.rows_in_shard >= self.rows_per_shard:
                self.open_next_shard()
            assert self.writer is not None
            n = min(len(rows), self.rows_per_shard - self.rows_in_shard)
            self.writer.write_batch(
                pa.RecordBatch.from_pylist(rows[:n], schema=self.schema)
            )
            self.rows_in_shard += n
            self.n_rows += n
            rows = rows[n:]

    def open_next_shard(self) -> None:
        self.close_shard()
        path = self.folder / f"{self.n_shards:05d}.arrow"
        self.sink = pa.OSFile(str(path), "wb")
        self.writer = pa.ipc.new_file(self.sink, self.schema)
        self.n_shards += 1
        self.rows_in_shard = 0

    def close_shard(self) -> None:
        if self.writer is not None and self.sink is not None:
            self.writer.close()
            self.sink.close()
        self.writer = self.sink = None

    def close(self) -> None:
        self.flush()
        self.close_shard()

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
