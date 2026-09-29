from collections.abc import Callable
from pathlib import Path

import pyarrow as pa

from jwt.data.prepared import (
    PreparedMeta,
    ShardWriter,
    build_schema,
    read_meta,
    shard_paths,
    write_meta,
)


def row(i: int) -> dict:
    return {
        "utt_id": f"u{i}",
        "dataset": "d",
        "speaker": "d/s",
        "session": None,
        "session_idx": None,
        "duration": 1.0,
        "text": "t",
        "phonemes": "p",
        "word_start": [0.0],
        "word_end": [1.0],
        "word_phoneme_start": [0],
        "waveform_i16": b"\x00\x00",
        "num_samples": 1,
        "sample_rate": 100,
        "loudness": -24.0,
    }


def read_table(path: Path) -> pa.Table:
    return pa.ipc.open_file(pa.memory_map(str(path))).read_all()


def test_shard_writer_splits_rows_across_shards(tmp_path: Path) -> None:
    """Rows fill numbered shards of at most rows_per_shard, in order."""
    schema = build_schema()
    with ShardWriter(tmp_path, schema, rows_per_shard=2, batch_size=3) as writer:
        for i in range(5):
            writer.write(row(i))
    paths = shard_paths(tmp_path)
    assert [p.name for p in paths] == ["00000.arrow", "00001.arrow", "00002.arrow"]
    ids = []
    for path in paths:
        ids += read_table(path)["utt_id"].to_pylist()
    assert ids == [f"u{i}" for i in range(5)]
    assert writer.n_rows == 5


def test_meta_round_trips(tmp_path: Path) -> None:
    """meta.json reloads to the same PreparedMeta."""
    meta = PreparedMeta(sample_rate=24000, target_loudness=-24.0, aligner="MMS_FA")
    write_meta(tmp_path, meta)
    assert read_meta(tmp_path) == meta


def test_make_prepared_fixture_geometry(make_prepared: Callable[..., Path]) -> None:
    """The fixture writes 0.5 s words of 'ab' at 100 Hz (5 frames of 10 per word)."""
    folder = make_prepared("d", [("u0", "d/a", "s0", 0, 3)])
    table = read_table(shard_paths(folder)[0])
    assert table["phonemes"][0].as_py() == "ab ab ab"
    assert table["word_phoneme_start"][0].as_py() == [0, 3, 6]
    assert table["num_samples"][0].as_py() == 150
    assert "acoustic_dim" not in table.column_names
    assert read_meta(folder).sample_rate == 100
