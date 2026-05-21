import pyarrow as pa

from tts.data.source import ArrowTTSSource


def _write_arrow(path, sample_rate: int, n: int = 3) -> None:
    """Write a minimal arrow file carrying just a sample_rate column."""
    table = pa.table({"sample_rate": pa.array([sample_rate] * n, type=pa.int32())})
    with (
        pa.OSFile(str(path), "wb") as sink,
        pa.ipc.new_file(sink, table.schema) as writer,
    ):
        writer.write_table(table)


def test_arrow_source_exposes_sample_rate(tmp_path) -> None:
    """The training pipeline reads the sample rate from the datafile, not the
    codec — ArrowTTSSource surfaces it from the stored column."""
    path = tmp_path / "data.arrow"
    _write_arrow(path, sample_rate=16000)
    source = ArrowTTSSource(str(path), tokenizer=None, codec_name="rawaudio256")
    assert source.sample_rate == 16000
