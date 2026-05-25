import pyarrow as pa
import pytest

from jwt.data.source import ArrowTTSSource, LJTTSSource


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


def test_arrow_source_sample_rate_rejects_empty_file(tmp_path) -> None:
    """An empty arrow file has no rows to read a sample rate from — fail with a
    clear error rather than an opaque IndexError."""
    path = tmp_path / "empty.arrow"
    _write_arrow(path, sample_rate=16000, n=0)
    source = ArrowTTSSource(str(path), tokenizer=None, codec_name="rawaudio256")
    with pytest.raises(ValueError, match="empty"):
        _ = source.sample_rate


def test_lj_source_works_without_a_tokenizer(tmp_path) -> None:
    """create_vocabulary.py builds the phoneme vocabulary the tokenizer needs,
    so it must read an LJTTSSource before any tokenizer can exist — construction
    and item access must not require one."""
    (tmp_path / "metadata.csv").write_text(
        "LJ001-0001|raw text|normalized text\n", encoding="utf-8"
    )
    source = LJTTSSource(str(tmp_path))
    assert len(source) == 1
    _, text = source[0]
    assert text.tokenizer is None
    assert text.text == "normalized text"
