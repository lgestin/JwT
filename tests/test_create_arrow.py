import importlib.util
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pytest
import soundfile as sf
import torch

from jwt.data.prepared import read_meta, shard_paths

ASSETS_FOLDER = Path(__file__).parent / "assets"
SCRIPT = Path(__file__).parent.parent / "scripts" / "data" / "create_arrow.py"


def load_script():
    spec = importlib.util.spec_from_file_location("create_arrow", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["create_arrow"] = module
    spec.loader.exec_module(module)
    return module


def lj_folder(tmp_path: Path, n_wavs: int, metadata: str) -> Path:
    lj = tmp_path / "lj"
    (lj / "wavs").mkdir(parents=True)
    wav, sr = sf.read(ASSETS_FOLDER / "physicsworks.wav", dtype="int16")
    for i in range(n_wavs):
        clip = wav[i * 3 * sr : (i + 1) * 3 * sr]
        sf.write(lj / "wavs" / f"LJ001-000{i + 1}.wav", clip, sr)
    (lj / "metadata.csv").write_text(metadata, encoding="utf-8")
    return lj


def run_create_arrow(script, lj: Path, out: Path):
    return script.main(
        script.Args(
            dataset="ljspeech",
            output_dir=out,
            sample_rate=16000,
            source_path=lj,
            device="cuda" if torch.cuda.is_available() else "cpu",
            n_workers=1,
        )
    )


def test_create_arrow_ljspeech_end_to_end(tmp_path) -> None:
    """LJSpeech clips become aligned rows with speaker, session and meta.json."""
    metadata = "LJ001-0001|r|Physics works.\nLJ001-0002|r| It works in 1905.\n"
    lj = lj_folder(tmp_path, 2, metadata)
    out = tmp_path / "out"
    drops = run_create_arrow(load_script(), lj, out)
    assert sum(drops.values()) == 0
    meta = read_meta(out)
    assert (meta.sample_rate, meta.aligner) == (16000, "MMS_FA")
    table = pa.concat_tables(
        pa.ipc.open_file(pa.memory_map(str(p))).read_all() for p in shard_paths(out)
    )
    assert table["speaker"].to_pylist() == ["ljspeech/lj"] * 2
    assert table["session_idx"].to_pylist() == [1, 2]
    for i in range(2):
        n_words = len(table["word_start"][i].as_py())
        assert n_words == len(table["word_phoneme_start"][i].as_py()) > 0
        assert n_words == len(table["word_text_start"][i].as_py())
    assert table["text"][1].as_py() == "It works in 1905."
    assert "tokens" not in table.column_names
    assert table["num_samples"][0].as_py() == 3 * 16000


def test_incomplete_data_is_reported(tmp_path) -> None:
    """Dropped utterances raise a warning and are recorded in meta.json."""
    metadata = "LJ001-0001|r|Physics works.\nLJ001-0002|r|Missing wav.\n"
    lj = lj_folder(tmp_path, 1, metadata)
    out = tmp_path / "out"
    with pytest.warns(UserWarning, match="incomplete"):
        drops = run_create_arrow(load_script(), lj, out)
    assert sum(drops.values()) == 1
    assert read_meta(out).dropped == dict(drops)


def test_missing_dataset_arguments_raise(tmp_path) -> None:
    """Each dataset reader needs its own paths; a missing one fails up front."""
    script = load_script()
    args = script.Args(dataset="ljspeech", output_dir=tmp_path, sample_rate=16000)
    with pytest.raises(ValueError, match="source_path"):
        script.utterances(args, Counter())


def test_bounded_map_keeps_a_bounded_window_in_flight() -> None:
    """Prep must not queue the whole dataset: at most `window` items run ahead."""
    script = load_script()
    consumed = []

    def items():
        for i in range(20):
            consumed.append(i)
            yield i

    results = []
    with ThreadPoolExecutor(2) as executor:
        for future in script.bounded_map(executor, lambda x: x * 2, items(), window=3):
            assert len(consumed) - len(results) <= 3
            results.append(future.result())
    assert results == [2 * i for i in range(20)]
