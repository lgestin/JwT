import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "data" / "create_vocabulary.py"


def load_script():
    spec = importlib.util.spec_from_file_location("create_vocabulary", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["create_vocabulary"] = module
    spec.loader.exec_module(module)
    return module


def test_vocabulary_is_union_of_prepared_phonemes(make_prepared, tmp_path) -> None:
    """The vocabulary holds every phoneme of every given dataset."""
    a = make_prepared("a", [("u", "a/x", None, None, 2)])
    b = make_prepared("b", [("u", "b/x", None, None, 1)])
    script = load_script()
    out = tmp_path / "v.json"
    vocab = script.main(script.Args(dataset_paths=[a, b], output=out))
    assert set(vocab) == {"a", "b", " "}
    assert out.exists()
