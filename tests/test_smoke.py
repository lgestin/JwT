import tts


def test_version_exposed() -> None:
    assert isinstance(tts.__version__, str)
    assert tts.__version__
