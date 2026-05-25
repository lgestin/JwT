import jwt


def test_version_exposed() -> None:
    assert isinstance(jwt.__version__, str)
    assert jwt.__version__
