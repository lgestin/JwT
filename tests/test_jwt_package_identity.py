from pathlib import Path

import jwt


def test_jwt_is_local_package():
    here = Path(__file__).resolve().parents[1] / "src" / "jwt"
    assert Path(jwt.__file__).resolve().is_relative_to(here), (
        f"jwt resolved to {jwt.__file__} — PyJWT may have shadowed the local package"
    )
