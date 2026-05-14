# PEP-Compliant Project Initialization — Design

**Date:** 2026-05-13
**Status:** Approved (pending spec review)
**Project:** `tts` (publishable Python library, Python 3.13)

## 1. Purpose

Take the current `uv init` scaffold and bring it to a clean, modern,
PEP-aligned baseline suitable for a publishable library. After this work,
the project should:

- Be importable as `tts` after `pip install`.
- Follow PEP 621 metadata, PEP 517 build, PEP 561 typing marker, PEP 440
  versioning.
- Lint, format, type-check, and test under a single tool chain (ruff + ty
  + pytest) wired into CI.

Out of scope: feature code, model code, CLI entry points, PyPI publish
workflow, pre-commit hooks, coverage, multi-version test matrix, docs
site. These can be layered on later.

## 2. Final Layout

```
tts/
├── pyproject.toml
├── README.md
├── LICENSE                    # MIT
├── .gitignore
├── .gitattributes
├── .python-version            # 3.13 (already present)
├── src/
│   └── tts/
│       ├── __init__.py        # exposes __version__
│       └── py.typed           # PEP 561 marker (empty file)
├── tests/
│   ├── __init__.py
│   └── test_smoke.py
├── docs/
│   └── superpowers/specs/...  # already present
└── .github/
    └── workflows/
        └── ci.yml
```

Removed from current scaffold: `main.py` (was `uv init` placeholder, not
part of a library) and the empty `scripts/` directory.

### 2.1 Why src-layout

src-layout (PyPA recommendation) prevents accidental imports of the
package from the working directory. Tests must run against the installed
package, which catches missing-file packaging bugs that flat-layout hides.

### 2.2 Version single-source

`src/tts/__init__.py` defines `__version__ = "0.1.0"`. `pyproject.toml`
points at it via hatchling's `tool.hatch.version` so version lives in one
place.

## 3. `pyproject.toml`

PEP 621 metadata block, hatchling build backend, dependency groups, and
all tool config (ruff, ty, pytest) inline. Single file as source of truth.

### 3.1 Build system

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Hatchling is uv's default backend for libraries, well-supported, no
config-file sprawl.

### 3.2 Project metadata (PEP 621)

```toml
[project]
name = "tts"
description = "..."                 # to be filled by user
readme = "README.md"
license = "MIT"                     # SPDX expression, PEP 639
requires-python = ">=3.13"
authors = [{ name = "Lucas" }]      # email optional
classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.13",
    "License :: OSI Approved :: MIT License",
    "Typing :: Typed",
]
dynamic = ["version"]
dependencies = [
    "torch>=2.12.0",
    "torchaudio>=2.11.0",
]

[project.urls]
# Homepage, Repository — to be filled by user

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "ruff>=0.8",
    "ty",                           # alpha; pin to the latest published version at impl time
]

[tool.hatch.version]
path = "src/tts/__init__.py"

[tool.hatch.build.targets.wheel]
packages = ["src/tts"]
```

Notes:

- `version` is dynamic and read from `__init__.py` (single source).
- `license = "MIT"` uses PEP 639 SPDX expression form; the classifier is
  kept for tooling that still expects it.
- `Typing :: Typed` advertises the `py.typed` marker.

### 3.3 Ruff config

```toml
[tool.ruff]
target-version = "py313"
line-length = 100

[tool.ruff.lint]
select = [
    "E", "F",       # pycodestyle, pyflakes (PEP 8 baseline)
    "I",            # isort (import order)
    "B",            # bugbear (common pitfalls)
    "UP",           # pyupgrade (use modern syntax for py313)
    "SIM",          # simplifications
    "RUF",          # ruff-specific
]

[tool.ruff.format]
# defaults are fine; black-compatible
```

Rationale: this rule selection is the de-facto modern default — strict
enough to catch real issues, loose enough not to fight the user. Ruff's
formatter replaces black and isort.

### 3.4 ty config

```toml
[tool.ty.src]
include = ["src"]

[tool.ty.rules]
# leave at defaults; ty is alpha, strict mode is still evolving
```

Type-checks `src/` only (not tests) to keep the bar reasonable while ty
matures. Tests can opt in later.

### 3.5 pytest config

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

Minimal. `-ra` shows a short summary of all non-pass results.

## 4. Source files

### 4.1 `src/tts/__init__.py`

```python
__version__ = "0.1.0"
```

### 4.2 `src/tts/py.typed`

Empty file. PEP 561 marker — tells type checkers this package ships its
own type information.

### 4.3 `tests/__init__.py`

Empty file (makes `tests/` a package; helps some test runners and IDEs).

### 4.4 `tests/test_smoke.py`

```python
import tts


def test_version_exposed():
    assert isinstance(tts.__version__, str)
    assert tts.__version__
```

One trivial test so the CI green path is real, not theoretical.

## 5. Repository files

### 5.1 `LICENSE`

Standard MIT license text, copyright "Lucas, 2026".

### 5.2 `.gitignore`

Keep the existing entries; add common Python noise that's missing:

```
# additions
.pytest_cache/
.ruff_cache/
.ty_cache/
*.egg-info/
.coverage
htmlcov/
```

### 5.3 `.gitattributes`

```
* text=auto eol=lf
```

Normalizes line endings — small but worth it.

### 5.4 `README.md`

Currently empty. Add a minimal stub: title, one-line description,
install command, quick usage placeholder, license line. Enough for PyPI
to render something; details can grow later.

## 6. CI: `.github/workflows/ci.yml`

Single workflow, triggers on `push` and `pull_request`. Steps:

1. `actions/checkout@v4`
2. `astral-sh/setup-uv@v3` — installs uv, caches its store.
3. `uv python install 3.13`
4. `uv sync --extra dev`
5. `uv run ruff check .`
6. `uv run ruff format --check .`
7. `uv run ty check`
8. `uv run pytest`

One job, runs in order, fails fast. Single Python version (3.13). Matrix
can be added later when the project supports older versions.

## 7. Step Order (for the implementation plan)

1. Create the src-layout directories and move/create source files.
2. Delete `main.py` and the empty `scripts/` directory.
3. Rewrite `pyproject.toml` per Section 3.
4. Add `LICENSE`, expand `.gitignore`, add `.gitattributes`, stub
   `README.md`.
5. Add tests directory and smoke test.
6. Add CI workflow.
7. Run `uv sync --extra dev` locally to refresh the lockfile.
8. Run `ruff check`, `ruff format --check`, `ty check`, `pytest` locally
   — all must pass before completion.
9. Commit.

## 8. Acceptance Criteria

- `uv sync --extra dev` succeeds.
- `uv run ruff check .` — no errors.
- `uv run ruff format --check .` — no diff.
- `uv run ty check` — no errors.
- `uv run pytest` — smoke test passes.
- `uv build` produces a wheel and sdist without errors.
- Installing the built wheel into a fresh venv exposes
  `tts.__version__`.

## 9. Risks / Open Questions

- **ty is alpha.** API and rule names may shift. If a chosen version
  breaks, pin to the last known-good. Acceptable since the user
  explicitly chose ty.
- **License default.** Spec assumes MIT. If user wants Apache-2.0 / BSD /
  proprietary, this is a one-file change.
- **PEP 639 license expression.** Some older tooling may not yet
  understand SPDX `license = "MIT"`. The classifier is kept as a
  compatibility belt-and-suspenders.
- **No author email / URLs.** Left blank; user can fill in before first
  publish.
