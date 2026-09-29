# JwT style guide

## Principle: KISS
- The simplest thing that works. No speculative abstraction, unused options, or defensive branches for cases that can't happen.
- Straight-line code over clever indirection. Minimal, neat diffs in the existing style; don't refactor beyond the task.
- Add a config field or CLI flag only when a real experiment needs it; otherwise fix a sensible default.
- When the simple choice isn't obvious, ask or propose before building the general version.

## Tooling (enforced by CI)
- Python 3.13, modern syntax: `X | None`, `list[str]`, `type X = ...`, `match`, `StrEnum`.
- `ruff check` and `ruff format` (line length 88; rules E, F, I, B, UP, SIM, RUF).
- `ty check` on `src/`. Suppress with a targeted `# ty: ignore[rule]`, never a blanket ignore.
- Run everything through `uv run`.

## Layout
- `src/jwt/{data,model,training}`: `model` is architecture, `training` is trainer/logging/metrics. `scripts/` holds thin entrypoints, `tests/` is flat, `notebooks/` and `docs/` are excluded from lint.
- Small, single-purpose files.
- Module docstring: one line, or none. Longer only when it explains real rules (see `training/config.py`).

## Imports
- Order: stdlib, third-party, then `jwt.*`, blank line between groups.
- Absolute `jwt.` imports only, never relative.
- Import directly: no `try/except ImportError` fallbacks. Dependencies are declared in `pyproject.toml`; a missing one should fail at import.
- Use `from torch import nn` (not `import torch.nn as nn`).

## Errors: no `try/except` by default
- Let exceptions propagate. Use `try/except` only when very necessary: a specific recovery to perform or a resource to release.
- Prefer `finally` or a context manager for cleanup (`EMA.swapped` is the model).
- When necessary: catch the narrowest exception type (never bare `except` or `except Exception`), keep the `try` body to the one call that can fail, and add a one-line comment saying why.
- Prefer validating up front (`__post_init__`, explicit `if ... raise`) over catching afterwards. Prefer library flags that return errors as data (`pesq.py`'s `on_error=`) over wrapping calls.
- Raise specific exceptions: `ValueError` for bad config, `TypeError` for wrong types, `NotImplementedError` for unsupported options. Messages say what was received and why the rule exists.
- Bare `assert` only for internal type narrowing or shape invariants.

## Naming and structure
- **No `_`-prefixed names**: functions, methods, variables and attributes alike (`self.batches`, not `self._batches`). A name never signals "private". This includes test helpers. A bare `_` as a throwaway name is fine.
  - When a `@property` needs backing storage, give the stored attribute its own descriptive name rather than the property's name with a `_`.
  - If a function belongs to the class (it uses its state, or is conceptually part of what the class does), keep it in the class as a public method, `@staticmethod` or `@classmethod`. Don't move it out just to avoid a `_`.
  - Only a helper that is genuinely independent of the class (general-purpose, or shared with other code) becomes a plain module-level function in the same file.
- Use `@property` for cheap, side-effect-free, derived, argument-less values instead of `get_x()` methods or stored duplicates. Anything costly, mutating, or taking arguments stays a method with a verb name. Use `functools.cached_property` only for expensive, stable values. Don't make something a property merely because it has no arguments; it has to read like an attribute.
- Shape names are capitalised (`B, H, T, D`); other locals are lowercase.
- Small pure functions and straight-line code. Nested helpers are fine when used once.

## Types
- Annotate every function signature, including `-> None`.
- Tensor shapes go in comments or docstrings: `# (B, H, T, D)`.
- Use `Protocol` for pluggable interfaces; concrete classes subclass it explicitly.

## Config
- Configs are plain `@dataclass`es with defaults; nested ones use `field(default_factory=...)`.
- Config enums (`Codecs`, `FlowParametrizations`, `AttentionImplementations`, `TimestepSchedules`) map a string value to a class through a property. No string dispatch.
- Cross-field invariants are checked in `__post_init__` and raise `ValueError`.
- Section comments group `Args` fields (`# Data`, `# Run`); a one-line trailing comment explains a non-obvious default.

## Comments and docstrings
- One-line docstrings. Extra lines only for a real gotcha: an invariant, a perf trade-off, a hardware constraint.
- Public classes and methods get a docstring; trivial ones and helpers don't need one.
- Reference code in single backticks (`` `foo` ``), not double.
- Comments explain why, not what. Design rationale sits next to the field or check it justifies. No TODO/FIXME in `src/`.

## Performance
- `torch.no_grad` / `inference_mode` decorators; `_foreach_*` ops in hot paths.
- Mutate parameters in place so `torch.compile` doesn't recompile.
- `zip(..., strict=True)`.
- Comprehensions only when simple and readable (one loop, at most one short condition). Otherwise use a plain `for` loop or `map`/`filter`.
- Don't bundle logging-cadence or behaviour changes into perf commits.
- `print` for one-off status in scripts; the `rich`-based loggers for anything recurring.

## Tests
- `tests/test_<module>.py`, one per source module.
- `test_<behaviour>() -> None` with a one-line docstring stating the expected behaviour.
- Plain `assert` with `pytest.approx` / `torch.allclose`; `pytest.raises(..., match=...)`, combined with other context managers in one `with`.
- Tiny synthetic models and tensors, no real data. Arrange, act, assert, with a short comment giving the arithmetic behind the expected value.
- Cover edge and failure paths (e.g. state restored after an exception). Test through the public interface.

## Git
- Conventional Commits: `feat|fix|refactor|build|chore(scope): lowercase imperative summary`. Scopes are module or topic names (`model`, `logging`, `probe`, `metrics`, `audio`, `deps`, `configs`).
- The subject states the effect or the why, not just the file touched.
- Never stage `docs/superpowers/` or other scratch files.
- Pull with rebase (`git pull --rebase`).
- Merge branches fast-forward only (`git merge --ff-only`); rebase the branch onto the target first if needed. No merge commits.
- Before merging, review the branch's commits. Each commit must be one coherent, logical change: a change is never split across commits, and a commit never mixes unrelated changes. Reorganize (split, squash, reorder) until that holds.
