from __future__ import annotations

import os


def is_fast() -> bool:
    """Whether scenarios should run in reduced-workload (smoke test) mode."""

    return os.environ.get("SDAM_FAST", "").strip().lower() in {"1", "true", "yes"}


def fast_n_rep(default: int) -> int:
    """Return `3` in fast mode, else the provided default."""

    return 3 if is_fast() else int(default)


def fast_T(default: float, *, fast_value: float = 200.0) -> float:
    """Return a smaller simulation horizon in fast mode.

    The goal is to keep scenario smoke runs quick while preserving the teaching
    defaults in normal mode.
    """

    return float(fast_value) if is_fast() else float(default)
