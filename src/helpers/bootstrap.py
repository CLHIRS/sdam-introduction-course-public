from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional


def find_repo_root(start: Path) -> Path:
    """Best-effort repo root discovery.

    Looks for a `pyproject.toml` marker while walking parents.
    """

    p = start.resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return p


def ensure_src_on_path(*, start: Optional[Path] = None, verbose: bool = False) -> Path | None:
    """Ensure the repo's `src/` is available on `sys.path`.

    Returns the repo root if we injected a path, else None.

    This is intended as a small, centralized fallback for interactive usage.
    In normal usage (after `poetry install`), this should be a no-op.
    """

    repo = find_repo_root(start or Path.cwd())
    src_path = repo / "src"

    if str(src_path) in sys.path:
        return None

    sys.path.insert(0, str(src_path))

    if verbose:
        print("repo:", repo)
        print("src_path:", src_path)

    return repo


def ensure_importable(module: str, *, start: Optional[Path] = None, verbose: bool = False) -> Path | None:
    """Ensure `import module` works, falling back to injecting `src/`.

    Returns the repo root if a path injection occurred, else None.
    """

    try:
        importlib.import_module(module)
        return None
    except ModuleNotFoundError:
        repo = ensure_src_on_path(start=start, verbose=verbose)
        importlib.import_module(module)
        return repo
