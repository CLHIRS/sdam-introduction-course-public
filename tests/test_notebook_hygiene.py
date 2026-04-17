from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple


def _iter_student_notebooks(repo_root: Path) -> Iterable[Path]:
    # Only scan the student-facing top-level notebooks.
    # Intentionally excludes historical/teaching/archive subfolders.
    for p in sorted((repo_root / "lectures").glob("*.ipynb")):
        yield p
    for p in sorted((repo_root / "concepts").glob("*.ipynb")):
        yield p


def _cell_source_text(cell: dict) -> str:
    src = cell.get("source", [])
    if isinstance(src, str):
        return src
    if isinstance(src, list):
        return "".join(src)
    return ""


def test_student_notebooks_have_no_path_hacks_or_inline_installs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebooks = list(_iter_student_notebooks(repo_root))
    assert notebooks, "No notebooks found to scan (unexpected)."

    # Inspect only CODE cells. Mentions in markdown are fine.
    forbidden_checks: List[Tuple[str, re.Pattern[str]]] = [
        ("sys.path insert/append", re.compile(r"\bsys\.path\.(?:insert|append)\b")),
        ("site.addsitedir", re.compile(r"\bsite\.addsitedir\b")),
        ("os.chdir", re.compile(r"\bos\.chdir\s*\(")),
        ("%pip install", re.compile(r"^\s*%pip\s+install\b", flags=re.MULTILINE)),
        ("!pip install", re.compile(r"^\s*!pip\s+install\b", flags=re.MULTILINE)),
        ("!conda install", re.compile(r"^\s*!conda\s+install\b", flags=re.MULTILINE)),
        ("%conda install", re.compile(r"^\s*%conda\s+install\b", flags=re.MULTILINE)),
    ]

    violations: List[str] = []

    for nb_path in notebooks:
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
        except Exception as e:  # pragma: no cover
            raise AssertionError(f"Failed to parse notebook JSON: {nb_path} ({type(e).__name__}: {e})") from e

        for i, cell in enumerate(nb.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue

            text = _cell_source_text(cell)
            for label, pat in forbidden_checks:
                if pat.search(text):
                    violations.append(f"{nb_path.relative_to(repo_root)}: code cell {i}: {label}")

    assert not violations, "Forbidden notebook patterns found:\n" + "\n".join(violations)
