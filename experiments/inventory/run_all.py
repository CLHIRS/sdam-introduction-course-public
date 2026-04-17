from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from helpers.bootstrap import ensure_importable


@dataclass(frozen=True)
class UseCaseNotebook:
    path: Path
    relpath: str
    name: str


def _repo_root() -> Path:
    # experiments/inventory/run_all.py -> repo root is 2 parents up
    return Path(__file__).resolve().parents[2]


def _discover_use_cases() -> List[UseCaseNotebook]:
    folder = Path(__file__).resolve().parent
    patterns = [
        # Current inventory experiment conventions
        "EIC_*.ipynb",
        "EIMR_*.ipynb",
        # Current repo conventions
        "uc_*.ipynb",
        "tc_*.ipynb",
        # Backwards compatibility
        "use_case_*.ipynb",
    ]

    found = []
    for pat in patterns:
        found.extend(p for p in folder.rglob(pat) if ".ipynb_checkpoints" not in p.parts)

    # De-duplicate while preserving determinism
    notebooks = sorted({p.resolve() for p in found}, key=lambda p: str(p.relative_to(folder)).lower())
    out: List[UseCaseNotebook] = []
    for p in notebooks:
        rel = p.relative_to(folder).as_posix()
        display = rel.replace("_", " ")
        out.append(UseCaseNotebook(path=p, relpath=rel, name=display))
    return out


def _selected(items: Sequence[UseCaseNotebook], *, only: Sequence[str], skip: Sequence[str]) -> List[UseCaseNotebook]:
    only_list = [o for o in only if o]
    skip_set = {s for s in skip if s}

    def _is_selected(u: UseCaseNotebook) -> bool:
        if u.path.name in skip_set or u.relpath in skip_set or u.name in skip_set:
            return False
        if not only_list:
            return True
        return (u.path.name in only_list) or (u.relpath in only_list) or (u.name in only_list)

    selected = [u for u in items if _is_selected(u)]
    if not selected:
        raise RuntimeError("No use cases selected. Check --only/--skip values.")
    return selected


def _as_source_lines(source) -> List[str]:
    if isinstance(source, str):
        return source.splitlines(True)
    if isinstance(source, list) and all(isinstance(x, str) for x in source):
        return list(source)
    raise TypeError(f"Unexpected cell.source type: {type(source).__name__}")


def validate_notebook_json(path: Path, *, strict: bool) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:  # pragma: no cover
        return False, [f"Failed to read: {e}"]

    if not text.strip():
        return False, ["File is empty"]

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]

    if not isinstance(obj, dict):
        return False, [f"Top-level JSON must be an object; got {type(obj).__name__}"]

    nbformat = obj.get("nbformat")
    if nbformat is None:
        if strict:
            errors.append("Missing nbformat")
    elif int(nbformat) != 4:
        errors.append(f"Unsupported nbformat={nbformat} (expected 4)")

    cells = obj.get("cells")
    if not isinstance(cells, list):
        errors.append("Missing or invalid 'cells' list")
        return False, errors

    if strict and len(cells) == 0:
        errors.append("No cells")

    for i, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"Cell {i}: must be an object")
            continue
        ctype = cell.get("cell_type")
        if ctype not in {"markdown", "code", "raw"}:
            errors.append(f"Cell {i}: unknown cell_type={ctype!r}")
        if "source" not in cell:
            errors.append(f"Cell {i}: missing source")
            continue
        try:
            _ = _as_source_lines(cell.get("source"))
        except Exception as e:
            errors.append(f"Cell {i}: invalid source: {e}")

        if strict and ctype == "code":
            if "outputs" not in cell:
                errors.append(f"Cell {i}: code cell missing outputs")
            if "execution_count" not in cell:
                errors.append(f"Cell {i}: code cell missing execution_count")

    return (len(errors) == 0), errors


def _import_smoke() -> None:
    ensure_importable("inventory")
    import inventory  # noqa: F401
    from inventory.core.dynamics import DynamicSystemMVP  # noqa: F401
    from inventory.evaluation import make_eval_info, make_train_info  # noqa: F401


def run_all(
    *,
    only: Optional[Sequence[str]] = None,
    skip: Optional[Sequence[str]] = None,
    strict: bool = False,
    import_smoke: bool = True,
    verbose: bool = False,
) -> None:
    use_cases = _discover_use_cases()
    selected = _selected(use_cases, only=list(only or []), skip=list(skip or []))

    print("\n=== Validating inventory use cases ===")
    print(f"use_cases={len(use_cases)} selected={len(selected)} strict={strict} import_smoke={import_smoke}\n")

    failures: List[str] = []

    if import_smoke:
        try:
            _import_smoke()
            print("Import smoke: OK")
        except Exception as e:
            failures.append(f"Import smoke failed: {e}")

    for u in selected:
        ok, errs = validate_notebook_json(u.path, strict=strict)
        status = "OK" if ok else "FAIL"
        print(f"- {status}: {u.relpath}")
        if errs and (verbose or not ok):
            for msg in errs:
                print(f"    - {msg}")
        if not ok:
            failures.append(f"{u.relpath}: {len(errs)} issues")

    if failures:
        raise RuntimeError("Use-case validation failed:\n" + "\n".join(failures))

    print("\n=== Done ===")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Validate inventory use-case notebooks (no execution).")
    p.add_argument("--only", action="append", default=[], help="Run only a specific use case (repeatable): filename or display name")
    p.add_argument("--skip", action="append", default=[], help="Skip a use case (repeatable): filename or display name")
    p.add_argument("--strict", action="store_true", help="Require nbformat=4 and code-cell outputs/execution_count fields.")
    p.add_argument("--no-import-smoke", action="store_true", help="Skip inventory import smoke test.")
    p.add_argument("--verbose", action="store_true", help="Print all validation warnings/errors.")
    args = p.parse_args()

    run_all(
        only=args.only,
        skip=args.skip,
        strict=bool(args.strict),
        import_smoke=(not args.no_import_smoke),
        verbose=bool(args.verbose),
    )
