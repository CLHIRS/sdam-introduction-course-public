from __future__ import annotations

from pathlib import Path

from helpers.bootstrap import ensure_importable


def ensure_inventory_imports(*, verbose: bool = False) -> Path | None:
	"""Best-effort bootstrap for notebooks.

	If the project is installed (the normal case), this is a no-op.
	If not installed, we attempt to locate the repo root and add `src/` to
	`sys.path` so `import inventory` works.

	Returns the repo root Path if a path injection occurred, else None.
	"""

	# If the package is installed (normal usage), this is a no-op.
	# Otherwise we fall back to injecting `src/`.
	return ensure_importable("inventory", verbose=verbose)
