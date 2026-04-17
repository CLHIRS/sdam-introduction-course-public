"""Backwards-compatibility shim.

This module was renamed to `inventory_backorders.py`.
Prefer importing from `inventory.problems.inventory_backorders`.
"""

from __future__ import annotations

from inventory.problems.inventory_backorders import *  # noqa: F403

# Re-export public API
from inventory.problems.inventory_backorders import __all__  # noqa: F401
