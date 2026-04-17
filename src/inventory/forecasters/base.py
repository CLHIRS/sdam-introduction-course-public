from __future__ import annotations

from typing import Optional, Protocol

import numpy as np
from inventory.core.types import State


class DemandForecaster(Protocol):
    """Minimal mean-demand forecaster interface used by DLA/CFA examples.

    Returns a mean demand path of length H for times t..t+H-1 (or t+1..t+H depending on convention).

    Convention (matches your minimal_baseline):
    - `forecast_mean_path(state, t, H)` returns an array of shape (H,) representing
      mean demand for the next H steps starting at t.
    """

    def forecast_mean_path(self, state: State, t: int, H: int, info: Optional[dict] = None) -> np.ndarray:
        ...


__all__ = ["DemandForecaster"]
