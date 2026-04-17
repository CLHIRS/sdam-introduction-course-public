from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from inventory.core.types import State
from inventory.forecasters.base import DemandForecaster
from inventory.problems.demand_models import PoissonConstantDemand, PoissonRegimeDemand, PoissonSeasonalDemand


@dataclass(frozen=True)
class ConstantMeanForecaster(DemandForecaster):
    """Always forecasts a constant mean demand."""

    mean: float

    def forecast_mean_path(self, state: State, t: int, H: int, info: Optional[dict] = None) -> np.ndarray:
        H = int(H)
        return np.full(H, float(self.mean), dtype=float)


@dataclass(frozen=True)
class NaiveForecaster(DemandForecaster):
    """Forecast by repeating the last observed demand across the next H steps.

    The forecaster reads the most recent realized demand from ``info["last_demand"]``.
    If it is absent, ``default_value`` is used.
    """

    default_value: float = 0.0

    def _extract_last_observed_demand(self, info: Optional[dict]) -> float:
        if not info or "last_demand" not in info:
            return float(self.default_value)

        value = np.asarray(info["last_demand"], dtype=float).reshape(-1)
        if value.size >= 1 and np.isfinite(value[-1]):
            return float(max(0.0, value[-1]))
        return float(self.default_value)

    def forecast_mean_path(self, state: State, t: int, H: int, info: Optional[dict] = None) -> np.ndarray:
        last_value = self._extract_last_observed_demand(info)
        return ConstantMeanForecaster(mean=last_value).forecast_mean_path(state, t, H, info=info)


@dataclass(frozen=True)
class RollingMeanForecaster(DemandForecaster):
    """Forecast by repeating the mean of the most recent observed demands.

    The forecaster prefers demand history from ``info["demand_history"]`` and
    averages the last ``window_size`` values. If that history is absent, it falls
    back to ``info["last_demand"]`` as a one-step history. If no usable history is
    present, ``default_value`` is used.
    """

    window_size: int = 3
    default_value: float = 0.0

    def __post_init__(self) -> None:
        if int(self.window_size) <= 0:
            raise ValueError("window_size must be positive")

    def _extract_history(self, info: Optional[dict]) -> np.ndarray:
        if not info:
            return np.zeros(0, dtype=float)

        if "demand_history" in info:
            values = np.asarray(info["demand_history"], dtype=float).reshape(-1)
        elif "last_demand" in info:
            values = np.asarray(info["last_demand"], dtype=float).reshape(-1)
        else:
            return np.zeros(0, dtype=float)

        if values.size == 0:
            return np.zeros(0, dtype=float)

        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.zeros(0, dtype=float)

        return np.maximum(values, 0.0)

    def _rolling_mean(self, info: Optional[dict]) -> float:
        values = self._extract_history(info)
        if values.size == 0:
            return float(max(0.0, self.default_value))

        window = values[-int(self.window_size) :]
        if window.size == 0:
            return float(max(0.0, self.default_value))
        return float(np.mean(window))

    def forecast_mean_path(self, state: State, t: int, H: int, info: Optional[dict] = None) -> np.ndarray:
        mean_value = self._rolling_mean(info)
        return ConstantMeanForecaster(mean=mean_value).forecast_mean_path(state, t, H, info=info)


@dataclass(frozen=True)
class ExogenousAwareMeanForecaster(DemandForecaster):
    """A didactic "expert" forecaster that reads mean structure from an exogenous model.

    - For PoissonConstantDemand: uses lam.
    - For PoissonSeasonalDemand: uses lambda_t.
    - For PoissonRegimeDemand: propagates the Markov chain to get E[lambda_{t+k}].

    This mirrors the expert mean-path logic embedded in your minimal_baseline DLA MILP policy.
    """

    exogenous_model: object

    def forecast_mean_path(self, state: State, t: int, H: int, info: Optional[dict] = None) -> np.ndarray:
        H = int(H)
        exo = self.exogenous_model

        if isinstance(exo, PoissonConstantDemand):
            return np.full(H, float(exo.lam), dtype=float)

        if isinstance(exo, PoissonSeasonalDemand):
            return np.array([float(exo.lambda_t(t + k)) for k in range(H)], dtype=float)

        if isinstance(exo, PoissonRegimeDemand):
            r_t = int(np.round(float(state[exo.regime_index])))
            r_t = max(0, min(r_t, len(exo.lam_by_regime) - 1))
            dist = np.zeros(len(exo.lam_by_regime), dtype=float)
            dist[r_t] = 1.0

            mu = np.empty(H, dtype=float)
            for k in range(H):
                dist = dist @ exo.P
                mu[k] = float(dist @ exo.lam_by_regime)
            return mu

        return np.zeros(H, dtype=float)

# Backwards-compatible class name from lecture_12_b notebook
@dataclass(frozen=True)
class ExpertDemandForecasterConstant350(ConstantMeanForecaster):
    """Expert forecaster that always predicts a constant demand mean (default 350)."""

    mean: float = 350.0


__all__ = [
    "ConstantMeanForecaster",
    "NaiveForecaster",
    "RollingMeanForecaster",
    "ExogenousAwareMeanForecaster",
    "ExpertDemandForecasterConstant350",
]
