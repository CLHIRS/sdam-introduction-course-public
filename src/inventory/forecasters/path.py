from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from inventory.core.types import State


@runtime_checkable
class DemandPathForecaster(Protocol):
    """Mean+uncertainty demand path forecaster.

    Convention: returns arrays of shape (H,) for times t..t+H-1.

        - `forecast_mean_path` is required.
        - `forecast_std_path` is required.
    """

    def forecast_mean_path(self, state: State, t: int, H: int, info: Optional[dict] = None) -> np.ndarray: ...

    def forecast_std_path(self, state: State, t: int, H: int, info: Optional[dict] = None) -> np.ndarray: ...


@dataclass(frozen=True)
class ConstantMeanPathForecaster:
    mu: float

    def forecast_mean_path(self, state: State, t: int, H: int, info: Optional[dict] = None) -> np.ndarray:
        return np.full(int(H), float(self.mu), dtype=float)

    def forecast_std_path(self, state: State, t: int, H: int, info: Optional[dict] = None) -> np.ndarray:
        mu = float(self.mu)
        return np.full(int(H), float(np.sqrt(max(0.0, mu))), dtype=float)


@dataclass(frozen=True)
class SeasonalSinMeanPathForecaster:
    mu0: float
    amp: float
    period: int

    def forecast_mean_path(self, state: State, t: int, H: int, info: Optional[dict] = None) -> np.ndarray:
        t0 = int(t)
        H = int(H)
        tt = t0 + np.arange(H, dtype=float)
        mu = self.mu0 + self.amp * np.sin(2.0 * np.pi * tt / float(self.period))
        return np.maximum(0.0, mu).astype(float)

    def forecast_std_path(self, state: State, t: int, H: int, info: Optional[dict] = None) -> np.ndarray:
        mu = self.forecast_mean_path(state, t, H, info=info)
        return np.sqrt(np.maximum(0.0, mu))


@dataclass(frozen=True)
class ExogenousMeanPathForecaster:
    """Oracle forecaster using an exogenous model's mean interface (when available).

    Supports:
    - Seasonal Poisson: `exogenous_model.lambda_t(t)`
    - Regime Poisson: `exogenous_model.P`, `exogenous_model.lam_by_regime`, and `regime_index`.

    For regime models, this returns the expected mean path conditioned on current regime in state.
    """

    exogenous_model: object

    def forecast_mean_path(self, state: State, t: int, H: int, info: Optional[dict] = None) -> np.ndarray:
        H = int(H)
        t0 = int(t)
        S = np.asarray(state, dtype=float).reshape(-1)

        if hasattr(self.exogenous_model, "lambda_t"):
            return np.array([float(self.exogenous_model.lambda_t(t0 + k)) for k in range(H)], dtype=float)

        if hasattr(self.exogenous_model, "P") and hasattr(self.exogenous_model, "lam_by_regime"):
            P = np.asarray(getattr(self.exogenous_model, "P"), dtype=float)
            lam = np.asarray(getattr(self.exogenous_model, "lam_by_regime"), dtype=float).reshape(-1)
            if P.ndim != 2 or P.shape[0] != P.shape[1]:
                raise ValueError("exogenous_model.P must be a square matrix")
            if lam.shape[0] != P.shape[0]:
                raise ValueError("exogenous_model.lam_by_regime length must match P")

            regime_index = int(getattr(self.exogenous_model, "regime_index", 1))
            if regime_index < 0 or regime_index >= S.shape[0]:
                raise ValueError("regime_index is out of bounds for state")

            r0 = int(np.round(float(S[regime_index])))
            r0 = int(np.clip(r0, 0, P.shape[0] - 1))

            # Demand at step t uses next regime; start with distribution of next regime.
            pi = P[r0].copy()
            out = np.empty(H, dtype=float)
            for k in range(H):
                out[k] = float(pi @ lam)
                pi = pi @ P
            return out

        raise NotImplementedError("Exogenous model has no supported mean-path interface.")

    def forecast_std_path(self, state: State, t: int, H: int, info: Optional[dict] = None) -> np.ndarray:
        mu = self.forecast_mean_path(state, t, H, info=info)
        return np.sqrt(np.maximum(0.0, mu))


__all__ = [
    "DemandPathForecaster",
    "ConstantMeanPathForecaster",
    "SeasonalSinMeanPathForecaster",
    "ExogenousMeanPathForecaster",
]
