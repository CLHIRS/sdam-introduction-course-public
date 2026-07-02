from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np

from inventory.forecasters.base import DemandForecaster


@dataclass(frozen=True)
class DecisionContext:
    state: np.ndarray
    t: int
    info: Optional[dict] = None
    system: object | None = None


class PolicyFeatureAdapter(Protocol):
    def features(self, context: DecisionContext) -> np.ndarray: ...

    def feature_names(self) -> list[str]: ...

    def feature_dim(self) -> int: ...

    def describe(self) -> str: ...


def _state_block(context: DecisionContext, raw_state_dim: int) -> np.ndarray:
    state = np.asarray(context.state, dtype=np.float32).reshape(-1)
    if raw_state_dim <= 0:
        raise ValueError("raw_state_dim must be >= 1")
    if state.shape[0] < raw_state_dim:
        raise ValueError(f"State has dim {state.shape[0]} but expected at least {raw_state_dim}.")
    return state[:raw_state_dim].copy()


def _state_feature_names(raw_state_dim: int, state_feature_names: Optional[Sequence[str]]) -> list[str]:
    if state_feature_names is None:
        return [f"state_{i}" for i in range(raw_state_dim)]
    names = [str(name) for name in state_feature_names]
    if len(names) != raw_state_dim:
        raise ValueError(f"Expected {raw_state_dim} state feature names but got {len(names)}.")
    return names


@dataclass(frozen=True)
class RawStateFeatureAdapter:
    raw_state_dim: int
    state_feature_names: Optional[Sequence[str]] = None

    def features(self, context: DecisionContext) -> np.ndarray:
        return _state_block(context, self.raw_state_dim)

    def feature_names(self) -> list[str]:
        return _state_feature_names(self.raw_state_dim, self.state_feature_names)

    def feature_dim(self) -> int:
        return int(self.raw_state_dim)

    def describe(self) -> str:
        return f"Raw observed state: first {self.raw_state_dim} coordinates"


@dataclass(frozen=True)
class LastDemandFeatureAdapter:
    raw_state_dim: int
    demand_scale: float = 100.0
    include_raw_state: bool = True
    state_feature_names: Optional[Sequence[str]] = None
    last_demand_feature_name: str = "last_demand_scaled"

    def __post_init__(self) -> None:
        if self.demand_scale <= 0.0:
            raise ValueError("demand_scale must be > 0")

    def features(self, context: DecisionContext) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.include_raw_state:
            parts.append(_state_block(context, self.raw_state_dim))
        info = context.info or {}
        last_demand = float(info.get("last_demand", 0.0)) / float(self.demand_scale)
        parts.append(np.asarray([last_demand], dtype=np.float32))
        return np.concatenate(parts, axis=0)

    def feature_names(self) -> list[str]:
        names: list[str] = []
        if self.include_raw_state:
            names.extend(_state_feature_names(self.raw_state_dim, self.state_feature_names))
        names.append(self.last_demand_feature_name)
        return names

    def feature_dim(self) -> int:
        return int((self.raw_state_dim if self.include_raw_state else 0) + 1)

    def describe(self) -> str:
        prefix = "Raw state + " if self.include_raw_state else ""
        return f"{prefix}last observed demand scaled by {self.demand_scale:g}"


@dataclass(frozen=True)
class ForecastPathFeatureAdapter:
    forecaster: DemandForecaster
    raw_state_dim: int
    horizon: int = 3
    demand_scale: float = 100.0
    include_raw_state: bool = True
    clip_nonnegative: bool = True
    state_feature_names: Optional[Sequence[str]] = None
    forecast_feature_prefix: str = "forecast_mu_t+"

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be >= 1")
        if self.demand_scale <= 0.0:
            raise ValueError("demand_scale must be > 0")

    def forecast_features(self, context: DecisionContext) -> np.ndarray:
        state = np.asarray(context.state, dtype=np.float32).reshape(-1)
        try:
            mu = self.forecaster.forecast_mean_path(state, int(context.t), self.horizon, info=context.info)
        except TypeError:
            mu = self.forecaster.forecast_mean_path(state, int(context.t), self.horizon)
        mu = np.asarray(mu, dtype=np.float32).reshape(-1)
        if mu.shape[0] != self.horizon:
            raise ValueError(f"Forecaster returned shape {mu.shape}; expected ({self.horizon},).")
        if self.clip_nonnegative:
            mu = np.maximum(mu, 0.0)
        return mu / np.float32(self.demand_scale)

    def features(self, context: DecisionContext) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.include_raw_state:
            parts.append(_state_block(context, self.raw_state_dim))
        parts.append(self.forecast_features(context))
        return np.concatenate(parts, axis=0)

    def feature_names(self) -> list[str]:
        names: list[str] = []
        if self.include_raw_state:
            names.extend(_state_feature_names(self.raw_state_dim, self.state_feature_names))
        names.extend([f"{self.forecast_feature_prefix}{k}" for k in range(1, self.horizon + 1)])
        return names

    def feature_dim(self) -> int:
        return int((self.raw_state_dim if self.include_raw_state else 0) + self.horizon)

    def describe(self) -> str:
        prefix = "raw state + " if self.include_raw_state else ""
        return f"{prefix}{self.horizon}-step forecast mean path scaled by {self.demand_scale:g}"


@dataclass(frozen=True)
class DemandHistoryAugmentedFeatureAdapter:
    raw_state_dim: int
    history_len: int = 3
    demand_scale: float = 100.0
    include_raw_state: bool = True
    state_feature_names: Optional[Sequence[str]] = None
    history_feature_prefix: str = "demand_history_"

    def __post_init__(self) -> None:
        if self.history_len <= 0:
            raise ValueError("history_len must be >= 1")
        if self.demand_scale <= 0.0:
            raise ValueError("demand_scale must be > 0")

    def history_features(self, context: DecisionContext) -> np.ndarray:
        info = context.info or {}
        raw_history = np.asarray(info.get("demand_history", np.empty(0, dtype=np.float32)), dtype=np.float32).reshape(-1)
        if raw_history.shape[0] >= self.history_len:
            hist = raw_history[-self.history_len :]
        else:
            hist = np.zeros(self.history_len, dtype=np.float32)
            if raw_history.shape[0] > 0:
                hist[-raw_history.shape[0] :] = raw_history
        return hist / np.float32(self.demand_scale)

    def features(self, context: DecisionContext) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.include_raw_state:
            parts.append(_state_block(context, self.raw_state_dim))
        parts.append(self.history_features(context))
        return np.concatenate(parts, axis=0)

    def feature_names(self) -> list[str]:
        names: list[str] = []
        if self.include_raw_state:
            names.extend(_state_feature_names(self.raw_state_dim, self.state_feature_names))
        names.extend([f"{self.history_feature_prefix}{k}" for k in range(self.history_len)])
        return names

    def feature_dim(self) -> int:
        return int((self.raw_state_dim if self.include_raw_state else 0) + self.history_len)

    def describe(self) -> str:
        prefix = "raw state + " if self.include_raw_state else ""
        return f"{prefix}{self.history_len}-step demand history scaled by {self.demand_scale:g}"


__all__ = [
    "DecisionContext",
    "PolicyFeatureAdapter",
    "RawStateFeatureAdapter",
    "LastDemandFeatureAdapter",
    "ForecastPathFeatureAdapter",
    "DemandHistoryAugmentedFeatureAdapter",
]
