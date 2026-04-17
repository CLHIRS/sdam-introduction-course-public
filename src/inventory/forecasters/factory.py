from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from inventory.core.types import State
from inventory.forecasters.base import DemandForecaster
from inventory.forecasters.ml import (
    MlDemandForecaster,
    MlRegimeDemandForecaster,
    RegimeFeatureAdapter,
    SeasonalFeatureAdapter,
)
from inventory.problems.demand_models import PoissonRegimeDemand, PoissonSeasonalDemand

Adapter = SeasonalFeatureAdapter | RegimeFeatureAdapter
MlForecaster = MlDemandForecaster | MlRegimeDemandForecaster


@dataclass(frozen=True)
class FitConfig:
    n_samples: int = 6000
    seed: int = 7
    t_start: int = 0
    val_samples: int = 1000
    val_seed: int = 99
    val_t_start: int = 8000


def make_adapter(exogenous_model: object) -> Adapter:
    """Create the default feature adapter for a supported demand model."""

    if isinstance(exogenous_model, PoissonSeasonalDemand):
        return SeasonalFeatureAdapter(exogenous_model)
    if isinstance(exogenous_model, PoissonRegimeDemand):
        return RegimeFeatureAdapter(exogenous_model)

    raise NotImplementedError(
        "No default adapter for this exogenous model type. "
        "Supported: PoissonSeasonalDemand, PoissonRegimeDemand."
    )


def make_ml_forecaster(
    adapter: Adapter,
    *,
    model_type: str = "tree",
    random_state: int = 0,
) -> MlForecaster:
    """Create the default ML forecaster backend for a given adapter."""

    if isinstance(adapter, SeasonalFeatureAdapter):
        return MlDemandForecaster(adapter, model_type=model_type, random_state=random_state)
    if isinstance(adapter, RegimeFeatureAdapter):
        return MlRegimeDemandForecaster(adapter, model_type=model_type, random_state=random_state)

    raise NotImplementedError("No default ML forecaster for this adapter type.")


def fit_ml_forecaster_from_exogenous(
    exogenous_model: object,
    *,
    model_type: str = "tree",
    random_state: int = 0,
    fit: FitConfig | None = None,
    # regime-specific (optional)
    r0: int | None = None,
) -> Tuple[MlForecaster, dict]:
    """One-liner: build adapter + ML forecaster + synthetic fit.

    Returns: (forecaster, fit_report)

    Notes:
    - Seasonal uses `t` features.
    - Regime uses state-dependent features; fitting simulates regimes.
    """

    fit = fit or FitConfig()
    adapter = make_adapter(exogenous_model)
    forecaster = make_ml_forecaster(adapter, model_type=model_type, random_state=random_state)

    if isinstance(adapter, SeasonalFeatureAdapter):
        forecaster.fit_from_exogenous(
            n_samples=fit.n_samples,
            seed=fit.seed,
            t_start=fit.t_start,
            val_samples=fit.val_samples,
            val_seed=fit.val_seed,
            val_t_start=fit.val_t_start,
        )
        return forecaster, dict(forecaster.fit_report)

    if isinstance(adapter, RegimeFeatureAdapter):
        forecaster.fit_from_exogenous(
            n_samples=fit.n_samples,
            seed=fit.seed,
            t_start=fit.t_start,
            val_samples=fit.val_samples,
            val_seed=fit.val_seed,
            val_t_start=fit.val_t_start,
            r0=r0,
        )
        return forecaster, dict(forecaster.fit_report)

    raise AssertionError("Unreachable: unsupported adapter type")


def forecast_with_default_state(
    forecaster: DemandForecaster,
    exogenous_model: object,
    *,
    t: int,
    H: int,
    inventory_level: float = 300.0,
    regime: int = 0,
) -> np.ndarray:
    """Convenience helper for quick interactive notebooks.

    Builds a minimal compatible state vector for seasonal (1D) and regime (2D) models.
    """

    if isinstance(exogenous_model, PoissonRegimeDemand):
        S: State = np.array([float(inventory_level), float(regime)], dtype=float)
    else:
        S = np.array([float(inventory_level)], dtype=float)

    return np.asarray(forecaster.forecast_mean_path(S, int(t), int(H), info={}), dtype=float).reshape(-1)


__all__ = [
    "FitConfig",
    "make_adapter",
    "make_ml_forecaster",
    "fit_ml_forecaster_from_exogenous",
    "forecast_with_default_state",
]
