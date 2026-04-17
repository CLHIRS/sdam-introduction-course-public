from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from inventory.core.exogenous import ExogenousModel
from inventory.core.types import Action, Exog, State


def _validate_transition_matrix(P: np.ndarray, name: str) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"{name} must be a square (K,K) matrix. Got shape {P.shape}.")
    if np.any(P < -1e-12):
        raise ValueError(f"{name} has negative entries.")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError(f"Rows of {name} must sum to 1.")
    return P


def _clamp_regime(r: int, K: int) -> int:
    return max(0, min(int(r), int(K) - 1))


def _sample_next_regime(P: np.ndarray, r_t: int, rng: np.random.Generator) -> int:
    probs = P[r_t]
    return int(rng.choice(len(probs), p=probs))


class PoissonConstantDemand(ExogenousModel):
    """Poisson demand with a constant mean.

    Demand model:
      D_{t+1} ~ Poisson(lam)

    Returns W_{t+1} = [D_{t+1}].

    Notes:
    - `lam` is clamped below by `lam_min` before sampling.
    - This class is intentionally simple and does not depend on (state, action, t).
    """

    def __init__(
        self,
        *,
        lam: float = 300.0,
        lam_min: float = 0.0,
        # Backwards-compatible parameter names used in older notebooks.
        lambda0: float | None = None,
        lam0: float | None = None,
    ):
        if lambda0 is not None and lam0 is not None:
            raise ValueError("Specify at most one of lambda0 or lam0.")
        if lambda0 is not None:
            lam = float(lambda0)
        if lam0 is not None:
            lam = float(lam0)

        self.lam = float(lam)
        self.lam_min = float(lam_min)
        if self.lam_min < 0.0:
            raise ValueError("lam_min must be >= 0.")

    def lambda_t(self, t: int) -> float:
        _ = int(t)
        return float(max(self.lam_min, self.lam))

    def sample(self, state: State, action: Action, t: int, rng: np.random.Generator) -> Exog:
        lam_eff = self.lambda_t(t)
        d = rng.poisson(lam_eff)
        return np.array([float(d)], dtype=float)


class PoissonSeasonalDemand(ExogenousModel):
    """Poisson demand with a seasonal mean (plus optional spikes).

    Returns W_{t+1} = [D_{t+1}].

    seasonal mean:
      lam(t) = base + amp*sin(2π t / period) + spike_add*1{t % spike_every == 0, t>0}
    """

    def __init__(
        self,
        *,
        base: float = 300.0,
        lambda0: float | None = None,
        lam0: float | None = None,
        amp: float = 90.0,
        period: int = 20,
        spike_every: int = 50,
        spike_add: float = 120.0,
        lam_min: float = 1.0,
    ):
        # Backwards-compatible parameter names from earlier notebooks/scripts.
        # Canonical name is `base`.
        if lambda0 is not None and lam0 is not None:
            raise ValueError("Specify at most one of lambda0 or lam0.")
        legacy_base = lambda0 if lambda0 is not None else lam0
        if legacy_base is not None:
            base = float(legacy_base)

        self.base = float(base)
        self.amp = float(amp)
        self.period = int(period)
        self.spike_every = int(spike_every)
        self.spike_add = float(spike_add)
        self.lam_min = float(lam_min)

    def lambda_t(self, t: int) -> float:
        t = int(t)
        seasonal = self.base + self.amp * np.sin(2.0 * np.pi * t / float(self.period))
        spike = self.spike_add if (self.spike_every > 0 and (t % self.spike_every == 0) and t > 0) else 0.0
        return float(max(self.lam_min, seasonal + spike))

    def sample(self, state: State, action: Action, t: int, rng: np.random.Generator) -> Exog:
        lam = self.lambda_t(t)
        d = rng.poisson(lam)
        return np.array([float(d)], dtype=float)


class PoissonRegimeDemand(ExogenousModel):
    """Observable regime Markov chain + Poisson demand.

    Regime evolution:
      R_{t+1} ~ P(R_{t+1} | R_t)
      D_{t+1} ~ Poisson(lam[R_{t+1}])

    Returns W_{t+1} = [D_{t+1}, R_{t+1}].

    Assumptions for MVP clarity:
    - current regime is stored in state vector at index `regime_index`
    - regime is in {0..K-1} but stored as float
    """

    def __init__(self, lam_by_regime: Sequence[float], P: np.ndarray, *, regime_index: int = 1):
        self.lam_by_regime = np.array(lam_by_regime, dtype=float)
        self.P = np.asarray(P, dtype=float)
        self.regime_index = int(regime_index)

        K = int(len(self.lam_by_regime))
        if self.P.shape != (K, K):
            raise ValueError(f"P must have shape (K,K) with K={K}. Got {self.P.shape}.")
        if np.any(self.P < -1e-12):
            raise ValueError("P has negative entries.")
        if not np.allclose(self.P.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("Rows of P must sum to 1.")

    def sample(self, state: State, action: Action, t: int, rng: np.random.Generator) -> Exog:
        r_t = int(np.round(float(state[self.regime_index])))
        r_t = max(0, min(r_t, int(len(self.lam_by_regime)) - 1))

        probs = self.P[r_t]
        r_next = int(rng.choice(len(probs), p=probs))

        lam = float(self.lam_by_regime[r_next])
        d = int(rng.poisson(lam))

        return np.array([float(d), float(r_next)], dtype=float)


class PoissonMultiRegimeDemand(ExogenousModel):
    """Observable multi-regime Markov chain + Poisson demand.

    Example regimes:
    - season (slow-moving)
    - day-of-week (fast-moving)
    - weather (fast-moving)

    Demand model:
      λ_eff = λ_base_season[r_season_next] * (1 + coeff_day[r_day_next] + coeff_weather[r_weather_next])
      D ~ Poisson(max(lam_min, λ_eff))

    Returns W_{t+1} = [D_{t+1}, r_season_next, r_day_next, r_weather_next].

    Assumptions:
    - current regimes are stored in the state vector at the indices provided
    - regimes are integer labels stored as floats in the state
    """

    def __init__(
        self,
        *,
        P_season: Optional[np.ndarray] = None,
        lambda_base_season: Optional[Sequence[float]] = None,
        P_day: Optional[np.ndarray] = None,
        lambda_coeff_day: Optional[Sequence[float]] = None,
        P_weather: Optional[np.ndarray] = None,
        lambda_coeff_weather: Optional[Sequence[float]] = None,
        season_index: int = 1,
        day_index: int = 2,
        weather_index: int = 3,
        season_period: int = 90,
        lam_min: float = 1.0,
    ):
        # Defaults match the teaching logic in lecture notebooks.
        if P_season is None:
            P_season = np.array(
                [
                    [0.00, 1.00, 0.00, 0.00],
                    [0.00, 0.00, 1.00, 0.00],
                    [0.00, 0.00, 0.00, 1.00],
                    [1.00, 0.00, 0.00, 0.00],
                ],
                dtype=float,
            )
        if lambda_base_season is None:
            lambda_base_season = [200, 300, 600, 300]
        if P_day is None:
            P_day = np.array(
                [
                    [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
                    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
                    [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                ],
                dtype=float,
            )
        if lambda_coeff_day is None:
            lambda_coeff_day = [-0.33, -0.33, 0.0, 0.0, 0.15, 0.33, 0.33]
        if P_weather is None:
            P_weather = np.array(
                [
                    [0.60, 0.30, 0.10],
                    [0.20, 0.60, 0.20],
                    [0.10, 0.30, 0.60],
                ],
                dtype=float,
            )
        if lambda_coeff_weather is None:
            lambda_coeff_weather = [-0.3, 0.0, 0.3]

        self.P_season = _validate_transition_matrix(np.asarray(P_season, dtype=float), "P_season")
        self.P_day = _validate_transition_matrix(np.asarray(P_day, dtype=float), "P_day")
        self.P_weather = _validate_transition_matrix(np.asarray(P_weather, dtype=float), "P_weather")

        self.lambda_base_season = np.asarray(lambda_base_season, dtype=float)
        self.lambda_coeff_day = np.asarray(lambda_coeff_day, dtype=float)
        self.lambda_coeff_weather = np.asarray(lambda_coeff_weather, dtype=float)

        if self.lambda_base_season.ndim != 1:
            raise ValueError("lambda_base_season must be 1D")
        if self.lambda_coeff_day.ndim != 1:
            raise ValueError("lambda_coeff_day must be 1D")
        if self.lambda_coeff_weather.ndim != 1:
            raise ValueError("lambda_coeff_weather must be 1D")
        if self.P_season.shape[0] != self.lambda_base_season.shape[0]:
            raise ValueError("P_season size must match lambda_base_season length")
        if self.P_day.shape[0] != self.lambda_coeff_day.shape[0]:
            raise ValueError("P_day size must match lambda_coeff_day length")
        if self.P_weather.shape[0] != self.lambda_coeff_weather.shape[0]:
            raise ValueError("P_weather size must match lambda_coeff_weather length")

        self.season_index = int(season_index)
        self.day_index = int(day_index)
        self.weather_index = int(weather_index)
        self.season_period = int(season_period)
        self.lam_min = float(lam_min)

    def lambda_for_regimes(self, season_regime: int, day_regime: int, weather_regime: int) -> float:
        r_season = _clamp_regime(int(season_regime), self.P_season.shape[0])
        r_day = _clamp_regime(int(day_regime), self.P_day.shape[0])
        r_weather = _clamp_regime(int(weather_regime), self.P_weather.shape[0])

        lam_eff = float(self.lambda_base_season[r_season]) * (
            1.0 + float(self.lambda_coeff_day[r_day]) + float(self.lambda_coeff_weather[r_weather])
        )
        return float(max(self.lam_min, lam_eff))

    def sample(self, state: State, action: Action, t: int, rng: np.random.Generator) -> Exog:
        if max(self.season_index, self.day_index, self.weather_index) >= state.shape[0]:
            raise ValueError(
                "State is missing regime components. "
                "Expected indices season/day/weather to be within state."
            )

        r_season_t = _clamp_regime(int(np.round(float(state[self.season_index]))), self.P_season.shape[0])
        r_day_t = _clamp_regime(int(np.round(float(state[self.day_index]))), self.P_day.shape[0])
        r_weather_t = _clamp_regime(int(np.round(float(state[self.weather_index]))), self.P_weather.shape[0])

        if self.season_period > 0 and t >= 0 and ((int(t) + 1) % self.season_period != 0):
            r_season_next = r_season_t
        else:
            r_season_next = _sample_next_regime(self.P_season, r_season_t, rng)

        r_day_next = _sample_next_regime(self.P_day, r_day_t, rng)
        r_weather_next = _sample_next_regime(self.P_weather, r_weather_t, rng)

        lam_eff = self.lambda_for_regimes(r_season_next, r_day_next, r_weather_next)
        d = int(rng.poisson(lam_eff))

        return np.array([float(d), float(r_season_next), float(r_day_next), float(r_weather_next)], dtype=float)


# Backwards-compatible aliases (from minimal_baseline naming)
ExogenousPoissonConstant = PoissonConstantDemand
ExogenousPoissonSeasonal = PoissonSeasonalDemand
ExogenousPoissonRegime = PoissonRegimeDemand
ExogenousPoissonMultiRegime = PoissonMultiRegimeDemand


__all__ = [
    "PoissonConstantDemand",
    "PoissonSeasonalDemand",
    "PoissonRegimeDemand",
    "PoissonMultiRegimeDemand",
    "ExogenousPoissonConstant",
    "ExogenousPoissonSeasonal",
    "ExogenousPoissonRegime",
    "ExogenousPoissonMultiRegime",
]
