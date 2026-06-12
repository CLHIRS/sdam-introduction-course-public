from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log
from typing import Dict, Optional, Tuple

import numpy as np


RegimeTuple = Tuple[int, int, int]


@dataclass(frozen=True)
class BeliefMarginals:
    season: np.ndarray
    day: np.ndarray
    weather: np.ndarray


class MultiRegimeBeliefTracker:
    """Exact Bayes filter for hidden season/day/weather regimes.

    The tracker maintains a categorical belief over all hidden regime tuples
    implied by a hidden-state multi-regime demand model. Inventory is assumed
    fully observed and is therefore not part of the belief state.
    """

    def __init__(
        self,
        exo,
        *,
        initial_regimes: Optional[RegimeTuple] = None,
        initial_belief: Optional[np.ndarray] = None,
    ) -> None:
        self._validate_exogenous_model(exo)
        self.exo = exo

        self._n_season = int(self.exo.P_season.shape[0])
        self._n_day = int(self.exo.P_day.shape[0])
        self._n_weather = int(self.exo.P_weather.shape[0])

        self._regime_tuples = [
            (season, day, weather)
            for season in range(self._n_season)
            for day in range(self._n_day)
            for weather in range(self._n_weather)
        ]
        self._index_of: Dict[RegimeTuple, int] = {
            regime: idx for idx, regime in enumerate(self._regime_tuples)
        }
        self._regime_array = np.asarray(self._regime_tuples, dtype=int)
        self._lambda_by_tuple = np.asarray(
            [
                float(self.exo.lambda_for_regimes(season, day, weather))
                for season, day, weather in self._regime_tuples
            ],
            dtype=float,
        )
        self._belief = np.empty(len(self._regime_tuples), dtype=float)
        self.reset(initial_regimes=initial_regimes, initial_belief=initial_belief)

    def reset(
        self,
        *,
        initial_regimes: Optional[RegimeTuple] = None,
        initial_belief: Optional[np.ndarray] = None,
    ) -> None:
        """Reset the belief state to a point mass or a custom prior."""
        if initial_regimes is not None and initial_belief is not None:
            raise ValueError("Specify only one of initial_regimes or initial_belief.")

        if initial_belief is not None:
            self._belief = self._normalize_belief(initial_belief)
            return

        if initial_regimes is None:
            initial_regimes = (
                int(self.exo.initial_season_regime),
                int(self.exo.initial_day_regime),
                int(self.exo.initial_weather_regime),
            )

        season, day, weather = self._validate_regime_tuple(initial_regimes)
        self._belief.fill(0.0)
        self._belief[self._index_of[(season, day, weather)]] = 1.0

    def belief_vector(self) -> np.ndarray:
        """Return a copy of the current belief vector."""
        return self._belief.copy()

    def current_prior(self) -> np.ndarray:
        """Return a copy of the current prior/posterior belief."""
        return self.belief_vector()

    def regime_tuples_array(self) -> np.ndarray:
        """Return the fixed regime-tuple ordering used by the belief vector."""
        return self._regime_array.copy()

    def predict(self, t: int) -> np.ndarray:
        """Apply the hidden regime transition model for one step."""
        t = int(t)
        season_transition = self._season_transition_matrix_for_t(t)
        predicted = np.zeros_like(self._belief)

        for idx_from, prob_from in enumerate(self._belief):
            if prob_from <= 0.0:
                continue
            season_from, day_from, weather_from = self._regime_tuples[idx_from]

            season_probs = season_transition[season_from]
            day_probs = self.exo.P_day[day_from]
            weather_probs = self.exo.P_weather[weather_from]

            for season_to, p_season in enumerate(season_probs):
                if p_season <= 0.0:
                    continue
                for day_to, p_day in enumerate(day_probs):
                    if p_day <= 0.0:
                        continue
                    for weather_to, p_weather in enumerate(weather_probs):
                        if p_weather <= 0.0:
                            continue
                        idx_to = self._index_of[(season_to, day_to, weather_to)]
                        predicted[idx_to] += prob_from * p_season * p_day * p_weather

        self._belief = self._normalize_belief(predicted)
        return self.belief_vector()

    def update(self, observed_demand: float | int) -> np.ndarray:
        """Condition the predicted belief on an observed realized demand."""
        demand = self._validate_observed_demand(observed_demand)
        log_likelihood = np.asarray(
            [self._poisson_log_pmf(demand, lam) for lam in self._lambda_by_tuple],
            dtype=float,
        )
        max_log = float(np.max(log_likelihood))
        weights = np.exp(log_likelihood - max_log)
        posterior = self._belief * weights
        self._belief = self._normalize_belief(posterior)
        return self.belief_vector()

    def observe_step(self, observed_demand: float | int, t: int) -> np.ndarray:
        """Run one full Bayes-filter step: predict then update."""
        self.predict(t)
        return self.update(observed_demand)

    def belief_mean_lambda(self) -> float:
        """Return the posterior expected demand intensity."""
        return float(np.dot(self._belief, self._lambda_by_tuple))

    def belief_variance_lambda(self) -> float:
        """Return the posterior variance of the demand intensity."""
        mean_lambda = self.belief_mean_lambda()
        centered = self._lambda_by_tuple - mean_lambda
        return float(np.dot(self._belief, centered * centered))

    def belief_marginals(self) -> BeliefMarginals:
        """Return marginal posterior probabilities for season, day, and weather."""
        season = np.zeros(self._n_season, dtype=float)
        day = np.zeros(self._n_day, dtype=float)
        weather = np.zeros(self._n_weather, dtype=float)

        for prob, (season_regime, day_regime, weather_regime) in zip(self._belief, self._regime_tuples):
            season[season_regime] += prob
            day[day_regime] += prob
            weather[weather_regime] += prob

        return BeliefMarginals(season=season, day=day, weather=weather)

    def most_likely_regimes(self) -> RegimeTuple:
        """Return the maximum-a-posteriori hidden regime tuple."""
        return self._regime_tuples[int(np.argmax(self._belief))]

    def _season_transition_matrix_for_t(self, t: int) -> np.ndarray:
        if self.exo.season_period > 0 and t >= 0 and ((int(t) + 1) % int(self.exo.season_period) != 0):
            return np.eye(self._n_season, dtype=float)
        return np.asarray(self.exo.P_season, dtype=float)

    @staticmethod
    def _validate_exogenous_model(exo) -> None:
        required = (
            "P_season",
            "P_day",
            "P_weather",
            "season_period",
            "lambda_for_regimes",
            "initial_season_regime",
            "initial_day_regime",
            "initial_weather_regime",
        )
        missing = [name for name in required if not hasattr(exo, name)]
        if missing:
            raise TypeError(f"exogenous model is missing required hidden-state attributes: {missing}")

    def _validate_regime_tuple(self, regime: RegimeTuple) -> RegimeTuple:
        season, day, weather = regime
        season = self._validate_regime_component(season, self._n_season, "season")
        day = self._validate_regime_component(day, self._n_day, "day")
        weather = self._validate_regime_component(weather, self._n_weather, "weather")
        return season, day, weather

    @staticmethod
    def _validate_regime_component(value: int, size: int, name: str) -> int:
        value = int(value)
        if value < 0 or value >= size:
            raise ValueError(f"{name} regime {value} out of range for size {size}")
        return value

    def _normalize_belief(self, belief: np.ndarray) -> np.ndarray:
        arr = np.asarray(belief, dtype=float).reshape(-1)
        if arr.shape != (len(self._regime_tuples),):
            raise ValueError(
                f"belief must have shape ({len(self._regime_tuples)},). Got {arr.shape}."
            )
        if np.any(arr < 0.0):
            raise ValueError("belief entries must be nonnegative")
        total = float(arr.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("belief must have positive finite total mass")
        return arr / total

    @staticmethod
    def _validate_observed_demand(value: float | int) -> int:
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"observed_demand must be a nonnegative finite value. Got {value!r}.")
        rounded = int(round(numeric))
        if abs(numeric - rounded) > 1e-8:
            raise ValueError(f"observed_demand must be integer-valued. Got {value!r}.")
        return rounded

    @staticmethod
    def _poisson_log_pmf(demand: int, lam: float) -> float:
        lam = float(lam)
        if lam <= 0.0:
            raise ValueError(f"Poisson rate must be positive. Got {lam}.")
        return demand * log(lam) - lam - lgamma(demand + 1.0)


__all__ = [
    "RegimeTuple",
    "BeliefMarginals",
    "MultiRegimeBeliefTracker",
]