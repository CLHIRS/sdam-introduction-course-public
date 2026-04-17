from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from inventory.core.types import State
from inventory.forecasters.base import DemandForecaster
from inventory.problems.demand_models import (
    PoissonConstantDemand,
    PoissonMultiRegimeDemand,
    PoissonRegimeDemand,
    PoissonSeasonalDemand,
)


@dataclass(frozen=True)
class ConstantFeatureAdapter:
    """Adapter for constant Poisson demand.

    This is the simplest possible adapter:
    - Features do not depend on state or time: φ(t) = [1].
    - One-step labels are sampled from the exogenous model: D ~ Poisson(lam).

    It plugs directly into :class:`MlDemandForecaster`.
    """

    exog: PoissonConstantDemand

    def features(self, t: int) -> np.ndarray:
        _ = int(t)
        return np.array([1.0], dtype=float)

    def features_path(self, t: int, H: int) -> np.ndarray:
        _ = int(t)
        H = int(H)
        if H <= 0:
            return np.zeros((0, 1), dtype=float)
        return np.ones((H, 1), dtype=float)

    def generate_dataset(self, n_samples: int, seed: int, t_start: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        _ = int(t_start)
        rng = np.random.default_rng(int(seed))
        n_samples = int(n_samples)

        lam = float(getattr(self.exog, "lam", 0.0))
        X = np.ones((n_samples, 1), dtype=float)
        y = rng.poisson(lam, size=n_samples).astype(float)
        return X, y


@dataclass(frozen=True)
class SeasonalFeatureAdapter:
    """Adapter for seasonal demand models.

    Responsibilities:
      1) Define features φ(t)
      2) Generate synthetic training data (X,y) for one-step demand
      3) Provide a multi-step feature matrix for a horizon H

    Feature design follows lecture_12_b:
      φ(t) = [1, sin(2π t/period), cos(2π t/period), spike_flag]
    """

    exog: PoissonSeasonalDemand

    def features(self, t: int) -> np.ndarray:
        t = int(t)
        period = max(1, int(self.exog.period))
        ang = 2.0 * np.pi * (t % period) / float(period)
        sinv = float(np.sin(ang))
        cosv = float(np.cos(ang))
        spike = 1.0 if (self.exog.spike_every > 0 and (t % int(self.exog.spike_every) == 0) and t > 0) else 0.0
        return np.array([1.0, sinv, cosv, spike], dtype=float)

    def features_path(self, t: int, H: int) -> np.ndarray:
        H = int(H)
        return np.vstack([self.features(t + k) for k in range(H)])

    def generate_dataset(self, n_samples: int, seed: int, t_start: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Synthetic one-step dataset.

        X_i = φ(t_i)
        y_i = D_{t_i+1} ~ Poisson(lambda_t(t_i))
        """

        rng = np.random.default_rng(int(seed))
        n_samples = int(n_samples)

        X = np.vstack([self.features(int(t_start) + i) for i in range(n_samples)])
        y = np.empty(n_samples, dtype=float)
        for i in range(n_samples):
            lam = float(self.exog.lambda_t(int(t_start) + i))
            y[i] = float(rng.poisson(lam))
        return X, y


@dataclass(frozen=True)
class RegimeFeatureAdapter:
    """Adapter for observable-regime demand models.

    This is the regime analogue of :class:`SeasonalFeatureAdapter`.

    Key difference vs seasonal: the future regime is random, so multi-step features
    represent *expected regime distributions* under the Markov chain.

    Convention:
      - state stores current regime R_t at `exog.regime_index`
      - demand uses next regime: D_{t+1} ~ Poisson(lam[R_{t+1}])

    Feature design (per step k in horizon):
      φ_k = [1, pi_k[0], ..., pi_k[K-1]]
    where pi_k is the distribution of R_{t+k+1} given current R_t.
    """

    exog: PoissonRegimeDemand

    def _n_regimes(self) -> int:
        return int(np.asarray(self.exog.lam_by_regime).shape[0])

    def _regime_from_state(self, state: State) -> int:
        S = np.asarray(state, dtype=float).reshape(-1)
        idx = int(self.exog.regime_index)
        if idx < 0 or idx >= S.shape[0]:
            raise ValueError("regime_index is out of bounds for state")
        r = int(np.round(float(S[idx])))
        return int(np.clip(r, 0, self._n_regimes() - 1))

    def _pi_next_from_regime(self, r_t: int) -> np.ndarray:
        P = np.asarray(self.exog.P, dtype=float)
        return np.asarray(P[int(r_t)], dtype=float).reshape(-1)

    def features(self, state: State, t: int) -> np.ndarray:
        """One-step features for predicting demand at time t+1.

        Returns features that correspond to the distribution of R_{t+1} given R_t.
        """

        _ = int(t)  # kept for API symmetry with other adapters
        r_t = self._regime_from_state(state)
        pi = self._pi_next_from_regime(r_t)
        return np.concatenate([np.array([1.0], dtype=float), pi.astype(float)])

    def features_path(self, state: State, t: int, H: int) -> np.ndarray:
        """Multi-step feature matrix.

        Returns an array of shape (H, 1+K), where row k corresponds to time t+k.
        """

        _ = int(t)
        H = int(H)
        if H <= 0:
            return np.zeros((0, 1 + self._n_regimes()), dtype=float)

        P = np.asarray(self.exog.P, dtype=float)

        r_t = self._regime_from_state(state)
        pi = self._pi_next_from_regime(r_t)

        Xh = np.empty((H, 1 + pi.shape[0]), dtype=float)
        for k in range(H):
            Xh[k, 0] = 1.0
            Xh[k, 1:] = pi
            pi = pi @ P
        return Xh

    def generate_dataset(self, n_samples: int, seed: int, t_start: int = 0, *, r0: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Synthetic one-step dataset for regime demand.

        X_i = [1, pi_i] where pi_i is distribution of R_{t_i+1} given current regime.
        y_i = D_{t_i+1} sampled from the exogenous model.

        Notes:
        - This is time-homogeneous; `t_start` is accepted for symmetry.
        - We simulate a Markov chain of regimes to generate realistic (correlated) samples.
        """

        _ = int(t_start)
        rng = np.random.default_rng(int(seed))
        n_samples = int(n_samples)

        K = self._n_regimes()
        lam = np.asarray(self.exog.lam_by_regime, dtype=float).reshape(-1)
        P = np.asarray(self.exog.P, dtype=float)

        if r0 is None:
            r = int(rng.integers(0, K))
        else:
            r = int(np.clip(int(r0), 0, K - 1))

        X = np.empty((n_samples, 1 + K), dtype=float)
        y = np.empty(n_samples, dtype=float)

        for i in range(n_samples):
            pi = np.asarray(P[r], dtype=float).reshape(-1)
            X[i, 0] = 1.0
            X[i, 1:] = pi

            r_next = int(rng.choice(K, p=pi))
            y[i] = float(rng.poisson(float(lam[r_next])))
            r = r_next

        return X, y


@dataclass(frozen=True)
class MultiRegimeFeatureAdapter:
    """Adapter for observable *multi-regime* demand models.

    This generalizes :class:`RegimeFeatureAdapter` to the common teaching case with
    three independent observable regimes (season/day/weather).

    Convention (matches `ExogenousPoissonMultiRegime` / `PoissonMultiRegimeDemand`):
    - state stores current regimes at indices exog.season_index/day_index/weather_index
    - demand uses next regimes:
        λ_eff = λ_base_season[r_season_next] * (1 + coeff_day[r_day_next] + coeff_weather[r_weather_next])
        D_{t+1} ~ Poisson(max(lam_min, λ_eff))

    Feature design (per step k in horizon):
      φ_k = [1, pi_season_k..., pi_day_k..., pi_weather_k...]
    where each pi_*_k is the distribution of the *next* regime at time t+k+1.

    Notes on season:
    - season evolves only on boundaries determined by exog.season_period;
      otherwise it is held constant.
    """

    exog: PoissonMultiRegimeDemand

    def _n_seasons(self) -> int:
        return int(np.asarray(self.exog.P_season, dtype=float).shape[0])

    def _n_days(self) -> int:
        return int(np.asarray(self.exog.P_day, dtype=float).shape[0])

    def _n_weathers(self) -> int:
        return int(np.asarray(self.exog.P_weather, dtype=float).shape[0])

    def _regimes_from_state(self, state: State) -> tuple[int, int, int]:
        S = np.asarray(state, dtype=float).reshape(-1)
        si = int(self.exog.season_index)
        di = int(self.exog.day_index)
        wi = int(self.exog.weather_index)
        if max(si, di, wi) >= S.shape[0] or min(si, di, wi) < 0:
            raise ValueError("season/day/weather indices are out of bounds for state")

        r_season = int(np.round(float(S[si])))
        r_day = int(np.round(float(S[di])))
        r_weather = int(np.round(float(S[wi])))

        r_season = int(np.clip(r_season, 0, self._n_seasons() - 1))
        r_day = int(np.clip(r_day, 0, self._n_days() - 1))
        r_weather = int(np.clip(r_weather, 0, self._n_weathers() - 1))
        return r_season, r_day, r_weather

    def _one_hot(self, r: int, K: int) -> np.ndarray:
        v = np.zeros(int(K), dtype=float)
        v[int(np.clip(int(r), 0, int(K) - 1))] = 1.0
        return v

    def _season_updates_at_next_step(self, t: int) -> bool:
        season_period = int(getattr(self.exog, "season_period", 0))
        if season_period <= 0:
            return True
        return ((int(t) + 1) % season_period) == 0

    def _pi_next(self, P: np.ndarray, r_t: int) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        return np.asarray(P[int(r_t)], dtype=float).reshape(-1)

    def features(self, state: State, t: int) -> np.ndarray:
        """One-step features for predicting demand at time t+1."""

        r_season_t, r_day_t, r_weather_t = self._regimes_from_state(state)

        if self._season_updates_at_next_step(int(t)):
            pi_season = self._pi_next(self.exog.P_season, r_season_t)
        else:
            pi_season = self._one_hot(r_season_t, self._n_seasons())

        pi_day = self._pi_next(self.exog.P_day, r_day_t)
        pi_weather = self._pi_next(self.exog.P_weather, r_weather_t)

        return np.concatenate([np.array([1.0], dtype=float), pi_season, pi_day, pi_weather]).astype(float)

    def features_path(self, state: State, t: int, H: int) -> np.ndarray:
        """Multi-step feature matrix of shape (H, 1+K_season+K_day+K_weather)."""

        t = int(t)
        H = int(H)
        if H <= 0:
            return np.zeros((0, 1 + self._n_seasons() + self._n_days() + self._n_weathers()), dtype=float)

        P_season = np.asarray(self.exog.P_season, dtype=float)
        P_day = np.asarray(self.exog.P_day, dtype=float)
        P_weather = np.asarray(self.exog.P_weather, dtype=float)

        r_season_t, r_day_t, r_weather_t = self._regimes_from_state(state)
        pi_season_curr = self._one_hot(r_season_t, self._n_seasons())
        pi_day_curr = self._one_hot(r_day_t, self._n_days())
        pi_weather_curr = self._one_hot(r_weather_t, self._n_weathers())

        d = 1 + pi_season_curr.shape[0] + pi_day_curr.shape[0] + pi_weather_curr.shape[0]
        Xh = np.empty((H, d), dtype=float)

        for k in range(H):
            tk = t + k

            if self._season_updates_at_next_step(tk):
                pi_season_next = pi_season_curr @ P_season
            else:
                pi_season_next = pi_season_curr

            pi_day_next = pi_day_curr @ P_day
            pi_weather_next = pi_weather_curr @ P_weather

            Xh[k, 0] = 1.0
            j = 1
            Xh[k, j : j + pi_season_next.shape[0]] = pi_season_next
            j += pi_season_next.shape[0]
            Xh[k, j : j + pi_day_next.shape[0]] = pi_day_next
            j += pi_day_next.shape[0]
            Xh[k, j : j + pi_weather_next.shape[0]] = pi_weather_next

            pi_season_curr = pi_season_next
            pi_day_curr = pi_day_next
            pi_weather_curr = pi_weather_next

        return Xh

    def generate_dataset(
        self,
        n_samples: int,
        seed: int,
        t_start: int = 0,
        *,
        season0: int | None = None,
        day0: int | None = None,
        weather0: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synthetic one-step dataset.

        X_i = features representing distributions of next regimes.
        y_i = sampled demand D_{t_i+1} from the multi-regime exogenous model.

        We simulate regime paths so samples are correlated (more realistic).
        """

        rng = np.random.default_rng(int(seed))
        n_samples = int(n_samples)
        t_start = int(t_start)

        Ks, Kd, Kw = self._n_seasons(), self._n_days(), self._n_weathers()
        d = 1 + Ks + Kd + Kw

        P_season = np.asarray(self.exog.P_season, dtype=float)
        P_day = np.asarray(self.exog.P_day, dtype=float)
        P_weather = np.asarray(self.exog.P_weather, dtype=float)

        base = np.asarray(self.exog.lambda_base_season, dtype=float).reshape(-1)
        coeff_day = np.asarray(self.exog.lambda_coeff_day, dtype=float).reshape(-1)
        coeff_weather = np.asarray(self.exog.lambda_coeff_weather, dtype=float).reshape(-1)
        lam_min = float(getattr(self.exog, "lam_min", 1.0))

        r_season = int(rng.integers(0, Ks)) if season0 is None else int(np.clip(int(season0), 0, Ks - 1))
        r_day = int(rng.integers(0, Kd)) if day0 is None else int(np.clip(int(day0), 0, Kd - 1))
        r_weather = int(rng.integers(0, Kw)) if weather0 is None else int(np.clip(int(weather0), 0, Kw - 1))

        X = np.empty((n_samples, d), dtype=float)
        y = np.empty(n_samples, dtype=float)

        for i in range(n_samples):
            t = t_start + i

            if self._season_updates_at_next_step(t):
                pi_season = np.asarray(P_season[r_season], dtype=float).reshape(-1)
                r_season_next = int(rng.choice(Ks, p=pi_season))
            else:
                pi_season = self._one_hot(r_season, Ks)
                r_season_next = r_season

            pi_day = np.asarray(P_day[r_day], dtype=float).reshape(-1)
            pi_weather = np.asarray(P_weather[r_weather], dtype=float).reshape(-1)

            r_day_next = int(rng.choice(Kd, p=pi_day))
            r_weather_next = int(rng.choice(Kw, p=pi_weather))

            X[i, 0] = 1.0
            j = 1
            X[i, j : j + Ks] = pi_season
            j += Ks
            X[i, j : j + Kd] = pi_day
            j += Kd
            X[i, j : j + Kw] = pi_weather

            lam_eff = float(base[r_season_next]) * (1.0 + float(coeff_day[r_day_next]) + float(coeff_weather[r_weather_next]))
            lam_eff = max(lam_min, lam_eff)
            y[i] = float(rng.poisson(lam_eff))

            r_season, r_day, r_weather = r_season_next, r_day_next, r_weather_next

        return X, y


@dataclass(frozen=True)
class MultiRegimeAr1FeatureAdapter:
    """Multi-regime adapter with one autoregressive demand lag.

    Features combine the regime-transition structure from
    :class:`MultiRegimeFeatureAdapter` with the most recent realized demand.

    Per-step feature layout:
      φ_k = [1, pi_season_k..., pi_day_k..., pi_weather_k..., last_demand_k / lag_scale]

    Notes:
    - The lag term uses the most recently realized demand available at decision time.
    - During multi-step forecasting, the learned forecaster rolls the lag forward
      recursively using its own predicted means.
    """

    exog: PoissonMultiRegimeDemand
    default_last_demand: Optional[float] = None
    lag_scale: float = 1.0

    def __post_init__(self) -> None:
        if float(self.lag_scale) <= 0.0:
            raise ValueError("lag_scale must be positive")

    def _base_adapter(self) -> MultiRegimeFeatureAdapter:
        return MultiRegimeFeatureAdapter(self.exog)

    def _default_last_demand_from_state(self, state: State) -> float:
        base = self._base_adapter()
        r_season, r_day, r_weather = base._regimes_from_state(state)
        if self.default_last_demand is not None:
            return float(max(0.0, self.default_last_demand))
        return float(self.exog.lambda_for_regimes(r_season, r_day, r_weather))

    def extract_last_demand(self, state: State, info: Optional[dict]) -> float:
        if info and "last_demand" in info:
            value = np.asarray(info["last_demand"], dtype=float).reshape(-1)
            if value.size >= 1 and np.isfinite(value[-1]):
                return float(max(0.0, value[-1]))
        return self._default_last_demand_from_state(state)

    def regime_features(self, state: State, t: int) -> np.ndarray:
        return self._base_adapter().features(state, t)

    def regime_features_path(self, state: State, t: int, H: int) -> np.ndarray:
        return self._base_adapter().features_path(state, t, H)

    def one_step_features(self, state: State, t: int, last_demand: float) -> np.ndarray:
        phi_regime = self.regime_features(state, t)
        lag_term = np.array([float(max(0.0, last_demand)) / float(self.lag_scale)], dtype=float)
        return np.concatenate([phi_regime, lag_term]).astype(float)

    def generate_dataset(
        self,
        n_samples: int,
        seed: int,
        t_start: int = 0,
        *,
        season0: int | None = None,
        day0: int | None = None,
        weather0: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synthetic one-step dataset with one lagged realized-demand feature."""

        base_adapter = self._base_adapter()
        rng = np.random.default_rng(int(seed))
        n_samples = int(n_samples)
        t_start = int(t_start)

        Ks = base_adapter._n_seasons()
        Kd = base_adapter._n_days()
        Kw = base_adapter._n_weathers()
        d = 1 + Ks + Kd + Kw + 1

        P_season = np.asarray(self.exog.P_season, dtype=float)
        P_day = np.asarray(self.exog.P_day, dtype=float)
        P_weather = np.asarray(self.exog.P_weather, dtype=float)

        base = np.asarray(self.exog.lambda_base_season, dtype=float).reshape(-1)
        coeff_day = np.asarray(self.exog.lambda_coeff_day, dtype=float).reshape(-1)
        coeff_weather = np.asarray(self.exog.lambda_coeff_weather, dtype=float).reshape(-1)
        lam_min = float(getattr(self.exog, "lam_min", 1.0))

        r_season = int(rng.integers(0, Ks)) if season0 is None else int(np.clip(int(season0), 0, Ks - 1))
        r_day = int(rng.integers(0, Kd)) if day0 is None else int(np.clip(int(day0), 0, Kd - 1))
        r_weather = int(rng.integers(0, Kw)) if weather0 is None else int(np.clip(int(weather0), 0, Kw - 1))

        X = np.empty((n_samples, d), dtype=float)
        y = np.empty(n_samples, dtype=float)

        if self.default_last_demand is None:
            last_demand = float(rng.poisson(self.exog.lambda_for_regimes(r_season, r_day, r_weather)))
        else:
            last_demand = float(max(0.0, self.default_last_demand))

        for i in range(n_samples):
            t = t_start + i

            if base_adapter._season_updates_at_next_step(t):
                pi_season = np.asarray(P_season[r_season], dtype=float).reshape(-1)
                r_season_next = int(rng.choice(Ks, p=pi_season))
            else:
                pi_season = base_adapter._one_hot(r_season, Ks)
                r_season_next = r_season

            pi_day = np.asarray(P_day[r_day], dtype=float).reshape(-1)
            pi_weather = np.asarray(P_weather[r_weather], dtype=float).reshape(-1)

            r_day_next = int(rng.choice(Kd, p=pi_day))
            r_weather_next = int(rng.choice(Kw, p=pi_weather))

            X[i, 0] = 1.0
            j = 1
            X[i, j : j + Ks] = pi_season
            j += Ks
            X[i, j : j + Kd] = pi_day
            j += Kd
            X[i, j : j + Kw] = pi_weather
            j += Kw
            X[i, j] = float(max(0.0, last_demand)) / float(self.lag_scale)

            lam_eff = float(base[r_season_next]) * (1.0 + float(coeff_day[r_day_next]) + float(coeff_weather[r_weather_next]))
            lam_eff = max(lam_min, lam_eff)
            y[i] = float(rng.poisson(lam_eff))

            last_demand = float(y[i])
            r_season, r_day, r_weather = r_season_next, r_day_next, r_weather_next

        return X, y


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "Bias": float(np.mean(err)),
    }


def _build_ml_backend(
    model_type: str,
    *,
    random_state: int,
    quantile: float = 0.5,
    n_estimators: int = 150,
    learning_rate: float = 0.05,
    max_depth: int = 3,
) -> tuple[str, object]:
    model_type = str(model_type).lower()

    if model_type == "tree":
        try:
            from sklearn.tree import DecisionTreeRegressor

            return model_type, DecisionTreeRegressor(max_depth=6, random_state=int(random_state))
        except Exception:
            model_type = "linear"

    if model_type == "mlp":
        try:
            from sklearn.neural_network import MLPRegressor

            return model_type, MLPRegressor(
                random_state=int(random_state),
                hidden_layer_sizes=(32, 32),
                max_iter=500,
                early_stopping=True,
            )
        except Exception:
            model_type = "linear"

    if model_type == "elasticnet":
        try:
            from sklearn.linear_model import ElasticNet

            return model_type, ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=5000, random_state=int(random_state))
        except Exception:
            model_type = "linear"

    if model_type == "quantile_boosting":
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            return model_type, GradientBoostingRegressor(
                loss="quantile",
                alpha=float(quantile),
                n_estimators=int(n_estimators),
                learning_rate=float(learning_rate),
                max_depth=int(max_depth),
                random_state=int(random_state),
            )
        except Exception:
            model_type = "linear"

    return "linear", {"intercept": 0.0, "coef": None}


def _fit_ml_backend(model_type: str, model: object, X: np.ndarray, y: np.ndarray) -> None:
    if model_type == "linear":
        if not isinstance(model, dict):
            raise TypeError("Linear backend must use dict model storage.")
        X1 = np.hstack([np.ones((X.shape[0], 1)), X])
        w, *_ = np.linalg.lstsq(X1, y, rcond=None)
        model["intercept"] = float(w[0])
        model["coef"] = w[1:].copy()
        return

    model.fit(X, y)


def _predict_ml_backend(model_type: str, model: object, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if model_type == "linear":
        if not isinstance(model, dict) or model.get("coef") is None:
            raise RuntimeError("Linear model is not fit.")
        yhat = float(model["intercept"]) + X @ np.asarray(model["coef"], dtype=float)
    else:
        yhat = model.predict(X)

    return np.asarray(yhat, dtype=float).reshape(-1)


class MlDemandForecaster(DemandForecaster):
    """Generic ML forecaster for the seasonal adapter.

    Learns a one-step mapping: φ(t) -> E[D_{t+1}].
    Multi-step forecast: evaluate on φ(t+k) for k=0..H-1.

    `model_type` options:
      - `tree`: DecisionTreeRegressor (if sklearn installed) else linear fallback
      - `mlp`:  MLPRegressor (if sklearn installed) else linear fallback
            - `elasticnet`: ElasticNet regressor (if sklearn installed) else linear fallback
      - `linear`: always use least-squares linear model
    """

    def __init__(self, adapter: SeasonalFeatureAdapter, *, model_type: str = "tree", random_state: int = 0):
        self.adapter = adapter
        self.model_type = str(model_type).lower()
        self.random_state = int(random_state)
        self.model = None
        self._trained = False
        self.fit_report: dict = {}

        self._build_model_backend()

    def _build_model_backend(self) -> None:
        self.model_type, self.model = _build_ml_backend(self.model_type, random_state=self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        _fit_ml_backend(self.model_type, self.model, X, y)

        self._trained = True
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return _predict_ml_backend(self.model_type, self.model, X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        yhat = np.maximum(self._predict(X), 0.0)
        return regression_metrics(y, yhat)

    def fit_from_exogenous(
        self,
        *,
        n_samples: int = 6000,
        seed: int = 7,
        t_start: int = 0,
        val_samples: int = 1000,
        val_seed: int = 99,
        val_t_start: int = 8000,
    ):
        X_train, y_train = self.adapter.generate_dataset(n_samples=n_samples, seed=seed, t_start=t_start)
        self.fit(X_train, y_train)

        train_metrics = self.evaluate(X_train, y_train)
        X_val, y_val = self.adapter.generate_dataset(n_samples=val_samples, seed=val_seed, t_start=val_t_start)
        val_metrics = self.evaluate(X_val, y_val)

        self.fit_report = {
            "model_type": self.model_type,
            "random_state": self.random_state,
            "feature_dim": int(X_train.shape[1]),
            "train": {"n_samples": int(n_samples), "seed": int(seed), "t_start": int(t_start), **train_metrics},
            "val": {"n_samples": int(val_samples), "seed": int(val_seed), "t_start": int(val_t_start), **val_metrics},
        }
        return self

    def forecast_mean_path(self, state: State, t: int, H: int, info: Optional[dict] = None) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("MlDemandForecaster must be trained first.")
        Xh = self.adapter.features_path(int(t), int(H))
        mu = np.maximum(self._predict(Xh), 0.0)
        return mu.astype(float)


class MlRegimeDemandForecaster(DemandForecaster):
    """ML mean-demand forecaster for regime-aware adapters.

    Supports both :class:`RegimeFeatureAdapter` and :class:`MultiRegimeFeatureAdapter`.

    Learns a one-step mapping: φ(state,t) -> E[D_{t+1}].
    Multi-step forecast uses the adapter's ``features_path()`` implementation,
    which encodes regime-transition uncertainty as expected regime distributions.

        `model_type` options:
            - `tree`: DecisionTreeRegressor (if sklearn installed) else linear fallback
            - `mlp`: MLPRegressor (if sklearn installed) else linear fallback
            - `elasticnet`: ElasticNet regressor (if sklearn installed) else linear fallback
            - `linear`: always use least-squares linear model
    """

    def __init__(self, adapter: RegimeFeatureAdapter | MultiRegimeFeatureAdapter, *, model_type: str = "tree", random_state: int = 0):
        self.adapter = adapter
        self.model_type = str(model_type).lower()
        self.random_state = int(random_state)
        self.model = None
        self._trained = False
        self.fit_report: dict = {}

        self._build_model_backend()

    def _build_model_backend(self) -> None:
        self.model_type, self.model = _build_ml_backend(self.model_type, random_state=self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        _fit_ml_backend(self.model_type, self.model, X, y)

        self._trained = True
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return _predict_ml_backend(self.model_type, self.model, X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        yhat = np.maximum(self._predict(X), 0.0)
        return regression_metrics(y, yhat)

    def fit_from_exogenous(
        self,
        *,
        n_samples: int = 6000,
        seed: int = 7,
        t_start: int = 0,
        val_samples: int = 1000,
        val_seed: int = 99,
        val_t_start: int = 8000,
        r0: int | None = None,
    ):
        train_kwargs = {"n_samples": n_samples, "seed": seed, "t_start": t_start}
        val_kwargs = {"n_samples": val_samples, "seed": val_seed, "t_start": val_t_start}

        if isinstance(self.adapter, RegimeFeatureAdapter):
            train_kwargs["r0"] = r0
            val_kwargs["r0"] = r0

        X_train, y_train = self.adapter.generate_dataset(**train_kwargs)
        self.fit(X_train, y_train)

        train_metrics = self.evaluate(X_train, y_train)
        X_val, y_val = self.adapter.generate_dataset(**val_kwargs)
        val_metrics = self.evaluate(X_val, y_val)

        self.fit_report = {
            "model_type": self.model_type,
            "random_state": self.random_state,
            "feature_dim": int(X_train.shape[1]),
            "train": {"n_samples": int(n_samples), "seed": int(seed), "t_start": int(t_start), **train_metrics},
            "val": {"n_samples": int(val_samples), "seed": int(val_seed), "t_start": int(val_t_start), **val_metrics},
        }
        return self

    def forecast_mean_path(self, state: State, t: int, H: int, info: Optional[dict] = None) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("MlRegimeDemandForecaster must be trained first.")
        Xh = self.adapter.features_path(state, int(t), int(H))
        mu = np.maximum(self._predict(Xh), 0.0)
        return mu.astype(float)


class QuantileBoostingRegimeDemandForecaster(MlRegimeDemandForecaster):
    """Regime-aware forecaster using gradient-boosted quantile regression trees.

    This forecaster targets a conditional demand quantile rather than a conditional
    mean. It still exposes `forecast_mean_path(...)` for compatibility with the
    existing `DemandForecaster` protocol and downstream CFA policies.

    Parameters
    ----------
    quantile:
        Target conditional quantile in `(0, 1)`. `0.5` corresponds to the median.
    n_estimators, learning_rate, max_depth:
        Standard gradient boosting hyperparameters for the tree ensemble.
    """

    def __init__(
        self,
        adapter: RegimeFeatureAdapter | MultiRegimeFeatureAdapter,
        *,
        quantile: float = 0.5,
        random_state: int = 0,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 3,
    ):
        self.quantile = float(quantile)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)

        if not (0.0 < self.quantile < 1.0):
            raise ValueError("quantile must lie strictly between 0 and 1")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")

        super().__init__(adapter, model_type="quantile_boosting", random_state=random_state)

    def _build_model_backend(self) -> None:
        self.model_type, self.model = _build_ml_backend(
            "quantile_boosting",
            random_state=self.random_state,
            quantile=self.quantile,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
        )

    def fit_from_exogenous(
        self,
        *,
        n_samples: int = 6000,
        seed: int = 7,
        t_start: int = 0,
        val_samples: int = 1000,
        val_seed: int = 99,
        val_t_start: int = 8000,
        r0: int | None = None,
    ):
        super().fit_from_exogenous(
            n_samples=n_samples,
            seed=seed,
            t_start=t_start,
            val_samples=val_samples,
            val_seed=val_seed,
            val_t_start=val_t_start,
            r0=r0,
        )
        self.fit_report.update(
            {
                "quantile": self.quantile,
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
            }
        )
        return self


class MlAr1RegimeDemandForecaster(MlRegimeDemandForecaster):
    """Regime-aware ML forecaster with one autoregressive demand lag."""

    def __init__(self, adapter: MultiRegimeAr1FeatureAdapter, *, model_type: str = "tree", random_state: int = 0):
        super().__init__(adapter, model_type=model_type, random_state=random_state)

    @property
    def adapter(self) -> MultiRegimeAr1FeatureAdapter:  # type: ignore[override]
        return self.__dict__["adapter"]

    @adapter.setter
    def adapter(self, value: MultiRegimeAr1FeatureAdapter) -> None:  # type: ignore[override]
        self.__dict__["adapter"] = value

    def forecast_mean_path(self, state: State, t: int, H: int, info: Optional[dict] = None) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("MlAr1RegimeDemandForecaster must be trained first.")

        H = int(H)
        if H <= 0:
            return np.zeros(0, dtype=float)

        regime_Xh = self.adapter.regime_features_path(state, int(t), H)
        mu = np.empty(H, dtype=float)
        lag_value = self.adapter.extract_last_demand(state, info)

        for k in range(H):
            xk = np.concatenate(
                [
                    np.asarray(regime_Xh[k], dtype=float).reshape(-1),
                    np.array([float(max(0.0, lag_value)) / float(self.adapter.lag_scale)], dtype=float),
                ]
            )
            mu_k = float(max(0.0, self._predict(xk)[0]))
            mu[k] = mu_k
            lag_value = mu_k

        return mu.astype(float)


__all__ = [
    "ConstantFeatureAdapter",
    "SeasonalFeatureAdapter",
    "RegimeFeatureAdapter",
    "MultiRegimeFeatureAdapter",
    "MultiRegimeAr1FeatureAdapter",
    "MlDemandForecaster",
    "MlRegimeDemandForecaster",
    "QuantileBoostingRegimeDemandForecaster",
    "MlAr1RegimeDemandForecaster",
    "regression_metrics",
]
