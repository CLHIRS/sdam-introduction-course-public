from __future__ import annotations

from typing import Optional

import numpy as np
from inventory.core.types import State
from inventory.forecasters.base import DemandForecaster
from inventory.forecasters.ml import MultiRegimeFeatureAdapter, RegimeFeatureAdapter, SeasonalFeatureAdapter, regression_metrics


class EtsDemandForecaster(DemandForecaster):
    """Univariate ETS forecaster trained on a synthetic demand path.

    This is a classical time-series baseline: it fits an exponential smoothing
    model to a single demand series and forecasts the next H demand means. Unlike
    regime-aware ML or SARIMAX forecasters, it does not condition on the current
    state when producing forecasts.
    """

    def __init__(
        self,
        adapter: SeasonalFeatureAdapter | RegimeFeatureAdapter | MultiRegimeFeatureAdapter,
        *,
        trend: str | None = None,
        seasonal: str | None = None,
        seasonal_periods: int | None = None,
        damped_trend: bool = False,
        initialization_method: str = "estimated",
        use_boxcox: bool | str | float = False,
        remove_bias: bool = False,
    ):
        self.adapter = adapter
        self.trend = None if trend is None else str(trend)
        self.seasonal = None if seasonal is None else str(seasonal)
        self.seasonal_periods = None if seasonal_periods is None else int(seasonal_periods)
        self.damped_trend = bool(damped_trend)
        self.initialization_method = str(initialization_method)
        self.use_boxcox = use_boxcox
        self.remove_bias = bool(remove_bias)
        self._results = None
        self._trained = False
        self.fit_report: dict = {}

        if self.seasonal is not None and (self.seasonal_periods is None or self.seasonal_periods <= 1):
            raise ValueError("seasonal_periods must be > 1 when seasonal ETS is requested")

    @staticmethod
    def _build_model(
        y: np.ndarray,
        *,
        trend: str | None,
        seasonal: str | None,
        seasonal_periods: int | None,
        damped_trend: bool,
        initialization_method: str,
        use_boxcox: bool | str | float,
    ):
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except Exception as exc:
            raise ImportError("EtsDemandForecaster requires statsmodels.") from exc

        return ExponentialSmoothing(
            np.asarray(y, dtype=float).reshape(-1),
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            damped_trend=damped_trend,
            initialization_method=initialization_method,
            use_boxcox=use_boxcox,
        )

    def fit(self, y: np.ndarray):
        y = np.asarray(y, dtype=float).reshape(-1)
        model = self._build_model(
            y,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            damped_trend=self.damped_trend,
            initialization_method=self.initialization_method,
            use_boxcox=self.use_boxcox,
        )
        self._results = model.fit(optimized=True, remove_bias=self.remove_bias)
        self._trained = True
        return self

    def _evaluate_train(self, y: np.ndarray) -> dict:
        if not self._trained or self._results is None:
            raise RuntimeError("EtsDemandForecaster must be trained first.")
        fitted = np.asarray(self._results.fittedvalues, dtype=float).reshape(-1)
        fitted = np.maximum(fitted, 0.0)
        return regression_metrics(np.asarray(y, dtype=float).reshape(-1), fitted)

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
        _ = val_seed, val_t_start
        total_samples = int(n_samples) + int(val_samples)
        if total_samples <= 0:
            raise ValueError("total number of samples must be positive")

        data_kwargs = {"n_samples": total_samples, "seed": seed, "t_start": t_start}
        if isinstance(self.adapter, RegimeFeatureAdapter):
            data_kwargs["r0"] = r0

        X_all, y_all = self.adapter.generate_dataset(**data_kwargs)
        y_all = np.asarray(y_all, dtype=float).reshape(-1)
        y_train = y_all[: int(n_samples)]
        y_val = y_all[int(n_samples) : int(n_samples) + int(val_samples)]

        self.fit(y_train)
        train_metrics = self._evaluate_train(y_train)

        if int(val_samples) > 0:
            yhat_val = np.maximum(np.asarray(self._results.forecast(int(val_samples)), dtype=float).reshape(-1), 0.0)
            val_metrics = regression_metrics(y_val, yhat_val)
        else:
            val_metrics = {"MAE": 0.0, "RMSE": 0.0, "Bias": 0.0}

        self.fit_report = {
            "model_type": "ets",
            "trend": self.trend,
            "seasonal": self.seasonal,
            "seasonal_periods": self.seasonal_periods,
            "damped_trend": self.damped_trend,
            "feature_dim": int(np.asarray(X_all, dtype=float).shape[1]),
            "train": {"n_samples": int(n_samples), "seed": int(seed), "t_start": int(t_start), **train_metrics},
            "val": {"n_samples": int(val_samples), "seed": int(seed), "t_start": int(t_start) + int(n_samples), **val_metrics},
        }
        return self

    def forecast_mean_path(self, state: State, t: int, H: int, info: Optional[dict] = None) -> np.ndarray:
        _ = state, t, info
        if not self._trained or self._results is None:
            raise RuntimeError("EtsDemandForecaster must be trained first.")

        H = int(H)
        if H <= 0:
            return np.zeros(0, dtype=float)

        mu = np.asarray(self._results.forecast(H), dtype=float).reshape(-1)
        return np.maximum(mu, 0.0).astype(float)


class SarimaxRegimeDemandForecaster(DemandForecaster):
    """Regime-aware SARIMAX forecaster with exogenous regime features.

    The adapter supplies the exogenous feature matrix. The SARIMAX model is fit on
    a synthetic demand time series generated from the exogenous model, then used to
    produce a horizon-H conditional mean path.
    """

    def __init__(
        self,
        adapter: RegimeFeatureAdapter | MultiRegimeFeatureAdapter,
        *,
        order: tuple[int, int, int] = (1, 0, 0),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: str = "c",
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
    ):
        self.adapter = adapter
        self.order = tuple(int(v) for v in order)
        self.seasonal_order = tuple(int(v) for v in seasonal_order)
        self.trend = str(trend)
        self.enforce_stationarity = bool(enforce_stationarity)
        self.enforce_invertibility = bool(enforce_invertibility)
        self._results = None
        self._trained = False
        self.fit_report: dict = {}

    def _prepare_exog(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("SARIMAX exogenous features must be a 2-D array.")

        if X.shape[1] >= 1 and self.trend != "n":
            first_col = X[:, 0]
            if np.allclose(first_col, 1.0, atol=1e-12, rtol=0.0):
                return X[:, 1:]
        return X

    @staticmethod
    def _build_sarimax(
        y: np.ndarray,
        X: np.ndarray,
        *,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int],
        trend: str,
        enforce_stationarity: bool,
        enforce_invertibility: bool,
    ):
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except Exception as exc:
            raise ImportError("SarimaxRegimeDemandForecaster requires statsmodels.") from exc

        return SARIMAX(
            endog=np.asarray(y, dtype=float).reshape(-1),
            exog=np.asarray(X, dtype=float),
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = self._prepare_exog(X)
        y = np.asarray(y, dtype=float).reshape(-1)

        model = self._build_sarimax(
            y,
            X,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        self._results = model.fit(disp=False)
        self._trained = True
        return self

    def _predict_in_sample(self, X: np.ndarray) -> np.ndarray:
        if not self._trained or self._results is None:
            raise RuntimeError("SarimaxRegimeDemandForecaster must be trained first.")
        X = self._prepare_exog(X)
        fitted = np.asarray(self._results.fittedvalues, dtype=float).reshape(-1)
        if fitted.shape[0] != X.shape[0]:
            raise ValueError("Fitted SARIMAX values do not match the requested sample size.")
        return fitted

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        X = self._prepare_exog(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if not self._trained or self._results is None:
            raise RuntimeError("SarimaxRegimeDemandForecaster must be trained first.")

        try:
            applied = self._results.apply(endog=y, exog=X, refit=False)
            yhat = np.asarray(applied.fittedvalues, dtype=float).reshape(-1)
        except Exception:
            yhat = self._predict_in_sample(X)

        yhat = np.maximum(yhat, 0.0)
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

        aic = None
        bic = None
        if self._results is not None:
            aic = float(getattr(self._results, "aic", np.nan))
            bic = float(getattr(self._results, "bic", np.nan))

        self.fit_report = {
            "model_type": "sarimax",
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "trend": self.trend,
            "feature_dim": int(X_train.shape[1]),
            "aic": aic,
            "bic": bic,
            "train": {"n_samples": int(n_samples), "seed": int(seed), "t_start": int(t_start), **train_metrics},
            "val": {"n_samples": int(val_samples), "seed": int(val_seed), "t_start": int(val_t_start), **val_metrics},
        }
        return self

    def forecast_mean_path(self, state: State, t: int, H: int, info: Optional[dict] = None) -> np.ndarray:
        _ = info
        if not self._trained or self._results is None:
            raise RuntimeError("SarimaxRegimeDemandForecaster must be trained first.")

        H = int(H)
        if H <= 0:
            return np.zeros(0, dtype=float)

        Xh = self._prepare_exog(np.asarray(self.adapter.features_path(state, int(t), H), dtype=float))
        fc = self._results.get_forecast(steps=H, exog=Xh)
        mu = np.asarray(fc.predicted_mean, dtype=float).reshape(-1)
        return np.maximum(mu, 0.0).astype(float)


__all__ = ["EtsDemandForecaster", "SarimaxRegimeDemandForecaster"]