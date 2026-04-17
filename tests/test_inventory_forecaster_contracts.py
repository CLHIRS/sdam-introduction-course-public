import numpy as np

from inventory.forecasters.ml import MlAr1RegimeDemandForecaster, MlRegimeDemandForecaster, MultiRegimeAr1FeatureAdapter, MultiRegimeFeatureAdapter, QuantileBoostingRegimeDemandForecaster
from inventory.forecasters.naive import ConstantMeanForecaster, ExogenousAwareMeanForecaster, NaiveForecaster, RollingMeanForecaster
from inventory.forecasters.ts import EtsDemandForecaster, SarimaxRegimeDemandForecaster
from inventory.problems.demand_models import ExogenousPoissonMultiRegime, PoissonConstantDemand, PoissonSeasonalDemand


def _assert_mean_path_contract(mu: np.ndarray, H: int) -> None:
    assert isinstance(mu, np.ndarray)
    assert mu.shape == (H,)
    assert np.isfinite(mu).all()
    assert (mu >= 0.0).all()


def test_forecasters_forecast_mean_path_contract() -> None:
    S = np.array([80.0], dtype=float)

    forecasters = [
        ConstantMeanForecaster(mean=123.0),
        NaiveForecaster(default_value=15.0),
        ExogenousAwareMeanForecaster(exogenous_model=PoissonConstantDemand(lam=60.0)),
        ExogenousAwareMeanForecaster(exogenous_model=PoissonSeasonalDemand(base=50.0, amp=10.0, period=12, lam_min=1.0)),
    ]

    for f in forecasters:
        for t in [0, 3]:
            for H in [1, 5, 12]:
                mu1 = np.asarray(f.forecast_mean_path(S, t, H, info=None), dtype=float).reshape(-1)
                _assert_mean_path_contract(mu1, H)

                mu2 = np.asarray(f.forecast_mean_path(S, t, H, info={}), dtype=float).reshape(-1)
                _assert_mean_path_contract(mu2, H)


def test_constant_mean_forecaster_values_are_constant() -> None:
    f = ConstantMeanForecaster(mean=7.5)
    S = np.array([0.0], dtype=float)
    H = 6

    mu = np.asarray(f.forecast_mean_path(S, t=0, H=H), dtype=float)
    assert mu.shape == (H,)
    assert np.allclose(mu, 7.5, atol=0.0, rtol=0.0)


def test_naive_forecaster_repeats_last_observed_demand() -> None:
    f = NaiveForecaster(default_value=2.0)
    S = np.array([0.0], dtype=float)

    mu_last = np.asarray(f.forecast_mean_path(S, t=4, H=5, info={"last_demand": 11.0}), dtype=float)
    assert np.allclose(mu_last, 11.0, atol=0.0, rtol=0.0)

    mu_default = np.asarray(f.forecast_mean_path(S, t=4, H=3, info=None), dtype=float)
    assert np.allclose(mu_default, 2.0, atol=0.0, rtol=0.0)


def test_naive_forecaster_ignores_noncanonical_history_keys() -> None:
    f = NaiveForecaster(default_value=6.0)
    S = np.array([0.0], dtype=float)

    mu = np.asarray(f.forecast_mean_path(S, t=4, H=4, info={"demand_history": [3.0, 5.0, 9.0]}), dtype=float)
    assert np.allclose(mu, 6.0, atol=0.0, rtol=0.0)


def test_naive_forecaster_is_available_from_core_compat_path() -> None:
    from inventory.core.forecasters.naive import NaiveForecaster as CoreNaiveForecaster

    f = CoreNaiveForecaster(default_value=4.0)
    mu = np.asarray(f.forecast_mean_path(np.array([0.0], dtype=float), t=0, H=2, info=None), dtype=float)
    assert np.allclose(mu, 4.0, atol=0.0, rtol=0.0)


def test_rolling_mean_forecaster_averages_recent_history() -> None:
    f = RollingMeanForecaster(window_size=3, default_value=5.0)
    S = np.array([0.0], dtype=float)

    mu_recent = np.asarray(f.forecast_mean_path(S, t=4, H=4, info={"demand_history": [2.0, 4.0, 8.0, 10.0]}), dtype=float)
    assert np.allclose(mu_recent, (4.0 + 8.0 + 10.0) / 3.0, atol=0.0, rtol=0.0)

    mu_fallback = np.asarray(f.forecast_mean_path(S, t=4, H=2, info={"last_demand": 9.0}), dtype=float)
    assert np.allclose(mu_fallback, 9.0, atol=0.0, rtol=0.0)

    mu_default = np.asarray(f.forecast_mean_path(S, t=4, H=3, info=None), dtype=float)
    assert np.allclose(mu_default, 5.0, atol=0.0, rtol=0.0)


def test_rolling_mean_forecaster_is_available_from_core_compat_path() -> None:
    from inventory.core.forecasters.naive import RollingMeanForecaster as CoreRollingMeanForecaster

    f = CoreRollingMeanForecaster(window_size=2, default_value=4.0)
    mu = np.asarray(f.forecast_mean_path(np.array([0.0], dtype=float), t=0, H=2, info={"demand_history": [3.0, 7.0]}), dtype=float)
    assert np.allclose(mu, 5.0, atol=0.0, rtol=0.0)


def test_multi_regime_ar1_forecaster_contract_and_lag_usage() -> None:
    exo = ExogenousPoissonMultiRegime(season_index=1, day_index=2, weather_index=3, season_period=90)
    adapter = MultiRegimeAr1FeatureAdapter(exo, lag_scale=100.0)
    X, y = adapter.generate_dataset(n_samples=300, seed=5, season0=2, day0=0, weather0=2)

    f = MlAr1RegimeDemandForecaster(adapter, model_type="linear", random_state=0)
    f.fit(X, y)

    S = np.array([300.0, 2.0, 0.0, 2.0], dtype=float)
    H = 4

    mu_default = np.asarray(f.forecast_mean_path(S, t=0, H=H, info=None), dtype=float)
    _assert_mean_path_contract(mu_default, H)

    mu_low = np.asarray(f.forecast_mean_path(S, t=0, H=H, info={"last_demand": 150.0}), dtype=float)
    mu_high = np.asarray(f.forecast_mean_path(S, t=0, H=H, info={"last_demand": 850.0}), dtype=float)

    _assert_mean_path_contract(mu_low, H)
    _assert_mean_path_contract(mu_high, H)
    assert not np.allclose(mu_low, mu_high)


def test_multi_regime_ml_forecaster_supports_elasticnet() -> None:
    exo = ExogenousPoissonMultiRegime(season_index=1, day_index=2, weather_index=3, season_period=90)
    adapter = MultiRegimeFeatureAdapter(exo)

    f = MlRegimeDemandForecaster(adapter, model_type="elasticnet", random_state=0)
    f.fit_from_exogenous(n_samples=300, seed=7, val_samples=120, val_seed=9)

    state = np.array([300.0, 2.0, 0.0, 2.0], dtype=float)
    H = 4
    mu = np.asarray(f.forecast_mean_path(state, t=0, H=H), dtype=float)

    _assert_mean_path_contract(mu, H)
    assert f.fit_report.get("model_type") == "elasticnet"


def test_quantile_boosting_regime_forecaster_contract() -> None:
    exo = ExogenousPoissonMultiRegime(season_index=1, day_index=2, weather_index=3, season_period=90)
    adapter = MultiRegimeFeatureAdapter(exo)

    f = QuantileBoostingRegimeDemandForecaster(adapter, quantile=0.75, random_state=0, n_estimators=80, max_depth=2)
    f.fit_from_exogenous(n_samples=300, seed=7, val_samples=120, val_seed=9)

    state = np.array([300.0, 2.0, 0.0, 2.0], dtype=float)
    H = 4
    mu = np.asarray(f.forecast_mean_path(state, t=0, H=H), dtype=float)

    _assert_mean_path_contract(mu, H)
    assert abs(float(f.fit_report.get("quantile", 0.0)) - 0.75) < 1e-12
    assert f.fit_report.get("model_type") in {"quantile_boosting", "linear"}


def test_sarimax_regime_forecaster_contract() -> None:
    exo = ExogenousPoissonMultiRegime(season_index=1, day_index=2, weather_index=3, season_period=90)
    adapter = MultiRegimeFeatureAdapter(exo)

    f = SarimaxRegimeDemandForecaster(adapter, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), trend="c")
    f.fit_from_exogenous(n_samples=120, seed=7, val_samples=60, val_seed=9)

    state = np.array([300.0, 2.0, 0.0, 2.0], dtype=float)
    H = 4
    mu = np.asarray(f.forecast_mean_path(state, t=0, H=H), dtype=float)

    _assert_mean_path_contract(mu, H)
    assert f.fit_report.get("model_type") == "sarimax"
    assert tuple(f.fit_report.get("order", ())) == (1, 0, 0)


def test_ets_forecaster_contract() -> None:
    exo = ExogenousPoissonMultiRegime(season_index=1, day_index=2, weather_index=3, season_period=90)
    adapter = MultiRegimeFeatureAdapter(exo)

    f = EtsDemandForecaster(adapter, trend="add", seasonal=None, damped_trend=True)
    f.fit_from_exogenous(n_samples=120, seed=7, val_samples=40)

    state = np.array([300.0, 2.0, 0.0, 2.0], dtype=float)
    H = 4
    mu = np.asarray(f.forecast_mean_path(state, t=0, H=H), dtype=float)

    _assert_mean_path_contract(mu, H)
    assert f.fit_report.get("model_type") == "ets"
    assert f.fit_report.get("trend") == "add"
