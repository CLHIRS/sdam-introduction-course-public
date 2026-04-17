# Forecasters in This Repository

This page connects forecasting theory to the concrete forecaster classes implemented in `src/inventory/forecasters/`.

The goal for students is to answer two questions quickly:

1. What kind of forecasting idea is this method based on?
2. Which class in the repository actually implements that idea?

## 1. Forecasting interfaces used in the codebase

This repository uses two closely related forecasting interfaces.

### `DemandForecaster`

Used by inventory policies such as CFA-style and some forecast-aware policies.

- Main method: `forecast_mean_path(state, t, H, info=None) -> np.ndarray`
- Output: a mean-demand path of length `H`
- Typical use: provide the policy with a point forecast for the next `H` periods

Implemented by the classes in:

- `naive.py`
- `ml.py`
- `ts.py`

### `DemandPathForecaster`

Used when a policy wants both a mean path and an uncertainty path.

- Main methods:
  - `forecast_mean_path(state, t, H, info=None)`
  - `forecast_std_path(state, t, H, info=None)`
- Output: mean path plus standard-deviation path
- Typical use: hybrid DLA policies that reason over both expected demand and uncertainty

Implemented by the classes in:

- `path.py`

## 2. Implemented forecasters at a glance

The table below is the practical overview of what currently exists in the repository.

| Module | Class | Family | Main idea | Needs fitting? | Uses current state/info? | Typical teaching role |
| --- | --- | --- | --- | --- | --- | --- |
| `naive.py` | `ConstantMeanForecaster` | Heuristic baseline | Always predict one fixed mean | No | No | Simplest constant baseline |
| `naive.py` | `NaiveForecaster` | Heuristic baseline | Predict next demand as last realized demand | No | Uses `info["last_demand"]` | Classic persistence benchmark |
| `naive.py` | `RollingMeanForecaster` | Heuristic baseline | Predict with mean of recent realized demands | No | Uses `info["demand_history"]` or `info["last_demand"]` | Moving-average style baseline |
| `naive.py` | `ExogenousAwareMeanForecaster` | Expert/oracle-style benchmark | Read the mean structure directly from the exogenous demand model | No | Yes, for regime state when needed | Upper-reference benchmark for known demand structure |
| `naive.py` | `ExpertDemandForecasterConstant350` | Heuristic baseline | Backward-compatible constant forecaster | No | No | Legacy teaching/example helper |
| `ml.py` | `MlDemandForecaster` | Feature-based ML | Learn `phi(t) -> E[D_{t+1}]` for seasonal settings | Yes | Indirectly through engineered features | Generic seasonal ML forecaster |
| `ml.py` | `MlRegimeDemandForecaster` | Feature-based ML | Learn `phi(state, t) -> E[D_{t+1}]` for regime-aware settings | Yes | Yes | Main regime-aware ML mean forecaster |
| `ml.py` | `MlAr1RegimeDemandForecaster` | Feature-based ML plus lag | Regime-aware ML with one autoregressive demand lag | Yes | Yes, plus lag from `info` | Bridge from pure exogenous features to history-aware ML |
| `ml.py` | `QuantileBoostingRegimeDemandForecaster` | Nonlinear ML | Gradient-boosted quantile trees for regime-aware demand | Yes | Yes | Shows quantile-oriented tree ensembles |
| `ts.py` | `EtsDemandForecaster` | Classical time series | Exponential smoothing on a synthetic univariate demand path | Yes | No at forecast time | Classical pattern-based baseline |
| `ts.py` | `SarimaxRegimeDemandForecaster` | Classical time series | SARIMAX with regime/exogenous features from adapters | Yes | Yes | Classical linear dynamic model with exogenous structure |
| `path.py` | `ConstantMeanPathForecaster` | Path forecaster | Constant mean path plus simple std path | No | No | Simplest mean-plus-uncertainty path baseline |
| `path.py` | `SeasonalSinMeanPathForecaster` | Path forecaster | Hand-crafted sinusoidal mean path | No | Uses time index `t` | Simple seasonal path demo |
| `path.py` | `ExogenousMeanPathForecaster` | Oracle-style path forecaster | Read mean path directly from supported exogenous models | No | Yes, for regime state when needed | Expert path forecast for hybrid planning |

## 3. Adapters and helper utilities

Some forecasters are not standalone; they rely on feature adapters and helper builders.

### Feature adapters in `ml.py`

- `SeasonalFeatureAdapter`: builds time-based features for seasonal demand
- `RegimeFeatureAdapter`: builds regime-aware features for single-regime-state models
- `MultiRegimeFeatureAdapter`: builds richer regime-distribution features for multi-regime settings
- `MultiRegimeAr1FeatureAdapter`: extends multi-regime features with one demand lag

These adapters are what make the ML and SARIMAX forecasters state-aware: they define which information from `state`, `t`, and regime dynamics becomes model input.

### Factory helpers in `factory.py`

- `FitConfig`: small config object for synthetic fitting
- `make_adapter(...)`: create the default adapter for a supported exogenous model
- `make_ml_forecaster(...)`: create the default ML forecaster for an adapter
- `fit_ml_forecaster_from_exogenous(...)`: one-liner helper for adapter plus fitting
- `forecast_with_default_state(...)`: notebook convenience helper

These helpers are especially useful in teaching notebooks because they reduce setup noise.

## 4. Theory-to-code map: table of forecasting methods

The next table keeps the theoretical view, but now adds whether the method is currently implemented here and which class represents it.

| Family | Method | Intuition | Key idea | Strengths | Weaknesses | Best suited time-series characteristics | Complexity | Implemented here? | Repository class(es) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Heuristic baseline | **Naive / Last value** | "I assume tomorrow looks like today." | Forecast next value by the most recent observation | Extremely simple, strong benchmark, hard to beat on highly persistent series | Ignores trend, seasonality, covariates | Random-walk-like, highly persistent, short-horizon series | Very low | Yes | `NaiveForecaster` |
| Heuristic baseline | **Seasonal Naive** | "I assume the next period looks like the same period last season." | Repeat the last value from the same seasonal position | Excellent baseline for recurring business demand | Fails when seasonality shifts or trend changes strongly | Stable seasonality, repeated weekly/monthly cycles | Very low | Not yet | No direct class yet |
| Heuristic baseline | **Moving Average** | "I trust the recent past on average more than any single day." | Forecast with average of last `k` observations | Smooths noise, intuitive, easy to explain | Lags behind trend | Noisy but roughly stable series | Very low | Yes | `RollingMeanForecaster` |
| Heuristic baseline | **Drift / Trend Naive** | "I assume the recent direction of change continues a bit longer." | Extend recent linear change forward | Easy bridge to trend models | Sensitive to noise and structural breaks | Mild trend, short-horizon extrapolation | Very low | Not yet | No direct class yet |
| Heuristic / expert benchmark | **Constant mean** | "I assume demand fluctuates around one fixed level." | Predict the same mean every period | Simplest sanity check, very transparent | Too rigid when demand changes | Stable demand around one level | Very low | Yes | `ConstantMeanForecaster`, `ExpertDemandForecasterConstant350` |
| Heuristic / expert benchmark | **Exogenous mean oracle** | "I know the mean structure of the demand generator." | Query the exogenous model directly | Strong teaching benchmark, shows value of structural knowledge | Unrealistic if the true model is unknown | Synthetic teaching setups where the exogenous model is known | Very low | Yes | `ExogenousAwareMeanForecaster`, `ExogenousMeanPathForecaster` |
| Classical statistical | **ETS / Exponential Smoothing** | "I keep updating my belief about level, trend, and seasonality, giving more weight to recent observations." | Recursively update level, trend, and seasonality | Interpretable, practical, strong benchmark for many business series | Mostly pattern-based, limited use of rich external features | Smooth level/trend/seasonal structure | Low-medium | Yes | `EtsDemandForecaster` |
| Classical statistical | **SARIMAX** | "I predict from past values, past errors, seasonality, and external drivers." | Model autocorrelation, differencing, seasonality, and exogenous effects | Strong statistical foundation, interpretable, handles lag dependence explicitly | More tuning effort, mostly linear dynamics | Autocorrelated series with known external drivers | Medium | Yes | `SarimaxRegimeDemandForecaster` |
| Feature-based linear ML | **Linear regression** | "I predict with a weighted combination of engineered features." | Linear mapping from features to demand | Easy to inspect, strong baseline after feature engineering | Misses nonlinear effects | Mostly additive relationships with useful engineered features | Medium | Yes | `MlDemandForecaster(model_type="linear")`, `MlRegimeDemandForecaster(model_type="linear")`, `MlAr1RegimeDemandForecaster(model_type="linear")` |
| Feature-based linear ML | **Elastic Net** | "I predict with a weighted linear combination, but regularize coefficients." | Linear regression with combined L1 plus L2 penalty | Handles many features, reduces overfitting, supports feature selection | Still linear, needs feature engineering | Lag, calendar, and exogenous features with mostly linear effects | Medium | Yes | `MlDemandForecaster(model_type="elasticnet")`, `MlRegimeDemandForecaster(model_type="elasticnet")`, `MlAr1RegimeDemandForecaster(model_type="elasticnet")` |
| Feature-based nonlinear ML | **Decision Tree** | "I split history into similar situations and predict differently in each region." | Partition feature space into regions with simple predictions | Interpretable, nonlinear, easy first nonlinear model | Can be unstable alone | Threshold effects, small-to-medium datasets | Medium | Yes | `MlDemandForecaster(model_type="tree")`, `MlRegimeDemandForecaster(model_type="tree")`, `MlAr1RegimeDemandForecaster(model_type="tree")` |
| Feature-based nonlinear ML | **Gradient-Boosted Trees / Quantile Boosting** | "I build many small decision rules that successively fix earlier mistakes." | Sequential tree ensemble targeting a conditional quantile | Strong practical performance, captures nonlinearities and interactions | More hyperparameters, less transparent than linear models | Nonlinear demand with rich features, regimes, covariates | Medium | Yes | `QuantileBoostingRegimeDemandForecaster` |
| Deep learning | **MLP** | "I learn a flexible nonlinear mapping from features to future demand." | Neural network on engineered features | Flexible, natural next step after linear models | Needs tuning and more data | Nonlinear patterns with enough data | Medium | Yes | `MlDemandForecaster(model_type="mlp")`, `MlRegimeDemandForecaster(model_type="mlp")`, `MlAr1RegimeDemandForecaster(model_type="mlp")` |
| Kernel method | **Kernel Ridge Regression** | "I predict by comparing the current situation to similar past situations." | Nonlinear regression in kernel space | Smooth nonlinear modeling | Scaling and tuning can be harder | Moderate datasets with smooth nonlinear relationships | Medium | Not yet | No direct class yet |
| Kernel method | **SVR with Gaussian/RBF kernel** | "Nearby cases should predict similarly." | Support-vector regression in kernel space | Powerful on smaller structured datasets | Hyperparameter sensitive | Nonlinear series with engineered features | Medium-high | Not yet | No direct class yet |
| Kernel method | **Gaussian Process Regression** | "I learn a distribution over plausible functions." | Bayesian regression with covariance kernels | Natural uncertainty quantification | Poor scalability | Small datasets with strong uncertainty focus | High | Not yet | No direct class yet |
| Deep learning | **TCN** | "I scan the past with learned filters over time." | Convolutional sequence model over history | Captures temporal patterns over longer horizons | More data and tuning needed | Longer histories, more complex temporal structure | Medium-high | Not yet | No direct class yet |

## 5. Core forecasting methods for this repository

This table is the compact student view: which methods matter most in this repository, what they are good for, and which implementation to look at first.

| Family | Method | Intuition | Best for | Complexity | Core in this repo? | Recommended class |
| --- | --- | --- | --- | --- | --- | --- |
| Heuristic | **Naive** | "Tomorrow = today." | Strong persistence, short horizons | Very low | Yes | `NaiveForecaster` |
| Heuristic | **Moving Average** | "Average recent past." | Noisy but stable series | Very low | Yes | `RollingMeanForecaster` |
| Heuristic | **Constant Mean** | "Demand stays around one level." | Sanity checks, simple baselines | Very low | Yes | `ConstantMeanForecaster` |
| Heuristic / expert | **Exogenous mean oracle** | "Use the known demand generator mean." | Synthetic teaching benchmarks | Very low | Yes | `ExogenousAwareMeanForecaster` |
| Classical TS | **ETS** | "Update level, trend, seasonality." | Smooth business series with temporal structure | Low-medium | Yes | `EtsDemandForecaster` |
| Classical TS | **SARIMAX** | "Use lags, dynamics, and exogenous drivers." | Regime-aware or exogenous linear dynamics | Medium | Yes | `SarimaxRegimeDemandForecaster` |
| Linear ML | **Linear / Elastic Net** | "Weighted engineered features." | Strong interpretable ML baselines | Medium | Yes | `MlDemandForecaster`, `MlRegimeDemandForecaster` |
| Nonlinear ML | **Decision Tree** | "Split into similar cases." | Threshold effects, easy nonlinear baseline | Medium | Yes | `MlDemandForecaster(model_type="tree")`, `MlRegimeDemandForecaster(model_type="tree")` |
| Nonlinear ML | **Quantile Boosting** | "Many small trees fix earlier mistakes." | Rich nonlinear regime-aware demand | Medium | Yes | `QuantileBoostingRegimeDemandForecaster` |
| Deep learning | **MLP** | "Learn a flexible nonlinear mapping." | Larger datasets with engineered features | Medium | Optional | `MlDemandForecaster(model_type="mlp")`, `MlRegimeDemandForecaster(model_type="mlp")` |

## 6. How to choose among the implemented forecasters

Use the following simple progression when teaching or experimenting.

1. Start with `ConstantMeanForecaster` and `NaiveForecaster` to establish basic baselines.
2. Add `RollingMeanForecaster` when recent realized demand history should matter.
3. Use `ExogenousAwareMeanForecaster` or `ExogenousMeanPathForecaster` when you want an expert or oracle benchmark based on the known synthetic demand generator.
4. Use `EtsDemandForecaster` when you want a classical univariate time-series baseline.
5. Use `SarimaxRegimeDemandForecaster` when you want a classical statistical model that also uses regime-aware exogenous features.
6. Use `MlRegimeDemandForecaster` when you want a flexible feature-based mean forecaster.
7. Use `MlAr1RegimeDemandForecaster` when you want the ML forecaster to explicitly incorporate the latest realized demand.
8. Use `QuantileBoostingRegimeDemandForecaster` when nonlinear regime effects are important and you want a stronger tree-based method.

## 7. Important implementation notes

- `NaiveForecaster` reads from `info["last_demand"]`.
- `RollingMeanForecaster` prefers `info["demand_history"]` and falls back to `info["last_demand"]`.
- The ML forecasters and SARIMAX rely on feature adapters; they do not invent features automatically.
- `EtsDemandForecaster` is intentionally univariate at forecast time, so it does not directly react to the current regime state.
- `SarimaxRegimeDemandForecaster` is the main classical model here that explicitly uses regime and exogenous structure.
- The classes in `path.py` implement `DemandPathForecaster`, not the simpler `DemandForecaster` protocol.

## 8. Suggested student storyline

For teaching, the implemented forecasters support a clean storyline:

1. Baselines: constant, naive, rolling mean
2. Expert benchmark: exogenous or oracle mean forecast
3. Classical time series: ETS, then SARIMAX
4. Feature-based ML: linear, elastic net, decision tree, MLP
5. Stronger nonlinear regime-aware model: quantile boosting

That progression ties the repository implementation to the broader forecasting taxonomy without forcing students to jump directly from theory into an undifferentiated list of classes.
