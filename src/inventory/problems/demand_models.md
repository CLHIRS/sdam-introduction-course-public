# Demand (exogenous) models for inventory

This note explains the demand / exogenous models implemented in
`inventory.problems.demand_models` (see [demand_models.py](demand_models.py)).

These classes implement the `ExogenousModel` interface used by `DynamicSystemMVP`:

```text
sample(S_t, X_t, t, rng) -> W_{t+1}
```

In the inventory lectures, we interpret `W_{t+1}` as “demand information revealed
between $t$ and $t+1$”.

## Common conventions

- `state`, `action`, and `exog` are always **1-D NumPy arrays** (vectors).
- Demand is stored in the first component:
  - `exog[0] = D_{t+1}`
- Most demand models return `W_{t+1} = [D_{t+1}]`.
- Regime-based models return `W_{t+1} = [D_{t+1}, R_{t+1}, ...]`.

## Strict CRN note (why `rng` is an argument)

For fair comparisons between policies, `DynamicSystemMVP` may run **strict common
random numbers (CRN)** experiments.

- The simulator controls randomness by constructing `rng` from a per-step seed.
- The exogenous model must use only the provided `rng` for randomness.

That is exactly why `sample(..., rng)` is part of the official API.

---

## `PoissonConstantDemand`

### `PoissonConstantDemand` concept

Demand is i.i.d. Poisson with a constant mean:

$$D_{t+1} \sim \text{Poisson}(\lambda)$$

This is the simplest “stationary demand” model.

### `PoissonConstantDemand` interface

Constructor parameters:

- `lam`: the Poisson mean $\lambda$ (constant in time)
- `lam_min`: clamp to keep $\lambda_{eff} = \max(\text{lam\_min}, \text{lam})$ (default 0)
- legacy aliases: `lambda0` or `lam0` (at most one)

Sampling output:

- `sample(...) -> np.array([D], dtype=float)`

### `PoissonConstantDemand` pseudocode

```text
function INIT_PoissonConstantDemand(lam=300, lam_min=0, lambda0=None, lam0=None):
    if lambda0 and lam0 are both provided:
        error
    if lambda0 provided: lam ← lambda0
    if lam0 provided:    lam ← lam0

    store lam, lam_min
    if lam_min < 0: error

function LAMBDA_T(t):
    return max(lam_min, lam)

function SAMPLE_PoissonConstantDemand(state, action, t, rng):
    lam_eff ← LAMBDA_T(t)
    D ← Poisson(lam_eff) using rng
    return [D]
```

### `PoissonConstantDemand` notes (implementation details)

- In the current code, `lambda_t(t)` ignores `t` and returns the clamped constant.
- Even though the signature includes `(state, action, t)`, the model is
  intentionally independent of these (useful as a clean baseline).

---

## `PoissonSeasonalDemand`

### `PoissonSeasonalDemand` concept

Demand is Poisson, but the mean varies with time via a sinusoidal seasonality,
plus optional “spike” periods:

$$D_{t+1} \sim \text{Poisson}(\lambda(t))$$

where

$$\lambda(t) = \max\Big(\text{lam\_min},\; \text{base} + \text{amp}\,\sin(2\pi t/\text{period}) + \text{spike}(t)\Big)$$

and

$$\text{spike}(t) = \begin{cases}
\text{spike\_add} & \text{if } t>0 \text{ and } t \bmod \text{spike\_every} = 0 \\
0 & \text{otherwise}
\end{cases}$$

### `PoissonSeasonalDemand` interface

Constructor parameters:

- `base`: baseline mean level
- `amp`: amplitude of the sinusoid
- `period`: season length
- `spike_every`: spike frequency (0 disables spikes)
- `spike_add`: spike magnitude
- `lam_min`: lower bound on $\lambda(t)$
- legacy aliases for `base`: `lambda0` or `lam0` (at most one)

Output:

- `sample(...) -> [D]`

### `PoissonSeasonalDemand` pseudocode

```text
function LAMBDA_T(t):
    seasonal ← base + amp * sin(2π * t / period)
    spike ← spike_add if (spike_every > 0 and t > 0 and t mod spike_every == 0) else 0
    return max(lam_min, seasonal + spike)

function SAMPLE_PoissonSeasonalDemand(state, action, t, rng):
    lam ← LAMBDA_T(t)
    D ← Poisson(lam) using rng
    return [D]
```

### `PoissonSeasonalDemand` notes

- Spikes occur at times `t` such that `t % spike_every == 0` and `t > 0`.
- This model is time-dependent but still independent of `(state, action)`.

---

## `PoissonRegimeDemand`

### `PoissonRegimeDemand` concept

Here demand depends on an observable regime (a Markov chain). You can interpret
this as “business conditions” (low/normal/high), “season class”, etc.

Regime evolves as:

$$R_{t+1} \sim P(\cdot \mid R_t)$$

Demand uses the *next* regime:

$$D_{t+1} \sim \text{Poisson}(\lambda[R_{t+1}])$$

The exogenous vector contains both demand and the next regime:

$$W_{t+1} = [D_{t+1},\; R_{t+1}]$$

### `PoissonRegimeDemand` interface

Constructor parameters:

- `lam_by_regime`: vector of Poisson means (length $K$)
- `P`: transition matrix shape `(K, K)`, rows sum to 1
- `regime_index`: where to read the current regime `R_t` from the state (default 1)

Output:

- `sample(...) -> [D, r_next]`

### `PoissonRegimeDemand` pseudocode

```text
function SAMPLE_PoissonRegimeDemand(state, action, t, rng):
    r_t ← round(state[regime_index])
    r_t ← clamp(r_t to {0, ..., K-1})

    r_next ← categorical_sample(P[r_t, :]) using rng

    lam ← lam_by_regime[r_next]
    D ← Poisson(lam) using rng

    return [D, r_next]
```

### `PoissonRegimeDemand` notes

- The regime is stored as a float in the state vector, but used as an integer.
- The regime is *observable* because it is part of the state.
- The model uses `R_{t+1}` (not `R_t`) to generate `D_{t+1}`.

---

## `PoissonMultiRegimeDemand`

### `PoissonMultiRegimeDemand` concept

This generalizes the single-regime model to multiple observable regime
components. In the current implementation there are three:

- **season** (slow-moving, optionally “sticky” for many steps)
- **day-of-week** (fast-moving cycle)
- **weather** (fast-moving Markov chain)

Each regime has its own transition matrix, and the resulting demand mean is a
combination of a season baseline and additive coefficients:

$$\lambda_{eff} = \lambda_{base\_season}[r_{season,next}]\cdot \Big(1 + c_{day}[r_{day,next}] + c_{weather}[r_{weather,next}]\Big)$$

Then

$$D_{t+1} \sim \text{Poisson}(\max(\text{lam\_min}, \lambda_{eff}))$$

The exogenous vector includes demand and all next regimes:

$$W_{t+1} = [D_{t+1},\; r_{season,next},\; r_{day,next},\; r_{weather,next}]$$

### `PoissonMultiRegimeDemand` interface

Constructor parameters (key ones):

- transition matrices: `P_season`, `P_day`, `P_weather`
- season baselines: `lambda_base_season`
- additive coefficients: `lambda_coeff_day`, `lambda_coeff_weather`
- indices in the state vector: `season_index`, `day_index`, `weather_index`
- `season_period`: controls how often season can change
- `lam_min`: lower bound on effective mean

Output:

- `sample(...) -> [D, r_season_next, r_day_next, r_weather_next]`

### `PoissonMultiRegimeDemand` pseudocode

```text
function SAMPLE_NEXT_REGIME(P, r_t, rng):
    return categorical_sample(P[r_t, :]) using rng

function SAMPLE_PoissonMultiRegimeDemand(state, action, t, rng):
    # read and clamp current regimes from state
    r_season_t  ← clamp(round(state[season_index]),  K_season)
    r_day_t     ← clamp(round(state[day_index]),     K_day)
    r_weather_t ← clamp(round(state[weather_index]), K_weather)

    # season changes only when (t+1) hits multiples of season_period
    if season_period > 0 and (t+1) mod season_period != 0:
        r_season_next ← r_season_t
    else:
        r_season_next ← SAMPLE_NEXT_REGIME(P_season, r_season_t, rng)

    r_day_next ← SAMPLE_NEXT_REGIME(P_day, r_day_t, rng)
    r_weather_next ← SAMPLE_NEXT_REGIME(P_weather, r_weather_t, rng)

    lam_eff ← lambda_base_season[r_season_next] * (1 + coeff_day[r_day_next] + coeff_weather[r_weather_next])
    lam_eff ← max(lam_min, lam_eff)

    D ← Poisson(lam_eff) using rng

    return [D, r_season_next, r_day_next, r_weather_next]
```

### `PoissonMultiRegimeDemand` notes

- The helper `_validate_transition_matrix` enforces that each transition matrix
  is square, nonnegative, and row-stochastic.
- The code uses a **simple bucketing** of regimes by rounding floats to ints.
- `season_period` is implemented using `(t+1) % season_period`:
  - if it is not a “season change” time, season stays fixed
  - otherwise it transitions via the season Markov chain

---

## Backwards-compatible aliases

The module also defines older names used in notebooks:

- `ExogenousPoissonConstant` → `PoissonConstantDemand`
- `ExogenousPoissonSeasonal` → `PoissonSeasonalDemand`
- `ExogenousPoissonRegime` → `PoissonRegimeDemand`
- `ExogenousPoissonMultiRegime` → `PoissonMultiRegimeDemand`
