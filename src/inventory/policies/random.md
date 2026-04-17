# Random inventory policies

This note explains the stochastic policy implemented in
`inventory.policies.random` (see [random.py](random.py)).

This policy is meant to be simple, interpretable, and useful as a
**stochastic benchmark**. In contrast to deterministic baseline policies such
as `OrderUpToPolicy`, it samples actions from a chosen probability
distribution.

---

## Common conventions

- **State**: the policy receives the full state vector, but does not use it
  directly when sampling an action.
- **Action**: returned as a length-1 vector, where `action[0]` is the order
  quantity.
- The returned order quantity always lies on the discrete grid from `x_min` to
  `x_max` in increments of `dx`.
- The sampled action is returned as a float array of length 1:
  `np.array([x], dtype=float)`.

### Random-order idea (the concept)

A random-order policy chooses an order quantity by sampling from a probability
distribution over the feasible action space, rather than from a structured
inventory rule like a base-stock target.

In this implementation, the feasible action grid is

$$
\{x_{\min}, x_{\min} + dx, x_{\min} + 2dx, \dots, x_{\max}\}.
$$

- In **equal** mode, every feasible grid point is equally likely.
- In **normal** mode, actions are sampled around a chosen mean and then mapped
  back to the feasible grid.

This makes the policy useful as a random baseline, a stress test for the
evaluation pipeline, or a simple stochastic heuristic.

## `OrderRandomPolicy`

This policy samples an order quantity from the discrete action grid above. It
supports two distribution modes:

- **`distr="equal"`**  
  Samples uniformly across all admissible grid points.

- **`distr="normal"`**  
  Samples from a normal distribution with mean `x_mean` and standard deviation `x_std`, then:
  1. clips the draw to the interval `[x_min, x_max]`,
  2. snaps the result to the nearest action grid point.

## `OrderRandomPolicy` pseudocode

Inputs:

- optional `info["deterministic"]`
- optional `info["crn_step_seed"]`
- action grid defined by `x_min`, `x_max`, `dx`
- distribution parameters `distr`, and possibly `x_mean`, `x_std`

Algorithm:

```text
function ACT_OrderRandomPolicy(state, t, info=None):
    deterministic ← bool(info.get("deterministic", False))
    step_seed ← info.get("crn_step_seed", None)

    if deterministic is True:
        require step_seed is available
        rng ← generator built from step_seed (and optional base seed)
        return SAMPLE_FROM_DISTRIBUTION(rng)

    if step_seed is available:
        rng ← generator built from step_seed (and optional base seed)
        return SAMPLE_FROM_DISTRIBUTION(rng)

    rng ← internal policy RNG
    return SAMPLE_FROM_DISTRIBUTION(rng)


function SAMPLE_FROM_DISTRIBUTION(rng):
    if distr == "equal":
        k ← uniform integer from {0, ..., n_actions - 1}
        x ← grid[k]
        return [x]

    if distr == "normal":
        x_raw ← Normal(mean=x_mean, std=x_std)
        x_clip ← clip(x_raw, x_min, x_max)
        k ← round((x_clip - x_min) / dx)
        k ← clamp(k, 0, n_actions - 1)
        x ← grid[k]
        return [x]
```

### `OrderRandomPolicy` notes (implementation details)

- The state `state` is not used directly in this policy; the action depends only
  on the configured distribution and the random seed stream.
- In deterministic mode, the policy is still random within a run, but
  reproducible across repeated runs with the same CRN step seeds.
- If a CRN step seed is available, the policy builds a local generator from that
  seed, optionally mixed with the policy's own base seed.
- For `distr="normal"`, the draw is not returned directly; it is first clipped
  to `[x_min, x_max]` and then projected onto the feasible discrete action
  grid.
- The class implements `__repr__`, so notebook output and printed policy
  dictionaries remain readable.

## Parameters

- **`x_min`**  
  Lower bound of the admissible order quantity grid.

- **`x_max`**  
  Upper bound of the admissible order quantity grid.

- **`dx`**  
  Step size of the discrete action grid.

- **`seed`**  
  Optional base seed for reproducibility.

- **`distr`**  
  Distribution mode. Must be either `"equal"` or `"normal"`.

- **`x_mean`**  
  Mean of the normal distribution. Required when `distr="normal"`.

- **`x_std`**  
  Standard deviation of the normal distribution. Required when `distr="normal"` and must be positive.

## Validation

The class performs several consistency checks during initialization:

- `dx > 0`
- `x_max >= x_min`
- `x_max - x_min` must be an integer multiple of `dx`
- `distr` must be either `"equal"` or `"normal"`
- if `distr="normal"`:
  - `x_mean` must be provided
  - `x_std` must be provided
  - `x_std > 0`

These checks ensure that the action grid is well-defined and that sampling stays
meaningful.

---

## How to choose parameters (quick intuition)

- `x_min` and `x_max` define the feasible action range.
- `dx` controls the granularity of the order grid.
- `distr="equal"` is useful when you want a neutral random baseline.
- `distr="normal"` is useful when you want random actions centered around a
  plausible operating point.
- `x_std` controls how dispersed the normal-mode actions are; small values keep
  actions concentrated near `x_mean`, while large values produce broader
  exploration.

## Example usage

### Uniform random ordering

```python
from inventory.policies.random import OrderRandomPolicy

policy = OrderRandomPolicy(
    x_min=0.0,
    x_max=480.0,
    dx=10.0,
    seed=42,
    distr="equal",
)
```

### Normally distributed random ordering

```python
from inventory.policies.random import OrderRandomPolicy

policy = OrderRandomPolicy(
    x_min=0.0,
    x_max=480.0,
    dx=10.0,
    seed=42,
    distr="normal",
    x_mean=300.0,
    x_std=60.0,
)
```

---

## Typical use case

`OrderRandomPolicy` is mainly useful for:

- stochastic baselines,
- stress testing evaluation pipelines,
- comparing structured policies against uninformed randomized behavior,
- demonstrating CRN reproducibility with random actions.

It is generally not meant to be an optimized inventory control rule, but rather
a simple and interpretable benchmark.