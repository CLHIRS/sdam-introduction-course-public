# Policy function approximations (PFAs) for inventory

This note explains the PFA (policy function approximation) policies implemented in
`inventory.policies.pfa` (see [pfa.py](pfa.py)):

- `OrderUpToBlackboxPFA`
- `OrderUpToRegimeTablePFA`
- `OrderUpToStateDependentPFA`

The goal is to make the *ideas* clear (what concept each class represents) and
also to match the *implementation* (what the code actually does).

## Common conventions

- **State**: inventory on hand is stored in `state[0]`.
- **Action**: returned as a length-1 vector, where `action[0]` is the order quantity.
- **Order-up-to (base-stock) rule**: order enough to reach a target level $S$:

$$x = \max(0,\, S - inv)$$

- **Action feasibility / realism** (used in these classes):
  - cap: `x ≤ x_max`
  - batch size: `x` is rounded to multiples of `dx` via `dx * round(x/dx)`

## What “PFA” means here

A PFA is a policy written as a function with tunable parameters:

$$x_t = \pi(S_t, t\,;\,\theta)$$

Instead of solving for an optimal action table (like DP), we choose a *family* of
simple policies (parameterizations) and then tune the parameters $\theta$.

In this repo, the PFA classes are built around the same “order-up-to” idea, but
with different ways to define the target $S$.

---

## `OrderUpToBlackboxPFA`

### `OrderUpToBlackboxPFA` concept

A **1-parameter** policy:

- parameter: `theta[0]` (interpreted as the target level $S$)
- decision rule: order up to that target

It also includes simple derivative-free / stochastic optimization helpers that
use **strict CRN** (common random numbers) when estimating the objective.

### `OrderUpToBlackboxPFA` parameterization

- $S = \theta_0$
- $x = \max(0, S - inv)$, then apply constraints (`x_max`, `dx`)

### `OrderUpToBlackboxPFA` pseudocode (action)

Inputs:

- inventory `inv = state[0]`
- parameter vector `theta` with shape `(1,)`
- caps `x_max`, `dx`

Algorithm:

```text
function ACT_OrderUpToBlackboxPFA(state, t, info=None):
    inv ← float(state[0])
    target ← theta[0]

    x ← max(0, target - inv)
    x ← min(x, x_max)
    x ← ROUND_TO_GRID(x, dx)     # dx * round(x/dx) (if dx>1)

    return [x]
```

### `OrderUpToBlackboxPFA` pseudocode (objective under strict CRN)

The optimization routines repeatedly evaluate the mean total cost across multiple
episodes using the *same* per-step random seeds for fair comparisons.

```text
function OBJECTIVE(theta_candidate, S0, step_seed_paths, T, info=None):
    temporarily set self.theta ← theta_candidate (project into [theta_min, theta_max])

    totals ← []
    for each step_seeds in step_seed_paths:
        traj, costs, exog, meta ← system.simulate_crn(policy=self, S0, step_seeds, info)
        totals.append( sum(costs) )

    restore original self.theta
    return mean(totals)
```

### Optimization helpers (high level)

- **FD** (`estimate_gradient_fd`): finite-difference gradient estimate
- **SPSA** (`_optimize_spsa`): simultaneous perturbation gradient estimate

Both methods:

- sample `n_episodes` seed paths
- estimate a gradient-like direction using the same seed paths (strict CRN)
- take a step `theta ← theta - alpha * grad`
- clamp (`project`) theta back into bounds

---

## `OrderUpToRegimeTablePFA`

### `OrderUpToRegimeTablePFA` concept

A **K-parameter** policy that uses a *lookup table* by regime:

- `theta[r]` is the target level for regime `r`
- the regime index is read from the state: `state[regime_index]`

This is a very common “structured” PFA: more flexible than a single global target,
but still interpretable.

### `OrderUpToRegimeTablePFA` parameterization

- $S(r) = \theta_r$
- regime is discretized by rounding: `r = int(round(state[regime_index]))`
- `r` is clamped into valid range `[0, K-1]`

### `OrderUpToRegimeTablePFA` pseudocode (action)

Inputs:

- inventory `inv = state[0]`
- regime raw value `state[regime_index]`
- parameter vector `theta` with length `K`

Algorithm:

```text
function ACT_OrderUpToRegimeTablePFA(state, t, info=None):
    inv ← float(state[0])

    r_raw ← float(state[regime_index])
    r ← int(round(r_raw))
    r ← clamp(r, 0, K-1)

    target ← theta[r]

    x ← max(0, target - inv)
    x ← min(x, x_max)
    x ← ROUND_TO_GRID(x, dx)

    return [x]
```

---

## `OrderUpToStateDependentPFA`

### `OrderUpToStateDependentPFA` concept

A **state-dependent** policy that *learns* a mapping:

$$S \approx f(\phi(S_t, t))$$

where:

- $\phi(S_t, t)$ is a feature vector (hand-crafted or provided)
- $f$ is a regressor (scikit-learn model by default)
- the output is an *implied target level* `target_level`, which is then converted
  into an action using the same order-up-to rule

The training method is **rollout-based policy improvement**:

1) simulate trajectories using a behavior policy
2) at visited states, evaluate several candidate targets using short rollouts
3) label each visited state with the best-performing candidate target
4) fit a regressor to predict that best target from features
5) iterate

### Key moving parts in code

- `feature_fn(state, t) -> features` (defaults to `[inv, reg, t]`)
- `regressor.fit(X, y)` / `regressor.predict(X)`
- a short rollout horizon `H` (default 5)
- feasibility constraints via `_project_target()` and `_action_from_target()`

### `OrderUpToStateDependentPFA` pseudocode (action)

```text
function ACT_OrderUpToStateDependentPFA(state, t, info=None):
    inv ← float(state[0])

    if is_fitted is False:
        # fallback target until a model is trained
        if target_max exists:
            target ← 0.5 * (target_min + target_max)
        else:
            target ← target_min
    else:
        target ← PREDICT_TARGET(state, t)

    x ← ACTION_FROM_TARGET(inv, target)
    return [x]
```

### `OrderUpToStateDependentPFA` pseudocode (predict target)

```text
function PREDICT_TARGET(state, t):
    feats ← feature_fn(state, t)
    pred  ← regressor.predict(feats)
    return PROJECT_TARGET(pred)
```

Where `PROJECT_TARGET` enforces bounds and integer rounding.

### `OrderUpToStateDependentPFA` pseudocode (training data collection)

This matches `collect_training_data()`.

```text
function COLLECT_TRAINING_DATA(S0, T, n_episodes, seed0,
                               behavior_policy,
                               base_policy_for_rollout,
                               candidate_targets,
                               info=None):

    candidate_targets ← [PROJECT_TARGET(z) for z in candidate_targets]

    X_rows ← []
    y_rows ← []

    for episode in 1..n_episodes:
        step_seeds ← system.sample_crn_step_seeds(ep_seed, T)
        traj, costs, exog, meta ← system.simulate_crn(behavior_policy, S0, step_seeds, info)

        for t in 0..T-1:
            state ← traj[t]

            # Evaluate each candidate target with the SAME suffix seeds (strict CRN)
            vals ← []
            for z in candidate_targets:
                v ← EVALUATE_CANDIDATE_TARGET_AT_STATE(state, t, z,
                                                     base_policy_for_rollout,
                                                     step_seeds,
                                                     info)
                vals.append(v)

            best_target ← candidate_targets[argmin(vals)]

            X_rows.append( feature_fn(state, t) )
            y_rows.append( best_target )

    X ← stack_rows(X_rows)
    y ← array(y_rows)
    return (X, y)
```

### `OrderUpToStateDependentPFA` pseudocode (candidate evaluation)

This matches `_evaluate_candidate_target_at_state()`.

```text
function EVALUATE_CANDIDATE_TARGET_AT_STATE(state, t, target, base_policy, step_seeds, info=None):
    inv ← float(state[0])

    x0 ← ACTION_FROM_TARGET(inv, target)

    # rollout policy: take x0 now, then follow base_policy
    pol ← OneStepOverridePolicy(base_policy, x0)

    suffix ← step_seeds[t : t+H]
    if suffix is empty:
        return 0

    traj, costs, exog, meta ← system.simulate_crn(pol, state, suffix, info)
    return sum(costs)
```

### `OrderUpToStateDependentPFA` pseudocode (fit loop)

This matches `fit_via_rollout_improvement()`.

```text
function FIT_VIA_ROLLOUT_IMPROVEMENT(S0, T, n_episodes, seed0,
                                    n_iter,
                                    candidate_targets=None,
                                    behavior_policy=None,
                                    info=None):

    if candidate_targets is None:
        candidate_targets ← linspace(target_min, target_max, 21)

    if behavior_policy is None:
        mid ← 0.5 * (target_min + target_max)
        behavior_policy ← OrderUpToPolicy(target_level=mid, x_max=x_max, dx=dx)

    base_for_rollout ← (self if is_fitted else behavior_policy)

    for it in 0..n_iter-1:
        X, y ← COLLECT_TRAINING_DATA(S0, T, n_episodes, seed0 + 1000*it,
                                     behavior_policy,
                                     base_for_rollout,
                                     candidate_targets,
                                     info)

        if X is non-empty:
            regressor.fit(X, y)
            is_fitted ← True

        # next iteration uses the learned policy as behavior
        behavior_policy ← self
        base_for_rollout ← self

    return self
```

---

## Notebook-compatible aliases

The module also defines aliases used by older lecture notebooks:

- `OrderUpTo_blackbox_PFA = OrderUpToBlackboxPFA`
- `OrderUpTo_regime_table_PFA = OrderUpToRegimeTablePFA`
- `OrderUpTo_state_dependent_PFA_2 = OrderUpToStateDependentPFA`

If you see older code importing these names, they are intentionally kept working.

## Learned vs fixed (quick summary)

| Class | Learned parameters | Fixed / chosen by you |
| --- | --- | --- |
| `OrderUpToBlackboxPFA` | `theta[0]` (target level) | `x_max`, `dx`, `theta_min/max`, optimizer settings (`n_iter`, `alpha0`, etc.) |
| `OrderUpToRegimeTablePFA` | `theta[r]` for each regime | `regime_index`, `x_max`, `dx`, `theta_min/max` |
| `OrderUpToStateDependentPFA` | regressor model parameters (mapping features → target) | `feature_fn`, `H`, `candidate_targets`, feasibility bounds (`target_min/max`), `x_max`, `dx`, optional `s_max` |

In all cases, the training/evaluation loops are written in **cost minimization**
terms (lower total cost is better).
