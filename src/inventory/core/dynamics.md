# Dynamic systems and strict CRN evaluation

This note explains the class implemented in `inventory.core.dynamics` (see [dynamics.py](dynamics.py)):

- `DynamicSystemMVP`

It is written to be a “minimal but complete” simulation engine for the inventory
lectures, supporting:

- a clean vector-based state/action/exogenous convention
- simulation (rollouts)
- **strict CRN** (common random numbers) for fair Monte Carlo comparisons

---

## Big picture: what a dynamic system is

A (controlled) stochastic dynamic system evolves over time:

- **State**: $S_t$ (what we observe / track)
- **Action / decision**: $X_t$ (what the policy chooses)
- **Exogenous information**: $W_{t+1}$ (randomness arriving “from the outside”)
- **Cost**: $C_t$ (what we pay at time $t$)

In this codebase the model is:

- $W_{t+1} \sim \text{exogenous}(S_t, X_t, t)$
- $S_{t+1} = \text{transition}(S_t, X_t, W_{t+1}, t)$
- $C_t = \text{cost}(S_t, X_t, W_{t+1}, t)$

A **policy** is a function that chooses actions:

$$X_t = \pi(S_t, t)$$

---

## Vector conventions (very important)

To keep everything uniform (even for 1D problems), we use vector conventions:

- `State` $S$ is a 1-D NumPy array (shape `(dS,)`)
- `Action` $X$ is a 1-D NumPy array (shape `(dX,)`)
- `Exog` $W$ is a 1-D NumPy array (shape `(dW,)`)

Even if the dimension is 1, we still use a length-1 vector.

`DynamicSystemMVP` can optionally store dimension hints (`d_s`, `d_x`, `d_w`) to
validate shapes at runtime.

---

## What “strict CRN” means

When comparing two policies via Monte Carlo, noise from randomness can obscure
real differences.

**Strict CRN** (common random numbers) reduces this noise by ensuring that each
policy sees the *same random number stream per time step*.

Key implementation detail:

- We do **not** store and replay realized $W_{t+1}$.
- Instead, we store and replay **per-step RNG seeds**.

This remains valid even when $W_{t+1}$ depends on $(S_t, X_t, t)$, because each
policy will generate its own $W_{t+1}$ from the same per-step seed.

---

## Public API (what you call from notebooks)

`DynamicSystemMVP` exposes these core methods:

- `sample_crn_step_seeds(episode_seed, T) -> (T,) int64 array`
- `simulate(policy, S0, T=10, seed=None) -> (traj, costs, actions, step_seeds)`
- `simulate_crn(policy, S0, step_seeds) -> (traj, costs, actions, step_seeds)`
- `collect_policies_crn_rollouts_mc(policies, S0, T, n_episodes, seed0, ...) -> rollouts`
- `evaluate_policy_crn_mc(policy, S0, T, n_episodes, seed0) -> totals`
- `evaluate_policies_crn_mc(policies, S0, T, n_episodes, seed0) -> (results, rollouts)`

`results` is a dictionary mapping policy name to a `CRNEvaluationResult` record:

- `mean`: Monte Carlo mean of total cost
- `std`: Monte Carlo standard deviation of total cost
- `totals`: per-episode total costs (NumPy array)

The class also includes an internal unified rollout engine `_rollout(...)` that
all other functions build on.

---

## `DynamicSystemMVP` interfaces

When constructing the system, you provide three model components:

1. **Transition function**

```text
transition_func(S_t, X_t, W_{t+1}, t) -> S_{t+1}
```

1. **Cost function**

```text
cost_func(S_t, X_t, W_{t+1}, t) -> scalar cost
```

1. **Exogenous model**

The `ExogenousModel` is expected to implement:

```text
sample(S_t, X_t, t, rng) -> W_{t+1}
```

---

## Pseudocode: seed generation (`sample_crn_step_seeds`)

This method deterministically expands one episode seed into a vector of per-step
seeds.

```text
function SAMPLE_CRN_STEP_SEEDS(episode_seed, T):
    ss ← SeedSequence(episode_seed)
    children ← ss.spawn(T)

    step_seeds ← empty int64 array of length T
    for t in 0..T-1:
        step_seeds[t] ← children[t].generate_state(1)[0]

    return step_seeds
```

Interpretation:

- episode seed identifies the episode
- step seeds identify the random stream at each time step

---

## Pseudocode: the unified rollout engine (`_rollout`)

`_rollout(...)` is the heart of the class.

Inputs:

- `policy`: object with `policy.act(S_t, t, info) -> X_t`
- `S0`: initial state vector
- `T`: horizon length
- randomness control:
  - either provide `step_seeds` (strict CRN)
  - or provide `seed` (converted into step seeds)
  - or provide neither (uses the system’s internal root seed sequence)

Algorithm:

```text
function ROLLOUT(policy, S0, T, step_seeds=None, seed=None, info=None, return_exog=False):
    S0 ← as_1d_float_array(S0)
    assert dim(S0) matches d_s if d_s is set

    # 1) determine step seeds
    if step_seeds is provided:
        assert step_seeds is 1-D length T
    else if seed is provided:
        step_seeds ← SAMPLE_CRN_STEP_SEEDS(seed, T)
    else:
        step_seeds ← generate T seeds from internal root SeedSequence

    traj  ← empty float array of shape (T+1, d_s)
    costs ← empty float array of shape (T,)
    actions ← allocate later (once we know d_x)
    exogs   ← allocate later (once we know d_w) if return_exog

    traj[0] ← S0

    for t in 0..T-1:
        S_t ← copy(traj[t])

        # 2) create per-step info dict and include the step seed
        step_info ← {} if info is None else copy(info)
        step_info["crn_step_seed"] ← step_seeds[t]

        # 3) query the policy for action
        X_t ← policy.act(S_t, t, info=step_info)
        X_t ← as_1d_float_array(X_t)
        assert dim(X_t) matches d_x if d_x is set
        store X_t into actions[t]

        # 4) sample exogenous information using per-step RNG
        rng_t ← default_rng(step_seeds[t])
        W_{t+1} ← exogenous_model.sample(S_t, X_t, t, rng_t)
        W_{t+1} ← as_1d_float_array(W_{t+1})
        assert dim(W_{t+1}) matches d_w if d_w is set
        if return_exog:
            store W_{t+1} into exogs[t]

        # 5) compute cost and next state
        C_t ← float( cost_func(S_t, X_t, W_{t+1}, t) )
        S_{t+1} ← transition_func(S_t, X_t, W_{t+1}, t)
        S_{t+1} ← as_1d_float_array(S_{t+1})
        assert dim(S_{t+1}) matches d_s if d_s is set

        costs[t] ← C_t
        traj[t+1] ← S_{t+1}

    if return_exog:
        return (traj, costs, actions, step_seeds, exogs)

    return (traj, costs, actions, step_seeds)
```

Notes:

- The system stores *one trajectory* of states of length `T+1`.
- Costs are per-step costs of length `T`.
- Exogenous samples are indexed by `t` but represent $W_{t+1}$.

---

## Pseudocode: `simulate` and `simulate_crn`

These are thin wrappers around `_rollout`.

```text
function SIMULATE(policy, S0, T=10, seed=None, info=None):
    return ROLLOUT(policy, S0, T, seed=seed, info=info, return_exog=False)

function SIMULATE_CRN(policy, S0, step_seeds, info=None):
    T ← len(step_seeds)
    return ROLLOUT(policy, S0, T, step_seeds=step_seeds, info=info, return_exog=False)
```

---

## Pseudocode: Monte Carlo evaluation for one policy (`evaluate_policy_crn_mc`)

This evaluates one policy for many episodes under strict CRN.

Important “evaluation convention” used in the code:

- if `info` is not provided, it sets `info = {"deterministic": True}`
- if `info` is provided but does not include `"deterministic"`, it injects it

This is meant to keep evaluation reproducible unless training code explicitly
requests stochastic behavior.

```text
function EVALUATE_POLICY_CRN_MC(policy, S0, T=10, n_episodes=200, seed0=1234, info=None):
    info ← ensure info["deterministic"] = True

    rng ← default_rng(seed0)
    episode_seeds ← rng.integers(1, 2^31-1, size=n_episodes)

    totals ← empty float array length n_episodes

    for i in 0..n_episodes-1:
        step_seeds ← SAMPLE_CRN_STEP_SEEDS(episode_seeds[i], T)
        traj, costs, actions, step_seeds_out ← SIMULATE_CRN(policy, S0, step_seeds, info)
        totals[i] ← sum(costs)

    return totals
```

---

## Pseudocode: evaluating many policies (`evaluate_policies_crn_mc`)

This compares a set of policies on the same seed paths.

Outputs:

- `results`: per-policy `CRNEvaluationResult(mean, std, totals, runtime_sec)`
- `rollouts`: one shared reference rollout per policy (episode 0 seeds)

```text
function EVALUATE_POLICIES_CRN_MC(policies_dict, S0, T=10, n_episodes=200, seed0=1234, info=None):
    info ← ensure info["deterministic"] = True

    rng ← default_rng(seed0)
    episode_seeds ← rng.integers(1, 2^31-1, size=n_episodes)
    step_seed_paths ← [SAMPLE_CRN_STEP_SEEDS(ep_seed, T) for ep_seed in episode_seeds]

    results ← empty dict

    for (name, policy) in policies_dict:
        totals ← empty float array length n_episodes
        for i in 0..n_episodes-1:
            step_seeds ← step_seed_paths[i]
            traj, costs, actions, _ ← SIMULATE_CRN(policy, S0, step_seeds, info)
            totals[i] ← sum(costs)

        results[name] ← {
            mean: mean(totals),
            std:  std(totals) with ddof=1 (or 0 if only one episode),
            totals: totals,
            runtime_sec: elapsed evaluation/runtime for this policy
        }

    # One shared reference rollout (first step_seed_path) for plotting
    rollouts ← {}
    if n_episodes > 0:
        ref_step_seeds ← step_seed_paths[0]
        for (name, policy) in policies_dict:
            traj, costs, actions, _, ws ← ROLLOUT(policy, S0, T, step_seeds=ref_step_seeds, info=info, return_exog=True)
            rollouts[name] ← {traj, costs, actions, ws}

    return (results, rollouts)
```

---

## Pseudocode: collecting full rollout datasets (`collect_policies_crn_rollouts_mc`)

This is mainly for plotting and diagnostics: it returns *all* episode rollouts
(including per-step exogenous vectors if requested).

```text
function COLLECT_POLICIES_CRN_ROLLOUTS_MC(policies_dict, S0, T=10, n_episodes=50, seed0=1234, info=None, store_step_seeds=False):
    info ← ensure info["deterministic"] = True

    rng ← default_rng(seed0)
    episode_seeds ← rng.integers(1, 2^31-1, size=n_episodes)
    step_seed_paths ← [SAMPLE_CRN_STEP_SEEDS(ep_seed, T) for ep_seed in episode_seeds]

    rollouts_by_policy ← {name: [] for name in policies_dict}

    for (name, policy) in policies_dict:
        episodes ← []
        for step_seeds in step_seed_paths:
            traj, costs, actions, step_seeds_out, ws ← ROLLOUT(policy, S0, T, step_seeds=step_seeds, info=info, return_exog=True)

            ep ← {"traj": traj, "actions": actions, "costs": costs, "ws": ws}
            if store_step_seeds:
                ep["step_seeds"] ← step_seeds_out

            episodes.append(ep)

        rollouts_by_policy[name] ← episodes

    return rollouts_by_policy
```

---

## Common student questions (answers)

### Why does `W_{t+1}` depend on `(S_t, X_t, t)`?

Because many problems have *state-dependent uncertainty*. For example, demand can
change with season/regime, and those can be part of the state.

### Why store seeds instead of storing realized `W`?

- Storing realized `W` would force every policy to face the exact same $W$ values,
  which is not always consistent when $W$ depends on $S_t$ and $X_t$.
- Storing per-step seeds is a clean compromise: all policies use the same random
  numbers, but each policy’s own state/action affects what randomness becomes.

### What is `info` used for?

`info` is an optional dictionary passed to the policy.

- `DynamicSystemMVP` creates a per-step copy `step_info`.
- It always injects `step_info["crn_step_seed"]`.
- Evaluation helpers also inject `info["deterministic"] = True` by default.

This lets policies behave differently in training vs evaluation (for example,
turning exploration on/off).
