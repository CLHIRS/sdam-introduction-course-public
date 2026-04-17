# VFA policies (post-decision TD and fitted Q iteration)

This note explains the two VFA (value function approximation) policies implemented in
`inventory.policies.vfa` (see [vfa.py](vfa.py)):

- `PostDecisionGreedyVfaPolicy` (online greedy policy using a post-decision VFA)
- `FqiGreedyVfaPolicy` (batch/offline fitted Q iteration for a discrete action set)

Both are written in a “lecture style”:

- Keep the state/action/exogenous interface consistent with `DynamicSystemMVP`.
- Use a small discrete action set (`dx` step size up to `x_max`).
- Learn an approximate value function from simulated data.

Throughout, we use the cost-minimization view.

---

## Common conventions

### State and action

- `state` is a 1-D NumPy array.
- Inventory on hand is `state[0]`.
- The action is a length-1 array `action = [x]`, where `x` is a nonnegative order quantity.

### Discrete action set

Both policies restrict actions to a finite set:

$$\mathcal{X} = \{0, dx, 2dx, \dots, x_{max}\}$$

In code this is implemented by enumerating candidate orders.

### What the model must provide

Both policies hold a reference to the environment model:

- `model: DynamicSystemMVP`

They use:

- `model.exogenous_model.sample(S, X, t, rng) -> W_{t+1}`
- `model.cost_func(S, X, W_{t+1}, t) -> cost`
- `model.transition_func(S, X, W_{t+1}, t) -> S_{t+1}`
- `model.sample_crn_step_seeds(episode_seed, T) -> step_seeds` (for reproducible rollouts)

---

## When to use which (quick guide)

- Use `PostDecisionGreedyVfaPolicy` when you want a simple online method that:
    learns continuously while it interacts with the simulator (TD learning), and
    chooses actions by combining a one-step expected cost estimate with a learned
    post-decision value.

- Use `FqiGreedyVfaPolicy` when you want a batch/offline method that:
    first collects a dataset of transitions, then trains a regression model for
    $\hat Q(S,t,x)$ using repeated Bellman backups (FQI), and finally acts greedily
    by minimizing the learned $\hat Q$.

---

## `PostDecisionGreedyVfaPolicy`

### Concept: post-decision state and greedy action selection

The **post-decision state** is the state *after the action is applied*, but
*before* new exogenous information arrives.

For the baseline inventory model (immediate replenishment), the key post-decision
quantity is:

- inventory after ordering: $I_t^x = I_t + x_t$.

The policy chooses an action by approximately minimizing:

$$x_t \in \arg\min_{x \in \mathcal{X}}\ \mathbb{E}[C(S_t, x, W_{t+1})] + \gamma\,\hat V(I_t^x, t)$$

Where:

- $\hat V$ is a learned post-decision value approximation.
- The expectation is approximated deterministically (see below) so `act(...)`
  does not introduce additional randomness.

### Interface (PostDecisionGreedyVfaPolicy)

- `act(state, t, info=None) -> [x]`
- `train_td_value(S0, T, n_episodes, seed0, epsilon) -> None`
- `vbar_hat(S_post, t) -> float`

Key configuration parameters include:

- `feature`: either `"linear"` or `"poly2"`
- `inv_scale`: positive scaling factor applied to the inventory coordinate before
    feature construction; default `1.0` preserves the original unscaled behavior

### Features and value approximation

The value approximation is linear in features:

$$\hat V(S^{post}, t) = w^T \phi(S^{post}, t)$$

The feature map `phi` depends on:

- inventory (always)
- normalized time $\tau = t/50$ (always)
- optional regime component if the state is 2-D (`[inventory, regime]`)

Two feature modes:

- `feature="linear"`: small linear basis
- `feature="poly2"` (default): adds quadratic and interaction terms

Before features are built, the inventory coordinate is scaled as
$I_t^{x,scaled} = I_t^x / \texttt{inv\_scale}$. This is mainly useful for
`poly2`, where otherwise the quadratic term can become numerically large when
inventory levels are in the hundreds.

For regime states, if the exogenous model is `ExogenousPoissonRegime`, the regime
feature is one-hot; otherwise it uses a simple polynomial in the regime value.

### Pseudocode: greedy action (`act`, PostDecisionGreedyVfaPolicy)

```text
function ACT_PostDecisionGreedyVfaPolicy(state, t, info=None):
    xs ← {0, dx, 2dx, ..., x_max}

    best_x ← xs[0]
    best_obj ← +∞

    for x in xs:
        S_post ← MAKE_POST_STATE(state, x)

        # Deterministic Monte Carlo estimate of E[cost]
        one_step ← EXPECTED_ONE_STEP_COST(state, x, t)

        obj ← one_step + gamma * VBAR_HAT(S_post, t)

        if obj < best_obj:
            best_obj ← obj
            best_x ← x

    return [int(round(best_x))]
```

### Pseudocode: deterministic cost expectation (`_expected_one_step_cost`)

This method approximates $\mathbb{E}[C(S_t, x, W_{t+1})]$ by averaging over `n`
random samples, but uses a **deterministic seed** based on `(t, inventory)` so
repeated calls are repeatable.

```text
function EXPECTED_ONE_STEP_COST(S, x, t):
    seed ← hash(expectation_seed, t, round(S[0]))
    rng ← default_rng(seed)

    X ← [int(round(x))]
    n ← expectation_samples

    if exogenous is seasonal Poisson:
        lam ← lambda_t(t)
        demands ~ Poisson(lam)  (n samples)
        return mean_i cost_func(S, X, [demands[i]], t)

    else if exogenous is regime-switching Poisson:
        r_t ← current regime from state
        r_next ~ Categorical(P[r_t])  (n samples)
        d_i ~ Poisson(lam_by_regime[r_next[i]])
        W_i ← [d_i, r_next[i]]
        return mean_i cost_func(S, X, W_i, t)

    else:
        W_i ← exogenous.sample(S, X, t, rng)
        return mean_i cost_func(S, X, W_i, t)
```

### Learning: TD(0) on post-decision value (`train_td_value`)

The code performs TD(0) updates on the post-decision value function.

For each step, it uses:

- current post-decision value $\hat V(S_t^x, t)$
- one-step cost $C_t$
- next post-decision value $\hat V(S_{t+1}^{x'}, t+1)$ where $x'$ is the greedy
  action at the next state

The TD error is:

$$\delta_t = C_t + \gamma \hat V(S_{t+1}^{x'}, t+1) - \hat V(S_t^x, t)$$

and the linear weight update is:

$$w \leftarrow w + \alpha\,\delta_t\,\phi(S_t^x,t)$$

Pseudocode:

```text
function TRAIN_TD_VALUE(S0, T, n_episodes, seed0, epsilon):
    episode_seeds ← RNG(seed0).integers(...)  (n_episodes)

    for ep_seed in episode_seeds:
        step_seeds ← model.sample_crn_step_seeds(ep_seed, T)
        S ← S0

        for t in 0..T-1:
            # epsilon-greedy exploration over discrete actions
            if Uniform(0,1) < epsilon:
                x ← random choice from {0, dx, ..., x_max}
            else:
                x ← ACT(S, t)[0]

            X ← [int(round(x))]
            S_post ← MAKE_POST_STATE(S, x)
            phi ← PHI_POST(S_post, t)
            v_curr ← wᵀ phi

            rng_t ← default_rng(step_seeds[t])
            W ← exogenous.sample(S, X, t, rng_t)
            C ← cost_func(S, X, W, t)
            S_next ← transition_func(S, X, W, t)

            if t is terminal (t==T-1):
                v_next ← 0
            else:
                x_next ← ACT(S_next, t+1)[0]
                S_post_next ← MAKE_POST_STATE(S_next, x_next)
                v_next ← VBAR_HAT(S_post_next, t+1)

            delta ← C + gamma*v_next - v_curr
            w ← w + alpha * delta * phi

            S ← S_next
```

### Notes (PostDecisionGreedyVfaPolicy implementation details)

- `act(...)` is deterministic given `(state, t)` because the expectation uses a
  deterministic seed.
- The post-decision state keeps the regime component (if present) unchanged.
- If `w` is not initialized, the first call to `vbar_hat` allocates zeros.
- `inv_scale=1.0` preserves the pre-scaling behavior; larger values can improve
    numerical stability for quadratic features without changing the policy API.

---

## `FqiGreedyVfaPolicy`

### Concept: fitted Q iteration (batch / offline)

Fitted Q iteration approximates the **Q-function**:

$$Q(S_t, x_t) = \mathbb{E}[C_t + \gamma \min_{x'} Q(S_{t+1}, x')]$$

Training is done in batch mode:

1. Collect a dataset of transitions $(S_t, t, x_t, C_t, S_{t+1}, t+1)$.
1. Repeatedly build Bellman targets using the current $\hat Q$.
1. Fit a regression model to map features $\phi(S,t,x)$ to targets $y$.

After training, the online policy is greedy:

$$x_t \in \arg\min_{x \in \mathcal{X}} \hat Q(S_t, t, x)$$

### Interface (FqiGreedyVfaPolicy)

- `act(state, t, info=None) -> [x]`
- `train_fqi(S0, T, n_episodes, seed0, n_iterations, behavior, eps_behavior) -> None`

Key configuration parameters include:

- `feature`: either `"linear"` or `"poly2"`
- `inv_scale`: positive scaling factor applied to the inventory coordinate before
    feature construction; default `1.0` preserves the original unscaled behavior
- `a_scale`: positive scaling factor applied to the action coordinate before
    feature construction; default `1.0` preserves the original unscaled behavior

### Features and model choices

The policy featurizes state-action-time into a vector:

$$\phi(S,t,x)$$

with linear or quadratic (`poly2`) terms, plus optional regime features.

Before features are built, the inventory and action coordinates are scaled as
$I_t^{scaled} = I_t / \texttt{inv\_scale}$ and $x_t^{scaled} = x_t / \texttt{a\_scale}$.
This is especially useful for `poly2`, where squared and interaction terms can
otherwise become numerically large when inventory and order quantities are in the
hundreds.

It supports different regressors via `mode`:

- `ridge` (with a NumPy fallback if sklearn is missing)
- `elastic` (ElasticNet with scaling)
- `tree` (hist gradient boosting)
- `gbrt` (classic gradient boosting)
- `mlp` (MLP regressor with scaling)

For `elastic` and `mlp`, the implementation also applies `StandardScaler` inside
the sklearn pipeline. The explicit `inv_scale` and `a_scale` parameters are still
useful for keeping the raw feature map numerically well behaved across modes,
especially for `ridge`, `tree`, `gbrt`, and `poly2` features.

### Pseudocode: greedy action (`act`, FqiGreedyVfaPolicy)

```text
function ACT_FqiGreedyVfaPolicy(state, t, info=None):
    xs ← {0, dx, ..., x_max}

    if q_model is None:
        # Predict 0 for all actions ⇒ choose the first action (typically x=0)
        return [0]

    q_values[x] ← q_model.predict( phi(state, t, x) ) for x in xs
    best_x ← argmin_x q_values[x]

    return [int(round(best_x))]
```

### Pseudocode: dataset collection + fitted Q iteration (`train_fqi`)

The training method first simulates episodes to collect transitions.
Exogenous sampling is made reproducible by using CRN step seeds per episode.

```text
function TRAIN_FQI(S0, T, n_episodes, seed0, n_iterations, behavior, eps_behavior):
    episode_seeds ← RNG(seed0).integers(...)  (n_episodes)

    D ← empty list of transitions
    xs ← {0, dx, ..., x_max}

    # 1) Collect batch data
    for ep_seed in episode_seeds:
        step_seeds ← model.sample_crn_step_seeds(ep_seed, T)
        S ← S0

        for t in 0..T-1:
            if behavior == "random":
                x ← random choice from xs
            else if behavior == "egreedy":
                with prob eps_behavior: x ← random choice from xs
                otherwise:             x ← ACT(S, t)[0]

            X ← [int(round(x))]
            rng_t ← default_rng(step_seeds[t])
            W ← exogenous.sample(S, X, t, rng_t)
            C ← cost_func(S, X, W, t)
            S_next ← transition_func(S, X, W, t)

            append (S, t, x, C, S_next, t+1) to D
            S ← S_next

    # Precompute features for all (S,t,x) in D
    Phi ← stack_i phi(S_i, t_i, x_i)
    c   ← vector of costs C_i

    # 2) FQI iterations
    for k in 1..n_iterations:
        for each transition i in D:
            if t_i is terminal (t_i >= T-1):
                y_i ← c_i
            else:
                y_i ← c_i + gamma * min_{x' in xs} Q_hat(S_next_i, t_next_i, x')

        fit regressor q_model on (Phi, y)

    store q_model
```

### Notes (FqiGreedyVfaPolicy implementation details)

- If `q_model` is unset, predictions default to zero, so the greedy action is
  the first element of the action grid (usually `x=0`).
- The training behavior policy can be purely random or epsilon-greedy.
- The target uses `min_{x'}` because we are minimizing cost.

---

## Small naming note (aliases)

The module defines backwards-compatible names used in notebooks:

- `PostDecision_Greedy_VFA` is an alias for `PostDecisionGreedyVfaPolicy`.
- `FQI_Greedy_VFA` is an alias for `FqiGreedyVfaPolicy`.
