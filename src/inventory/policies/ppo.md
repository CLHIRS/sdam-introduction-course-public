# PPO policy (actor-critic with feasibility masking)

This note explains the PPO-based policy implemented in
`inventory.policies.ppo` (see [ppo.py](ppo.py)):

- `HybridPpoPolicy`

The goal is to keep the implementation approachable for students while being
faithful to what the code actually does.

---

## Big picture: what PPO is doing here

`HybridPpoPolicy` is a **policy function approximation (PFA)** trained with an
**actor-critic** method:

- The **actor** is the policy $\pi_\theta(x\mid s)$.
- The **critic** is a value function $V_\theta(s)$ used to reduce variance.

In this codebase we are minimizing cost. PPO is typically written in
reward-maximization form, so the implementation uses:

- `reward = -cost`

So “maximize reward” is equivalent to “minimize cost”.

---

## Common conventions

### Discrete action grid (batching)

The policy uses a discrete order grid:

$$\mathcal{X} = \{0, dx, 2dx, \dots, x_{max}\}$$

This is enforced by construction:

- `action_grid = np.arange(0, x_max+1, dx)`

The returned action is always a length-1 NumPy vector: `action = [x]`.

### Feasibility: inventory capacity masking

The policy enforces a simple feasibility rule:

$$\text{inventory} + x \le S_{max}$$

Instead of projecting an infeasible action after sampling, it uses **masking**:

- it sets logits of infeasible actions to a very negative number (`-1e9`)
- the categorical distribution then assigns essentially zero probability to those actions

If no action is feasible, it forces `x=0` to be feasible.

### Normalization of state inputs

The neural network does not see raw states directly.

- A running mean/std normalizer (`_RunningNorm`) tracks state statistics.
- States are normalized before being fed into the network.

---

## `HybridPpoPolicy`

### Interface

- Constructor:
  - `HybridPpoPolicy(d_s, s_max, x_max, dx, hidden, device, hparams, seed, deterministic_eval)`
- Acting:
  - `act(state, t, info=None) -> [x]`
- Training:
  - `train_ppo(system, S0, T, n_episodes, seed0=..., verbose=True) -> losses_dict`
- Save/restore (evaluation gate):
  - `get_params() -> dict`
  - `set_params(params) -> None`

### Network structure (conceptual)

The actor and critic share a trunk network:

```text
s_norm -> shared MLP ->
    policy_head: logits over discrete actions
    value_head:  scalar value V(s)
```

### Acting modes

The policy supports two broad modes:

1) **Stochastic sampling** (default)

- Needed during PPO training.
- Samples an action index from `Categorical(logits=masked_logits)`.

2) **Deterministic evaluation**

Enabled if either:

- the policy was created with `deterministic_eval=True`, or
- `info["deterministic"] == True`

Deterministic modes (selected by `info["det_mode"]`):

- `"mean"` (default): compute mean action under the distribution and round to the grid.
- `"argmax"`: choose the highest-probability action (optionally risk-penalized).

### Strict-CRN compatibility

If `info["crn_step_seed"]` is present and the policy is sampling stochastically,
then the sampling is made reproducible under strict CRN by deriving a seed from:

- `[policy_seed, crn_step_seed, t]`

This ensures the policy’s own sampling randomness is comparable across policies
when using CRN evaluation.

---

## Pseudocode: feasibility-masked logits

```text
function MASKED_LOGITS(logits, inv):
    feasible[a] ← (inv + action_grid[a] ≤ S_max)
    if no feasible action:
        feasible[action=0] ← True

    for a where feasible[a] is False:
        logits[a] ← -1e9

    return logits
```

---

## Pseudocode: `act` (HybridPpoPolicy)

```text
function ACT_HybridPpoPolicy(state, t, info=None):
    info ← info or {}

    deterministic ← deterministic_eval OR info["deterministic"]
    det_mode ← info.get("det_mode", "mean")
    risk_alpha ← info.get("risk_alpha", 0.0)

    inv ← state[0]

    # forward pass
    s_norm ← normalize(state)
    logits, value ← net(s_norm)
    logits ← MASKED_LOGITS(logits, inv)

    dist ← Categorical(logits)

    if deterministic is False:
        if info contains "crn_step_seed":
            # sample using a deterministic seed derived from (seed, crn_step_seed, t)
            seed' ← SeedSequence([seed, crn_step_seed, t]).generate_state(1)[0]
            a_idx ← sample from softmax(logits) using numpy RNG(seed')
        else:
            a_idx ← dist.sample()

        x ← action_grid[a_idx]
        return [x]

    # -------------------------
    # Deterministic evaluation
    # -------------------------
    probs ← dist.probs

    if det_mode == "argmax":
        if risk_alpha > 0:
            score[a] ← log(probs[a]) - risk_alpha * (action_grid[a] / S_max)
            a_idx ← argmax score[a]
        else:
            a_idx ← argmax probs[a]

        return [action_grid[a_idx]]

    # det_mode == "mean" (default)
    x_mean ← Σ_a probs[a] * action_grid[a]

    # round to dx-grid and enforce feasibility
    x ← round_to_multiple(x_mean, dx)
    x ← clip(x, 0, x_max)

    x_cap ← max(0, S_max - inv)
    x ← min(x, x_cap)

    # snap down to keep feasibility
    x ← dx * floor(x / dx)

    return [x]
```

---

## PPO training in this file (what `train_ppo` does)

### Data collection

The training method runs rollouts using `system.simulate(self, ...)`.

- It collects trajectories of length `T` for `n_episodes` episodes.
- It stores per-step:
  - state $s_t$
  - chosen discrete action index $a_t$ (matching `action_grid`)
  - reward $r_t = -cost_t$
  - log-prob under the current policy $\log\pi_\theta(a_t\mid s_t)$
  - value estimate $V_\theta(s_t)$

### Advantage estimation (GAE)

It computes generalized advantage estimates (GAE) per episode:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$$A_t = \delta_t + \gamma\lambda A_{t+1}$$

with $V(s_T)=0$ at the end of each episode chunk.

Returns are:

$$R_t = A_t + V(s_t)$$

Advantages are normalized before optimization.

### PPO update (clipped objective)

For each minibatch it computes:

- probability ratio: $\rho_t = \exp(\log\pi_\theta - \log\pi_{\text{old}})$
- clipped surrogate objective
- value loss (squared error to returns)
- entropy bonus

and minimizes:

$$\mathcal{L} = -\mathbb{E}[\min(\rho A, \text{clip}(\rho) A)]
+ c_v\,\mathbb{E}[(R - V)^2]
- c_e\,\mathbb{E}[\mathsf{H}(\pi)]$$

Important detail: during training it applies the **same feasibility mask** to
logits before computing log-probs/entropy.

---

## Evaluation gate helper (`train_ppo_with_eval_gate`)

This file also provides an AlphaZero-style “accept/reject” loop:

- Train PPO for a chunk of episodes.
- Evaluate `Baseline` vs `PPO` under strict CRN using
  `system.evaluate_policies_crn_mc`.
- If PPO improves the best mean cost so far, accept the parameters.
- Otherwise revert PPO back to the last accepted parameters.

This is helpful in teaching contexts because it prevents training from
accidentally “getting worse” due to optimizer noise.

---

## Small naming note (alias)

For notebook compatibility (Lecture 16 naming), the module also defines:

- `Hybrid_PPO` as an alias for `HybridPpoPolicy`.
