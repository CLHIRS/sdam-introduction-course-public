# Hybrid policies (mixing DLA with CFA/VFA/PFA ideas)

This note explains the hybrid policies implemented in
`inventory.policies.hybrids` (see [hybrids.py](hybrids.py)):

- `HybridDlaCfaBufferedForecastPolicy`
- `HybridDlaVfaTerminalPolicy`
- `HybridDlaRolloutBasePfaPolicy`

All three policies share a common “hybrid” strategy:

- use a **simple model-based lookahead** for a short horizon $H$ (DLA idea), but
- replace parts of the true stochastic problem with approximations:
  - **CFA-ish**: use a deterministic “buffered” demand path as a surrogate for uncertainty
  - **VFA-ish**: add a learned/hand-tuned terminal value at the horizon
  - **PFA-ish**: use a base policy as the rollout policy beyond the first step

They are designed to be easy to understand and fast enough for lecture demos.

---

## Common conventions

### State and action

- The state is a 1-D NumPy vector.
- Inventory on hand is always `state[0]`.
- The action is always a length-1 vector `action = [x]`.

### Discrete action grid

All three policies use a discrete order grid (batching):

$$\mathcal{X} = \{0, dx, 2dx, \dots, x_{max}\}$$

### Cost convention

The “fast paths” use the same per-step proxy cost components:

- ordering cost: `c * order`
- holding cost: `h * inventory_next`
- lost sales penalty: `b * lost`
- sales revenue as negative cost: `-p * sales`

So per-step cost is:

$$c x + h I_{t+1} + b\,\text{lost} - p\,\text{sales}$$

---

## Forecast path interface (used by Hybrid #1 and #2)

The first two hybrids require a **path forecaster** that can produce mean and
standard deviation paths:

- `forecast_mean_path(state, t, H, info=...) -> mu[0..H-1]`
- `forecast_std_path(state, t, H, info=...) -> sigma[0..H-1]`

The policies convert many possible forecasters into this interface using
`as_path_forecaster(...)`.

### Buffered demand path

Both policies build a deterministic “buffered” demand path:

$$\hat d_k = \max\big(0,\ \mu_k + buffer\_k \cdot \sigma_k\big)$$

This is the key approximation: it converts uncertainty into a single planning
trajectory.

---

## `HybridDlaCfaBufferedForecastPolicy`

### Concept (HybridDlaCfaBufferedForecastPolicy)

This policy does deterministic lookahead using the buffered demand path, then
computes the best first action by solving a small deterministic DP on grids.

- It is “CFA-ish” in the sense that it plans against a cost model with a single
  forecast path (not full stochastic demand).

### Interface (HybridDlaCfaBufferedForecastPolicy)

- `act(state, t, info=None) -> [x]`

### Pseudocode: action selection (`act`, HybridDlaCfaBufferedForecastPolicy)

```text
function ACT_HybridDlaCfaBufferedForecastPolicy(state, t, info=None):
    inv0 ← state[0]

    d_hat[0..H-1] ← max(0, mu + buffer_k * sigma)

    if FAST_PATH_AVAILABLE:
        x0 ← DP_VECTORIZED(inv0, d_hat)
        return PACK_ACTION(x0)

    # Generic fallback: same DP but using system.cost_func / transition_func calls
    build grids:
        Sgrid = {0, s_grid_step, ..., s_max}
        Agrid = {0, dx, ..., x_max}

    J_next[s] ← 0 for all s in Sgrid

    for k from H-1 down to 0:
        for each s in Sgrid:
            J[s] ← min over a in Agrid of:
                c0 ← cost_func([s], [a], [d_hat[k]], t+k)
                s_next ← transition_func([s], [a], [d_hat[k]], t+k)[0]
                c0 + interp(J_next, s_next)
        J_next ← J

    choose best first action a minimizing:
        c0(inv0, a, d_hat[0]) + interp(J_next, s_next)

    return PACK_ACTION(best_a)
```

### Notes (HybridDlaCfaBufferedForecastPolicy implementation details)

- The fast path triggers only when the system is the canonical inventory MVP
  (`inventory_cost` and `inventory_transition`), allowing a fully vectorized DP.
- The vectorized DP uses interpolation on the inventory grid to approximate
  value-to-go at non-grid next states.

---

## `TerminalVFA` and `TerminalVfaLinear` (used by Hybrid #2)

### Concept (TerminalVFA)

Hybrid #2 uses a terminal value approximation at the horizon boundary.

The interface is:

- `terminal_vfa.V(inv, t) -> float`

`TerminalVfaLinear` is a simple example:

$$\hat V(inv,t) = \theta_0 + \theta_1\,inv$$

---

## `HybridDlaVfaTerminalPolicy`

### Concept (HybridDlaVfaTerminalPolicy)

This is the same deterministic lookahead DP as Hybrid #1, but with a nonzero
terminal value at time $t+H$.

Instead of ending the horizon with value 0, it sets:

$$J_{H}(s) = -\hat V(s, t+H)$$

The minus sign is intentional in the implementation: the DP is minimizing cost,
while $\hat V$ is a value (higher is better), so we subtract value as negative
cost.

### Interface (HybridDlaVfaTerminalPolicy)

- `act(state, t, info=None) -> [x]`

### Pseudocode: action selection (`act`, HybridDlaVfaTerminalPolicy)

```text
function ACT_HybridDlaVfaTerminalPolicy(state, t, info=None):
    inv0 ← state[0]

    d_hat[0..H-1] ← max(0, mu + buffer_k * sigma)

    build grids:
        Sgrid = {0, s_grid_step, ..., s_max}
        Agrid = {0, dx, ..., x_max}

    # terminal condition uses terminal value approximation
    J_next[s] ← -terminal_vfa.V(s, t+H)   for s in Sgrid

    for k from H-1 down to 0:
        J[s] ← min over a:
            stage_cost(s, a, d_hat[k]) + interp(J_next, s_next)
        J_next ← J

    choose best first action a minimizing:
        stage_cost(inv0, a, d_hat[0]) + interp(J_next, s_next)

    return PACK_ACTION(best_a)
```

### Notes (HybridDlaVfaTerminalPolicy implementation details)

- The “fast path” is available under the same conditions as Hybrid #1.
- If the system is not the canonical inventory MVP, it falls back to calling
  `system.cost_func` and `system.transition_func` in loops.

---

## `HybridDlaRolloutBasePfaPolicy`

### Concept (HybridDlaRolloutBasePfaPolicy)

This policy evaluates candidate first actions by Monte Carlo rollout:

- At step 0, we try a candidate action $x_0$.
- For steps 1..H-1, we follow a **base policy** (this is the “PFA-ish” part).
- We repeat this for multiple random demand rollouts and pick the candidate with
  the lowest average total cost.

This is a standard rollout improvement idea: “try a few alternatives now, then
use a reasonable policy afterward.”

### Interface (HybridDlaRolloutBasePfaPolicy)

- `act(state, t, info=None) -> [x]`

Important tuning parameters include:

- `H`, `n_rollouts`
- `candidate_radius_steps`, `include_full_action_grid`
- `adaptive_candidate_radius`, `adaptive_radius_near`, `adaptive_radius_far`, `adaptive_gap_threshold`

`base_policy` can be either:

- a `Policy` object with an `act(...)` method, or
- a callable `base_policy(state, t) -> action`.

### Candidate action set

The candidate set can be either:

- the full grid (if `include_full_action_grid=True`), or
- a neighborhood around the base policy’s action:

$$\{x_{base} - r\,dx, \dots, x_{base}, \dots, x_{base} + r\,dx\}$$

where `r = candidate_radius_steps`.

If `adaptive_candidate_radius=True`, the policy replaces the fixed radius with
an adaptive rule:

- use `adaptive_radius_near` when the current state is close to the base-policy target,
- use `adaptive_radius_far` when the current state is far from that target,
- where “far” means the absolute gap exceeds `adaptive_gap_threshold`.

For an order-up-to style base policy, the gap is simply:

$$|target - inv|$$

This gives a more compute-efficient rollout policy:

- small local search in routine states,
- wider local search in harder states where the base policy is more likely to be suboptimal.

### CRN compatibility

This policy uses `info["crn_step_seed"]` (when present) to seed its internal
randomness:

- `SeedSequence([decision_seed, crn_step_seed, t])`

This ensures that comparing two policies under strict CRN also makes this
rollout policy’s planning randomness comparable.

### Pseudocode: action selection (`act`, HybridDlaRolloutBasePfaPolicy)

```text
function ACT_HybridDlaRolloutBasePfaPolicy(state, t, info=None):
    cand ← CANDIDATE_ACTIONS(state, t)

    rng ← default_rng(SeedSequence([decision_seed, info.crn_step_seed, t]))

    if FAST_PATH_AVAILABLE:
        # Vectorized rollouts for seasonal Poisson demand (lambda_t exists)
        D[rollout, k] ← Poisson(lambda_t(t+k)) for k=0..H-1

        for x0 in cand:
            inv_vec ← inv0 repeated n_rollouts times
            total_vec ← zeros(n_rollouts)

            for k in 0..H-1:
                if k == 0:
                    order_vec ← x0 repeated
                else:
                    order_vec ← BASE_ORDER(inv_vec)   # vectorized when possible

                demand_vec ← D[:, k]

                on_hand ← inv_vec + order_vec
                sales ← min(on_hand, demand_vec)
                lost  ← max(0, demand_vec - on_hand)
                inv_vec ← max(0, on_hand - demand_vec)

                total_vec += c*order_vec + h*inv_vec + b*lost - p*sales

            v(x0) ← mean(total_vec)

        return argmin_x0 v(x0)

    else:
        # Generic fallback: reuse rollout seeds across candidates (CRN across candidates)
        rollout_seeds ← rng.integers(..., size=n_rollouts)

        for x0 in cand:
            costs ← []
            for seed in rollout_seeds:
                rng_r ← default_rng(seed)
                costs.append( ROLLOUT_COST_GENERIC(state, t, x0, rng_r) )

            v(x0) ← mean(costs)

        return argmin_x0 v(x0)
```

### Notes (HybridDlaRolloutBasePfaPolicy implementation details)

- The fast path tries to vectorize the base policy too, but only if it looks like
  an order-up-to style policy (has `target_level` or `S_bar`). Otherwise it calls
  the base policy per rollout.
- The generic fallback enforces fairness across candidate actions by reusing the
  same rollout seeds for all candidates.
- The adaptive search option changes only the root candidate set. The rollout
    evaluation itself is unchanged.

---

## Small naming note (aliases)

The module defines notebook-compatible aliases:

- `Hybrid_DLA_CFA_BufferedForecast` → `HybridDlaCfaBufferedForecastPolicy`
- `Hybrid_DLA_VFA_Terminal` → `HybridDlaVfaTerminalPolicy`
- `Hybrid_DLA_Rollout_BasePFA` → `HybridDlaRolloutBasePfaPolicy`
- `TerminalVFA_Linear` → `TerminalVfaLinear`
