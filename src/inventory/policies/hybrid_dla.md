# Composable Hybrid DLA policies

This note explains the new composable hybrid DLA policies implemented in
`inventory.policies.hybrid_dla` (see [hybrid_dla.py](hybrid_dla.py)):

- `HybridDlaDeterministicComposablePolicy`
- `HybridDlaRolloutComposablePolicy`

The purpose of this module is different from the legacy [hybrids.py](hybrids.py)
module:

- `hybrids.py` contains lecture-oriented examples of three specific hybrid motifs
- `hybrid_dla.py` provides a more composable experimental layer built around
  **roles** such as forecast provider, terminal value provider, rollout policy,
  and candidate policy

Throughout we assume a cost-minimization framing.

---

## Deterministic lookahead vs rollout-based lookahead

The most important conceptual split is:

- **Deterministic lookahead**:
  replace future uncertainty by one surrogate future path, then optimize against
  that path
- **Rollout-based lookahead**:
  keep uncertainty explicit by sampling many possible futures, then compare
  candidate first actions by average simulated cost

So the mental models are:

- deterministic lookahead:
  `Given this forecasted future, what should I do now?`
- rollout-based lookahead:
  `Across many sampled futures, which current action performs best on average?`

This is why the module contains **two** public classes rather than one giant
switch-based class.

The practical comparison is:

| Aspect | `HybridDlaDeterministicComposablePolicy` | `HybridDlaRolloutComposablePolicy` |
| --- | --- | --- |
| Main question | “Given this forecasted future, what should I do now?” | “Across many sampled futures, which first action performs best on average?” |
| Future treatment | Replace uncertainty by one surrogate path | Keep uncertainty explicit through sampled futures |
| Planning mechanism | Deterministic backward DP | Root-action rollout evaluation by Monte Carlo simulation |
| Core mechanism | Deterministic backward DP on an inventory/action grid | Root-action rollout comparison by Monte Carlo |
| Explicitly optimized decisions | Whole horizon inside the deterministic recursion, then execute only the first action | Only the first action is optimized explicitly; later actions come from the rollout continuation policy |
| Main plug-in roles | `forecast_provider`, `terminal_value_provider`, optional `candidate_policy` | `rollout_policy`, optional `candidate_policy` |
| Most similar older class | `DlaMpcMilpPolicy` | `DlaMctsUctPolicy` |
| Typical teaching interpretation | Simplified planning over one forecasted future | Sampled-future policy improvement over candidate first actions |

---

## Common conventions

### State and action

- The state `state` is a 1-D NumPy vector.
- Inventory on hand is `state[0]`.
- The action is a length-1 vector `action = [x]` with `x >= 0`.

### Discrete action grid

Both classes use a batched action grid:

$$\mathcal{X} = \{0, dx, 2dx, \dots, x_{max}\}$$

### Role-oriented composition

Instead of plugging in “a CFA” or “a VFA” directly, the new module composes
objects according to the role they play inside lookahead:

- `forecast_provider`
- `terminal_value_provider`
- `rollout_policy`
- `candidate_policy`

This is more precise than family labels, because a PFA, CFA, or VFA object may
fit different roles depending on how it is used.

### Roles in the composable hybrid DLA classes

A useful way to read the two hybrid classes is by the role each plug-in object
plays:

| Role | `HybridDlaDeterministicComposablePolicy` | `HybridDlaRolloutComposablePolicy` |
| --- | --- | --- |
| `forecast_provider` | **Core input.** Supplies the forecasted demand path that the internal deterministic DP optimizes against. | **Not used directly.** Forecast information can only enter indirectly through the chosen `rollout_policy` or `candidate_policy`. |
| `candidate_policy` | **Optional root proposal policy.** Suggests which current actions should be checked at the root; it does not plan the whole horizon. | **Optional root proposal policy.** Suggests which current actions should be tested in rollout comparison. |
| `rollout_policy` | Not used. This class does not simulate stochastic future tails. | **Core continuation policy.** Controls the simulated tail after the first candidate action in each rollout. |

So the main distinction is:

- In `HybridDlaDeterministicComposablePolicy`, the future is driven by the
  **forecast provider**.
- In `HybridDlaRolloutComposablePolicy`, the future is driven by the
  **rollout policy**.
- In both classes, `candidate_policy` only helps decide **which current/root
  actions are worth checking first**.

### Where to attach a forecaster

There is **no universal `forecaster=` argument** across all policy classes in
the repository. Forecast information is only consumed by classes that were
explicitly designed to use it.

So the practical rule is:

- if a class exposes `forecast_provider=...`, attach the forecasting object there
- if a class exposes `forecaster=...`, attach the forecasting object there
- if a class exposes neither, that class will not use the forecaster directly

For the new composable hybrid DLA module this means:

- `HybridDlaDeterministicComposablePolicy` consumes forecast information through
  its `forecast_provider`
- `HybridDlaRolloutComposablePolicy` has **no direct forecast slot**; it only
  knows about `rollout_policy` and `candidate_policy`

So if you want forecast information to matter inside
`HybridDlaRolloutComposablePolicy`, you must choose an inner policy that itself
uses a forecaster. Typical examples are:

- `OrderMilpCfaPolicy(forecaster=...)`
- `DlaMpcMilpPolicy(forecaster=...)`

By contrast, simple policies such as `OrderUpToPolicy` do not expose a
forecaster slot and therefore cannot consume one directly.

The practical lookup table is:

| Policy family / class | Direct forecast slot? | Where to inject the forecaster | Typical example |
| --- | --- | --- | --- |
| `HybridDlaDeterministicComposablePolicy` | Yes | `forecast_provider=...` on the hybrid itself | `forecast_provider=forecaster_ml_tree` |
| `HybridDlaRolloutComposablePolicy` | No | inside `rollout_policy` and/or `candidate_policy` | `rollout_policy=OrderMilpCfaPolicy(forecaster=forecaster_ml_tree, ...)` |
| `OrderMilpCfaPolicy` | Yes | `forecaster=...` on the CFA policy | `OrderMilpCfaPolicy(forecaster=forecaster_ml_tree, ...)` |
| `DlaMpcMilpPolicy` | Yes | `forecaster=...` on the MPC/DLA policy | `DlaMpcMilpPolicy(forecaster=forecaster_ml_tree, ...)` |
| Simple baselines / PFA-style heuristics | Usually no | not applicable unless wrapped inside another policy that uses forecasts | `OrderUpToPolicy` has no forecaster slot |
| Current VFA policies | Usually no direct forecaster slot | not injected as a forecaster; they act through their own learned value / Q features | `FqiGreedyVfaPolicy`, `PostDecisionFittedGreedyVfaPolicy` |

---

## Terminal value providers

The deterministic-lookahead class can use a terminal continuation value through
the protocol:

```text
value_terminal(state, t) -> float
```

Convention:

- larger terminal value is better
- the deterministic DP minimizes cost, so terminal value is subtracted as
  negative terminal cost

The module currently provides:

- `ZeroTerminalValueProvider`
- `InventoryLinearTerminalValue`
- `LegacyTerminalVfaAdapter`

### `InventoryLinearTerminalValue`

This is the simple built-in example:

$$\bar V(state, t) = \theta_0 + \theta_1 \cdot inventory$$

using only `state[0]`.

### `LegacyTerminalVfaAdapter`

This adapter wraps the older interface from `hybrids.py`:

```text
terminal_vfa.V(inv, t)
```

so older terminal VFA objects can still be reused inside the new module.

---

## `HybridDlaDeterministicComposablePolicy`

### Concept

This class performs deterministic DLA on a buffered forecast path and allows
three composable ingredients:

- a `forecast_provider`
- an optional `terminal_value_provider`
- an optional `candidate_policy` to narrow the **root** action set

It is the natural class for:

- CFA-style forecast-driven lookahead
- deterministic lookahead with a terminal VFA boundary
- future regime-aware deterministic hybrids

### Forecast interface

Required:

- `forecast_mean_path(state, t, H, info=None) -> mu[0..H-1]`

Optional:

- `forecast_std_path(state, t, H, info=None) -> sigma[0..H-1]`

If the provider has no `forecast_std_path`, the implementation uses:

$$\sigma_k = 0$$

so the deterministic path reduces to the mean path.

### Buffered deterministic path

The planning path is:

$$\hat d_k = \max(0, \mu_k + buffer_k \cdot \sigma_k)$$

### Cost parameters

The deterministic hybrid evaluates a **built-in surrogate stage-cost formula**
inside its dynamic program. That is why its constructor asks for:

- `p`
- `c`
- `h`
- `b`
- `K`

These coefficients are used directly when the class computes deterministic
lookahead costs over the inventory/action grid.

The role of `K` is:

- pay a fixed setup cost whenever the chosen order is strictly positive

So the deterministic hybrid can now align more closely with simulator cost
functions that include a fixed setup charge.

If `s_max` is supplied, the deterministic hybrid also mirrors the capped
inventory semantics used by the capped simulator variants:

- pre-demand on-hand stock is clipped to `min(inv + order, s_max)`
- purchase cost and setup cost are still charged on the **actual order
  quantity**, not on the clipped received amount

This matches the intended “over-ordering is possible but economically harmful”
interpretation used in the capped inventory-cost helpers.

This still differs from `DlaMpcMilpPolicy`, whose surrogate planning model
represents setup decisions explicitly through MILP indicator variables. The
composable deterministic hybrid remains a simpler grid-based deterministic DP.

### Candidate policy role

If `candidate_policy` is supplied and `include_full_action_grid=False`, the
policy does **not** change the whole DP recursion. It only narrows the **first
action candidates** at the root to a neighborhood around the proposal action:

$$\{x_{base} - r\,dx, \dots, x_{base}, \dots, x_{base} + r\,dx\}$$

where `r = candidate_radius_steps`.

All interior dynamic-programming steps still use the full action grid.

This is the important conceptual point:

- the `candidate_policy` is **not** the planner for the whole horizon
- the hybrid itself is still the planner, because it runs the deterministic
  backward DP internally

So if you plug in a simple heuristic or PFA-style policy as `candidate_policy`,
that policy is only acting as a **root proposal mechanism**:

- it suggests where the root search should look
- it does **not** control the later lookahead steps

The actual horizon-level reasoning is still performed by the hybrid’s own
deterministic recursion.

### Interface

- `act(state, t, info=None) -> [x]`

### Pseudocode

```text
function ACT_HybridDlaDeterministicComposablePolicy(state, t, info=None):
    d_hat[0..H-1] ← FORECAST_PATH(state, t, H, info)
    A_root ← ROOT_CANDIDATE_ACTIONS(state, t, info)

    build inventory grid Sgrid and full action grid A

    J_next[s] ← - TERMINAL_VALUE(terminal_state_from_inventory(s), t+H)

    for k from H-1 down to 0:
        for each s in Sgrid:
            J[s] ← min over a in A of:
                stage_cost(s, a, d_hat[k]) + interp(J_next, s_next)
        J_next ← J

    choose best first action only over A_root:
        c0(inv0, a, d_hat[0]) + interp(J_next, s_next)

    return PACK_ACTION(best_a)
```

### Important interpretation

This is still **deterministic lookahead**:

- uncertainty is compressed into one surrogate path
- terminal value only enters at the horizon boundary
- the optional candidate policy only narrows the root choice

So if you plug in a VFA or PFA policy as `candidate_policy`, that policy is
being used as a **proposal mechanism**, not as the main planner.

---

## `HybridDlaRolloutComposablePolicy`

### Concept

This class performs rollout-based DLA using:

- a `rollout_policy`
- an optional `candidate_policy`

It evaluates candidate first actions by simulating many stochastic futures and
then following the rollout policy after the first step.

This is the natural class for:

- PFA-guided rollout improvement
- using an existing VFA/PFA/CFA policy as a downstream continuation policy
- experiments where uncertainty should remain explicit instead of being
  compressed into one deterministic path

### Candidate policy role

The candidate set is built either:

- from the full action grid, or
- from a local neighborhood around the `candidate_policy` action

If no separate `candidate_policy` is supplied, the class uses the
`rollout_policy` itself as the proposal policy.

### Rollout policy role

For each candidate first action:

- step 0 uses the candidate action
- steps `1..H-1` use `rollout_policy.act(...)`

This also explains where forecast information enters on the rollout side:

- `HybridDlaRolloutComposablePolicy` itself does not read a `forecast_provider`
- any forecast information must be embedded in the chosen `rollout_policy`
  and/or `candidate_policy`

So if you want to use a common forecaster in both the deterministic and rollout
hybrids, the usual pattern is:

- deterministic hybrid: `forecast_provider=<shared forecaster>`
- rollout hybrid: choose a `rollout_policy` (and optionally `candidate_policy`)
  that itself uses that same forecaster

### Why the rollout hybrid has no separate `p, c, h, b`

`HybridDlaRolloutComposablePolicy` does not build its own analytical stage-cost
expression. Instead, it simulates forward and evaluates each step through the
system’s actual cost function.

So:

- deterministic hybrid: needs explicit `p, c, h, b` because it computes
  surrogate costs internally
- rollout hybrid: does not need those coefficients because it calls
  `system.cost_func(...)` directly during rollout simulation

This also means the rollout hybrid automatically inherits whatever cost model is
already embedded in the dynamic system, including fixed setup costs when the
system cost function includes them.

### Root proposal policy vs learned continuation policy

These two roles solve different problems inside rollout-based lookahead:

- the `candidate_policy` is a **root proposal policy**
- the `rollout_policy` is a **continuation policy**

An even shorter intuition is:

- `candidate_policy`: `Which first actions should we bother testing?`
- `rollout_policy`: `If we test one of those first actions, how should the future be played out in simulation?`

So the rollout-composable hybrid has a two-stage logic:

1. build a **shortlist of current actions**
2. evaluate each shortlist action by simulating **what happens next**

This is why the two roles are separated. One policy helps the hybrid decide
**where to look** at the root, while the other policy defines **how to score
what it looks at** through simulated continuation.

The root proposal policy affects only which **current first actions** are tested.
Its purpose is search efficiency:

- `candidate_policy.act(state, t, info)` proposes a reasonable current action
- the hybrid then tests a local neighborhood around that proposal at the root

So a simple policy such as `OrderUpToPolicy` can be useful here even if it is
not the policy we want to trust for the whole future.

The continuation policy affects how the **future tail** is simulated after each
candidate first action. Its purpose is rollout quality:

- step `0` still uses the candidate action under evaluation
- later steps `1..H-1` follow `rollout_policy.act(...)`

So a learned policy such as `FqiGreedyVfaPolicy` can be valuable here because it
shapes the simulated downstream behavior used to judge each candidate first
action.

### Memory trick

- `candidate_policy` chooses the **shortlist**
- `rollout_policy` chooses the **story of what happens next**

If both are the same policy, then the same logic drives both:

- root proposal
- tail simulation

If they differ, then the hybrid is deliberately mixing roles:

- one logic to anchor the root search
- another logic to evaluate downstream continuation

In short:

- **root proposal policy**: `Which first actions should we bother testing?`
- **continuation policy**: `If we take one of those first actions, how should the simulated future be controlled afterward?`

### Simulated `info`

During rollout simulation, the class updates:

- `crn_step_seed`
- `last_demand`
- `demand_history` when maintained

This allows rollout policies that rely on the repo’s standard `Policy.act`
information conventions to be reused directly.

### Interface

- `act(state, t, info=None) -> [x]`

### Pseudocode

```text
function ACT_HybridDlaRolloutComposablePolicy(state, t, info=None):
    A_root ← ROOT_CANDIDATE_ACTIONS(state, t, info)
    rollout_seeds ← shared seeds for CRN across candidate comparisons

    for each candidate x0 in A_root:
        vals ← []
        for seed in rollout_seeds:
            simulate H-step path:
                step 0: apply x0
                steps 1..H-1: use rollout_policy.act(...)
                sample exogenous noise each step
                update rollout info with last_demand and demand_history
            vals.append(total simulated cost)

        v[x0] ← mean(vals)

    return candidate with smallest mean rollout cost
```

### Important interpretation

This is still **rollout-based lookahead**:

- uncertainty stays explicit through sampled futures
- the rollout policy is a continuation policy, not a terminal value model
- the candidate policy only shapes which first actions are tested

---

## Natural mapping from family ideas to roles

The module is designed around roles, not family labels. Typical mappings are:

- **PFA**
  - natural as `rollout_policy`
  - natural as `candidate_policy`

- **CFA**
  - natural as `forecast_provider` when a path forecast is available
  - sometimes useful as `candidate_policy` when reusing a full policy object

- **VFA**
  - natural as `rollout_policy` when reusing a full value-based policy
  - natural as `terminal_value_provider` only when wrapped by a suitable
    terminal-value adapter

This is why the composable module does not ask for “a PFA/CFA/VFA object” in a
single generic slot. The same family can play different roles.

### How to read the hybrid roles

This student-facing reading pattern is often useful in notebooks:

if both hybrids use `baseline = OrderUpToPolicy(...)` as their plug-in policy,
then they are naturally **DLA + Heuristic** hybrids.

- `HybridDlaDeterministicComposablePolicy`:
  the DLA part is the internal deterministic lookahead DP, while
  `candidate_policy=baseline` only proposes which **root actions** should be
  checked.
- `HybridDlaRolloutComposablePolicy`:
  the DLA part is the rollout-based root-action evaluation, while
  `candidate_policy=baseline` proposes root actions and
  `rollout_policy=baseline` controls the **continuation tail** in each rollout.

The same role-reading extends to other families:

| Hybrid family | `candidate_policy` role | `rollout_policy` role | Typical examples |
| --- | --- | --- | --- |
| DLA + Heuristic | Root proposal policy | Heuristic continuation for rollout DLA | `OrderUpToPolicy` |
| DLA + PFA | Root proposal policy | PFA continuation for rollout DLA | `OrderUpToPolicy`, `OrderUpToRegimeTablePFA`, `OrderUpToStateDependentPFA`, `OrderUpToBlackboxPFA` |
| DLA + VFA | Optional root proposal policy; for deterministic DLA, VFA is often more natural as a `terminal_value_provider` | VFA continuation for rollout DLA | `PostDecisionGreedyVfaPolicy`, `PostDecisionFittedGreedyVfaPolicy`, `FqiGreedyVfaPolicy` |
| DLA + CFA | Optional root proposal policy; for deterministic DLA, CFA is usually more natural through `forecast_provider` | CFA continuation for rollout DLA | `OrderMilpCfaPolicy`, `DlaMpcMilpPolicy` |

So the general pattern is:

- `candidate_policy`: `Which current actions are worth checking at the root?`
- `rollout_policy`: `After the first action, how should the simulated future be controlled?`
- `forecast_provider` in deterministic DLA: `What future demand path should the deterministic planner optimize against?`

### How to read concrete hybrids

Suppose a notebook defines:

- `baseline = OrderUpToPolicy(...)`
- `hybrid_deterministic = HybridDlaDeterministicComposablePolicy(..., candidate_policy=baseline, ...)`
- `hybrid_rollout = HybridDlaRolloutComposablePolicy(..., candidate_policy=baseline, rollout_policy=baseline, ...)`

Then both hybrids are naturally read as **DLA + Heuristic** hybrids:

- deterministic hybrid:
  - the DLA part is the internal deterministic backward DP
  - the heuristic part is `candidate_policy=baseline`, which only proposes
    which **root actions** should be checked
- rollout hybrid:
  - the DLA part is the rollout-based root-action comparison
  - the heuristic part is `candidate_policy=baseline` for root proposal and
    `rollout_policy=baseline` for continuation during simulated tail control

The same reading extends to other families:

| Hybrid reading | Deterministic-lookahead side | Rollout-lookahead side |
| --- | --- | --- |
| **DLA + Heuristic** | `candidate_policy=<heuristic>` | `candidate_policy=<heuristic>`, `rollout_policy=<heuristic>` |
| **DLA + PFA** | most naturally `candidate_policy=<PFA>` as a root proposal mechanism | `candidate_policy=<PFA>` and/or `rollout_policy=<PFA>` |
| **DLA + VFA** | often most natural as `terminal_value_provider=<VFA adapter>`; sometimes `candidate_policy=<VFA policy>` as a root proposal mechanism | most naturally `rollout_policy=<VFA policy>`; optionally `candidate_policy=<VFA policy>` |
| **DLA + CFA** | most naturally `forecast_provider=<forecast object>`; sometimes `candidate_policy=<CFA policy>` when reusing a full policy object | most naturally `rollout_policy=<CFA policy>`; optionally `candidate_policy=<CFA policy>` |

So the family label depends on **which external object is plugged into which
role**:

- `candidate_policy`: `Which current actions should the hybrid inspect at the root?`
- `rollout_policy`: `How should the simulated future be controlled after the first action?`
- `forecast_provider`: `What deterministic future path should the planner optimize against?`
- `terminal_value_provider`: `What continuation value should be assigned at the horizon boundary?`

---

## Relation to older DLA classes

The new composable hybrids are not copies of the older DLA classes, but there
is a useful resemblance map:

- `HybridDlaDeterministicComposablePolicy`
  - behaves more like `DlaMpcMilpPolicy`
  - both are deterministic-lookahead policies that build a forecast path and
    optimize against that surrogate future
  - the main difference is that `DlaMpcMilpPolicy` solves a deterministic MILP
    with setup-cost modeling, while the composable deterministic hybrid solves a
    simpler grid-based deterministic DP with optional terminal value correction

So the key distinction is:

- **deterministic lookahead** is the shared high-level idea
- **backward DP** is only the inner solver used by
  `HybridDlaDeterministicComposablePolicy`
- `DlaMpcMilpPolicy` is not DP-based in repo code; it is
  **optimization-based deterministic lookahead** via a MILP

- `HybridDlaRolloutComposablePolicy`
  - behaves more like `DlaMctsUctPolicy`
  - both keep uncertainty explicit by evaluating sampled futures
  - the main difference is that `DlaMctsUctPolicy` performs full UCT tree search,
    whereas the composable rollout hybrid performs root-action rollout
    comparison without an internal search tree

---

## What the module is for

Use `hybrid_dla.py` when you want:

- new experiments with lego-brick composition of lookahead ingredients
- a cleaner architecture than the legacy lecture-focused `hybrids.py`
- explicit separation between deterministic-lookahead hybrids and rollout-based hybrids

Use `hybrids.py` when you want:

- the older lecture-ready hybrid examples exactly as previously named and documented
