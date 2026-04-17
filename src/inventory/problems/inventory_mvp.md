# Inventory MVP problem definition (cost + transition)

This note explains the functions implemented in
`inventory.problems.inventory_mvp` (see [inventory_mvp.py](inventory_mvp.py)).

These functions define a complete inventory “problem” for the simulator
`DynamicSystemMVP`:

- an objective (one-step **cost**) and
- system dynamics (state **transition**)

They are intentionally minimal and meant to be readable first, fast second.

## Common conventions

- **State** `S_t` is a 1-D vector.
  - `state[0]` is the start-of-period on-hand inventory.
  - Some experiments append observable regime components: `state[1]`, `state[2]`, ...
- **Action** `X_t` is a 1-D vector with one component:
  - `action[0]` is the order quantity arriving immediately.
- **Exogenous information** `W_{t+1}` is a 1-D vector.
  - `exog[0]` is demand revealed during the period.
  - Regime demand models may also include next regimes in `exog[1:]`.

Cost minimization:

- The simulator and policies treat “lower cost is better”.
- These cost functions implement: purchasing + holding + penalty − revenue.

---

## `inventory_cost`

### `inventory_cost` concept

This is the baseline single-period cost for a **lost-sales** inventory model.

- You start with inventory `inv`.
- You order `order` units that arrive immediately.
- Demand `demand` occurs.
- You sell what you can and lose the rest (no backlogging).

The cost is:

$$\text{cost} = c\,x + h\,I_{end} + b\,\text{lost} - p\,\text{sales}$$

where

- $x$ is the order quantity
- $I_{end}$ is the end inventory after meeting demand (clipped at 0)
- `lost` is unmet demand
- `sales` is fulfilled demand

### `inventory_cost` interface

```text
inventory_cost(state, action, exog, t, *, p, c, h, b) -> float
```

Inputs:

- `state[0]`: inventory on hand
- `action[0]`: order quantity
- `exog[0]`: demand

Output:

- a scalar cost (float)

### `inventory_cost` pseudocode

```text
function COST_inventory_cost(state, action, exog, t, p, c, h, b):
    inv ← float(state[0])
    order ← float(action[0])
    demand ← float(exog[0])

    on_hand ← inv + order
    sales ← min(on_hand, demand)
    lost ← max(0, demand - on_hand)
    inv_end ← max(0, on_hand - demand)

    revenue ← p * sales
    purchase_cost ← c * order
    holding_cost ← h * inv_end
    stockout_penalty ← b * lost

    return purchase_cost + holding_cost + stockout_penalty - revenue
```

### `inventory_cost` notes (implementation details)

- The code explicitly clips end inventory at 0 (`max(0, on_hand - demand)`), which
  makes this a **lost-sales** model.
- The time index `t` is accepted for API consistency but is not used.

### `inventory_cost` worked example (numbers)

Let

- `inv = 10`
- `order = 5`
- `demand = 20`

Then

- `on_hand = inv + order = 15`
- `sales = min(15, 20) = 15`
- `lost = max(0, 20 - 15) = 5`
- `inv_end = max(0, 15 - 20) = 0`

Using the default parameters `p=2.0`, `c=0.5`, `h=0.03`, `b=1.0`:

- `revenue = p * sales = 2.0 * 15 = 30.0`
- `purchase_cost = c * order = 0.5 * 5 = 2.5`
- `holding_cost = h * inv_end = 0.03 * 0 = 0.0`
- `stockout_penalty = b * lost = 1.0 * 5 = 5.0`

So the one-step cost is

$$\text{cost} = 2.5 + 0.0 + 5.0 - 30.0 = -22.5$$

This negative number is fine: it just means revenue dominates costs in this
period. Policies still aim to **minimize** the *sum of costs* over time.

---

## `inventory_cost_extended`

### `inventory_cost_extended` concept

This adds a fixed setup cost $K$ whenever you order a positive quantity:

$$\text{cost} = c\,x + K\,\mathbf{1}\{x>0\} + h\,I_{end} + b\,\text{lost} - p\,\text{sales}$$

This creates an incentive to order less frequently and in larger batches.

### `inventory_cost_extended` interface

```text
inventory_cost_extended(state, action, exog, t, *, p, c, h, b, K) -> float
```

### `inventory_cost_extended` pseudocode

```text
function COST_inventory_cost_extended(state, action, exog, t, p, c, h, b, K):
    inv ← float(state[0])
    order ← float(action[0])
    demand ← float(exog[0])

    on_hand ← inv + order
    sales ← min(on_hand, demand)
    lost ← max(0, demand - on_hand)
    inv_end ← max(0, on_hand - demand)

    revenue ← p * sales
    purchase_cost ← c * order
    setup_cost ← K if order > 0 else 0
    holding_cost ← h * inv_end
    stockout_penalty ← b * lost

    return purchase_cost + setup_cost + holding_cost + stockout_penalty - revenue
```

### `inventory_cost_extended` notes

- The only difference from `inventory_cost` is the conditional setup cost.
- In the default parameters, `b` is larger than in `inventory_cost` (stockouts
  are punished more heavily).

---

## `inventory_transition`

### `inventory_transition` concept

This transition defines the next state after ordering and demand.

Core inventory update (always):

$$I_{t+1} = \max(0,\; I_t + x_t - D_{t+1})$$

Regime propagation (optional):

- If the state has extra components (regimes), the transition tries to copy the
  “next regime values” from the exogenous vector.
- If exogenous does not include them, the regimes are carried forward from the
  state.

### `inventory_transition` interface

```text
inventory_transition(state, action, exog, t) -> next_state
```

### `inventory_transition` pseudocode

```text
function TRANSITION_inventory_transition(state, action, exog, t):
    inv ← float(state[0])
    order ← float(action[0])
    demand ← float(exog[0])

    inv_next ← max(0, inv + order - demand)

    d_s ← len(state)
    if d_s == 1:
        return [inv_next]

    next ← [inv_next]
    for i = 1..d_s-1:
        if i < len(exog):
            next.append(exog[i])          # use regime_next from W
        else:
            next.append(state[i])         # carry regime forward

    return next
```

### `inventory_transition` notes

- This is what makes the same transition function work for:
  - no-regime systems (`S=[inventory]`, `W=[demand]`)
  - single-regime systems (`S=[inventory, regime]`, `W=[demand, regime_next]`)
  - multi-regime systems (`S=[inventory, r1, r2, ...]`, `W=[demand, r1_next, r2_next, ...]`)

---

## `inventory_transition_capped_1d`

### `inventory_transition_capped_1d` concept

Same as the baseline transition, but additionally enforces an inventory capacity
limit `s_max`:

$$I_{t+1} = \min\big(s_{max},\; \max(0, I_t + x_t - D_{t+1})\big)$$

This is useful when your environment has a hard storage constraint.

### `inventory_transition_capped_1d` interface

```text
inventory_transition_capped_1d(state, action, exog, t, *, s_max) -> next_state
```

### `inventory_transition_capped_1d` pseudocode

```text
function TRANSITION_inventory_transition_capped_1d(state, action, exog, t, s_max):
    inv ← float(state[0])
    order ← float(action[0])
    demand ← float(exog[0])

    inv_next ← inv + order - demand
    inv_next ← max(0, inv_next)
    inv_next ← min(s_max, inv_next)

    return [inv_next]
```

### `inventory_transition_capped_1d` notes

- This function is only for the 1D state case (`S=[inventory]`).
- For regime states, the module provides `inventory_transition_regime_capped`,
  which caps inventory and also propagates regimes.

---

## Small note on the factories

The same module contains two helper constructors:

- `make_inventory_mvp_system(...)`
- `make_inventory_multi_regime_system(...)`

They build a `DynamicSystemMVP` with the correct `d_s`, `d_w`, and with the
appropriate transition (capped vs uncapped) and cost (baseline vs extended).

If you want, I can expand this note with pseudocode for the factories too, but I
kept this file focused on the four functions you listed.
