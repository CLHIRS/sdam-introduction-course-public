# Baseline inventory policies (order-up-to)

This note explains the simple baseline policies implemented in
`inventory.policies.baselines` (see [baselines.py](baselines.py)).

These policies are meant to be easy to understand and fast to run. They are also
useful as **benchmarks** for more advanced policies (DP, VFA/PFA, RL, etc.).

## Common conventions

- **State**: inventory on hand is stored in `state[0]`.
- **Action**: returned as a length-1 vector, where `action[0]` is the order quantity.
- The computed order quantity is ultimately rounded to an integer and returned as
  a float array: `np.array([x_int], dtype=float)`.

### Order-up-to idea (the concept)

An *order-up-to* (a.k.a. *base-stock*) policy tries to keep inventory near a
chosen **target level** $S$.

- If current inventory is below $S$, order the difference.
- If current inventory is already at/above $S$, order nothing.

In a one-period setting (ignoring constraints), the decision rule is:

$$x = \max(0,\, S - \text{inventory})$$

where $x$ is the order quantity.

## `OrderUpToPolicy`

This is the “plain” order-up-to policy, with two optional practical constraints:

- `x_max`: cap the order quantity (e.g., ordering limit).
- `dx`: restrict orders to multiples of `dx` (batching / case packs).

### `OrderUpToPolicy` pseudocode

Inputs:

- current inventory `inv = state[0]`
- target level `S = target_level`
- optional `x_max`
- optional `dx`

Algorithm:

```text
function ACT_OrderUpToPolicy(state, t, info=None):
    inv ← float(state[0])

    # base-stock rule
    x ← max(0, S - inv)

    # optional order cap
    if x_max is not None:
        x ← min(x, x_max)

    # optional batching (multiples of dx)
    if dx is not None and dx > 1:
        x ← dx * round(x / dx)
        x ← max(0, x)
        if x_max is not None:
            x ← min(x, x_max)

    # return as a length-1 vector (integer-valued)
    x_int ← int(round(x))
    return [x_int]
```

### `OrderUpToPolicy` notes (implementation details)

- Rounding happens twice if `dx` is used:
  - first to get to the nearest multiple of `dx` via `dx * round(x / dx)`
  - then to an integer with `int(round(x))`
- This class does **not** enforce a storage capacity constraint like
  `inv + x ≤ s_max`. If you need that feasibility condition, use
  `OrderUpToCapacityPolicy`.

## `OrderUpToCapacityPolicy`

This policy implements the same base-stock idea, but explicitly enforces
feasibility with respect to a maximum inventory level `s_max`.

It enforces three constraints:

1) **Batching**: $x$ must be a multiple of `dx` (when `dx > 1`).

2) **Order cap**: $x \le x_{max}$.

3) **Storage capacity**: inventory after ordering cannot exceed `s_max`:

$$inv + x \le s_{max} \quad\Rightarrow\quad x \le \max(0,\, s_{max} - inv)$$

### `OrderUpToCapacityPolicy` pseudocode

Inputs:

- current inventory `inv = state[0]`
- target level `S = target_level`
- order cap `x_max`
- batch size `dx`
- max inventory `s_max`

Algorithm:

```text
function ACT_OrderUpToCapacityPolicy(state, t, info=None):
    inv ← float(state[0])

    # desired order from base-stock rule
    x ← max(0, S - inv)

    # enforce feasibility constraints
    x ← min(x, x_max)
    x ← min(x, max(0, s_max - inv))

    # if batching is used, round to a multiple of dx,
    # then re-apply feasibility (rounding can violate constraints)
    if dx > 1:
        x ← dx * round(x / dx)
        x ← max(0, x)
        x ← min(x, x_max)
        x ← min(x, max(0, s_max - inv))

    x_int ← int(round(x))
    return [x_int]
```

### `OrderUpToCapacityPolicy` notes (implementation details)

- The code intentionally **re-checks constraints after rounding** to a multiple of
  `dx`. This is important because rounding can push `x` slightly above a limit.
- The final action is returned as `np.array([x], dtype=float)` with `x` being
  integer-valued.

## How to choose parameters (quick intuition)

- `target_level` controls the aggressiveness of replenishment (bigger target →
  more inventory on average).
- `x_max` limits responsiveness (too small → frequent stockouts; too large → less
  realistic / may destabilize learning experiments).
- `dx` models ordering in case packs (e.g., `dx=12` means 0, 12, 24, …).
- `s_max` enforces a hard storage/capacity constraint.

## Small naming note

In your message you mentioned `PolicyOrderUpToCapacity`. In the current code the
class is named `OrderUpToCapacityPolicy`.

If you want, I can also add a backwards-compatible alias so both names work
(without breaking notebooks).
