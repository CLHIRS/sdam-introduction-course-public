# CFA policies via MILP (forecast-then-optimize)

This note explains the two CFA (cost function approximation) policies implemented in
`inventory.policies.cfa_milp` (see [cfa_milp.py](cfa_milp.py)):

- `OrderMilpCfaPolicy` (no lead time)
- `OrderMilpCfaLeadTimePolicy` (explicit lead-time pipeline)

Both policies follow the same high-level idea:

- Forecast demand over a short horizon $H$.
- Treat that forecast path as “deterministic future demand”.
- Solve a small **mixed-integer linear program (MILP)** to choose orders.
- Execute only the **first** order decision (rolling horizon / MPC style).

In the lecture framing, we evaluate policies by **minimizing total cost**.
These MILPs therefore minimize a proxy objective that combines ordering cost,
holding cost, and stockout/lost-sales penalties (and optionally subtracts sales
revenue).

---

## Common conventions

### State / action vectors

- States and actions are always 1-D NumPy arrays.
- The action is always a length-1 vector `action = [x]` where $x \ge 0$ is the
  order quantity.

### Batch ordering (`dx`) and order cap (`x_max`)

Both policies enforce:

- **Order cap**: $0 \le x \le x_{max}$
- **Batching**: $x$ is a multiple of `dx` via an integer batch variable $q$:

$$x = dx \cdot q, \quad q \in \{0,1,2,\dots,q_{max}\}$$

where $q_{max} = \lfloor x_{max}/dx \rfloor$.

### Forecast interface

The policies require a forecaster object that can produce a mean forecast path
of length $H$.

Supported forecaster APIs:

- Preferred: `forecaster.forecast(state, t, H, info=...)` returning an object with
  a `.mean` vector.
- Fallback: `forecaster.forecast_mean_path(state, t, H, info=...)`.

The forecasted mean path is clipped to be nonnegative.

### Solver availability and fallback

The policies try to solve the MILP using `scipy.optimize.milp`.

If SciPy’s MILP solver is unavailable or the solve fails, they fall back to a
simple “order-up-to next-period mean” rule.

---

## MILP building blocks (shared idea)

To keep the MILP linear, the code uses a standard lost-sales split of demand:

- $s_k$ = realized sales (decision variable, bounded by demand)
- $\ell_k$ = lost sales (decision variable)
- $\mu_k$ = forecast mean demand for step $k$

with the identity:

$$s_k + \ell_k = \mu_k$$

and sales constrained by availability (details differ by lead-time model).

### Objective (cost-minimization proxy)

Per time step $k$, the MILP includes terms (names match parameters in code):

- ordering cost: `c * x_k`
- fixed cost: `K * y_k` where $y_k \in \{0,1\}$ indicates whether we order
- holding cost: `h * I_{k+1}`
- stockout penalty: `b * l_k`
- sales revenue (as negative cost): `-p * s_k`

So the MILP objective minimized is:

$$\sum_{k=0}^{H-1}\Big(c x_k + K y_k + h I_{k+1} + b \ell_k - p s_k\Big)$$

This is a *planning objective* built from forecasts; the true simulation cost in
the environment may differ.

---

## `OrderMilpCfaPolicy` (no lead time)

### Concept

This policy assumes **immediate replenishment**:

- If you order $x_k$ at step $k$, it is available immediately.

The within-MILP inventory state is:

- $I_k$ = on-hand inventory before ordering at MILP step $k$.

Sales are limited by what you have after ordering:

$$s_k \le I_k + x_k$$

### `OrderMilpCfaPolicy` pseudocode

Inputs:

- current inventory `inv0 = state[0]`
- forecast horizon `H`
- mean forecast path `mu[0..H-1]`
- parameters `dx, x_max, S_max (optional), p,c,h,b,K`

Algorithm:

```text
function ACT_OrderMilpCfaPolicy(state, t, info=None):
    inv0 ← float(state[0])
    H ← self.H
    mu ← FORECAST_MEAN_PATH(state, t, H, info)   # nonnegative length-H

    if SciPy milp not available:
        return [FALLBACK_ORDER(inv0, mu[0])]

    # Decision variables for each k=0..H-1
    #   q_k integer batches, x_k = dx*q_k
    #   y_k binary order indicator (for fixed cost K)
    #   s_k sales (continuous)
    #   l_k lost sales (continuous)
    # Inventory states I_k for k=0..H

    minimize  Σ_k ( c*dx*q_k + K*y_k + h*I_{k+1} + b*l_k - p*s_k )

    subject to for k=0..H-1:
        (1) s_k + l_k = mu_k
        (2) s_k ≤ I_k + dx*q_k
        (3) I_{k+1} = I_k + dx*q_k - s_k

        (4) 0 ≤ q_k ≤ q_max,  q_k integer
            0 ≤ y_k ≤ 1,      y_k binary
            0 ≤ s_k ≤ mu_k
            0 ≤ l_k ≤ mu_k

        (5) link fixed-cost indicator (if K>0, still safe if K=0):
            q_k ≤ q_max * y_k
            y_k ≤ q_k

        (6) optional storage cap after ordering (if S_max is set):
            I_k + dx*q_k ≤ S_max

    and fix initial inventory:
        I_0 = inv0

    solve MILP
    if solve fails:
        return [FALLBACK_ORDER(inv0, mu[0])]

    # rolling horizon: execute only first order
    q0 ← optimal q_0
    x0 ← dx * round(q0)
    x0 ← clip to [0, x_max]
    if S_max exists:
        x0 ← min(x0, max(0, S_max - inv0))

    return [int(round(x0))]
```

### `OrderMilpCfaPolicy` notes

- The code solves an MILP over horizon $H$, but **only returns** the first action
  $x_0$.
- `S_max` is enforced as a pre-demand cap inside the plan: $I_k + x_k \le S_{max}$.
- The fallback is essentially an order-up-to rule with target equal to the next
  forecast mean.

---

## `OrderMilpCfaLeadTimePolicy` (lead time via pipeline)

### Concept

This version models lead time by including the pipeline directly in the state.

State convention (for lead time `L`):

$$S_t = [I_t, p_{1,t}, \dots, p_{L,t}]$$

- $I_t$ is on-hand inventory.
- $p_{j,t}$ is inventory scheduled to arrive in $j$ steps.

Action convention:

- $X_t = [x_t]$ enters the **tail** of the pipeline and arrives after $L$ steps.

Inside the MILP, we keep track of both:

- inventory $I_k$ for $k=0..H$
- pipeline $P_{k,j}$ for $k=0..H$ and $j=1..L$

Sales in step $k$ can only use on-hand plus immediate arrivals:

$$s_k \le I_k + P_{k,1}$$

### `OrderMilpCfaLeadTimePolicy` pseudocode

Inputs:

- current state `state = [I0, p1_0, ..., pL_0]`
- forecast `mu[0..H-1]`
- parameters `H, L, dx, x_max, S_max (optional), p,c,h,b,K`

Algorithm:

```text
function ACT_OrderMilpCfaLeadTimePolicy(state, t, info=None):
    H ← self.H
    L ← self.L
    mu ← FORECAST_MEAN_PATH(state, t, H, info)

    if SciPy milp not available:
        return FALLBACK_LEADTIME_ORDER(state, mu[0])

    # Variables per k=0..H-1
    #   q_k integer batches, x_k = dx*q_k
    #   y_k binary indicator
    #   s_k sales
    #   l_k lost sales
    # States inside MILP:
    #   I_k for k=0..H
    #   P_{k,j} for k=0..H, j=1..L

    minimize  Σ_k ( c*dx*q_k + K*y_k + h*I_{k+1} + b*l_k - p*s_k )

    subject to for k=0..H-1:
        (1) s_k + l_k = mu_k

        (2) availability and inventory update:
            s_k ≤ I_k + P_{k,1}
            I_{k+1} = I_k + P_{k,1} - s_k

        (3) pipeline shift:
            P_{k+1,j} = P_{k,j+1}          for j=1..L-1
            P_{k+1,L} = dx*q_k            (new order enters the tail)

        (4) bounds and integrality:
            0 ≤ q_k ≤ q_max,  q_k integer
            0 ≤ y_k ≤ 1,      y_k binary
            0 ≤ s_k ≤ mu_k
            0 ≤ l_k ≤ mu_k
            I_k ≥ 0,  P_{k,j} ≥ 0

        (5) fixed-cost indicator link:
            q_k ≤ q_max * y_k
            y_k ≤ q_k

        (6) optional pre-demand on-hand cap (if S_max is set):
            I_k + P_{k,1} ≤ S_max

    and fix initial conditions from state:
        I_0 = state[0]
        P_{0,1..L} = state[1..L]

    solve MILP
    if solve fails:
        return FALLBACK_LEADTIME_ORDER(state, mu[0])

    q0 ← optimal q_0
    x0 ← dx * round(q0)
    x0 ← clip to [0, x_max]

    return [int(round(x0))]
```

### `OrderMilpCfaLeadTimePolicy` fallback pseudocode

The lead-time fallback uses an “inventory position” rule.

Define inventory position:

$$IP = I_0 + \sum_{j=1}^L p_{j,0}$$

Then order toward the next forecast mean:

```text
function FALLBACK_LEADTIME_ORDER(state, mu0):
    I0 ← state[0]
    pipe_sum ← sum(state[1..L])   # 0 if L=0

    IP ← I0 + pipe_sum
    x ← max(0, mu0 - IP)
    x ← min(x, x_max)

    if dx > 1:
        x ← dx * round(x / dx)

    return [int(round(max(0, x)))]
```

### `OrderMilpCfaLeadTimePolicy` notes

- This is a clean way to incorporate lead time without changing the dynamic
  system core: the state simply includes pipeline entries.
- The capacity `S_max` is applied to **pre-demand on-hand**:
  $I_k + P_{k,1} \le S_{max}$.

---

## Small naming note (aliases)

The module also defines backwards-compatible aliases used in older notebooks:

- `Order_MILP_CFA` is an alias for `OrderMilpCfaPolicy`.
- `Order_MILP_CFA_LeadTime` is an alias for `OrderMilpCfaLeadTimePolicy`.
