# Dynamic Programming (DP) for 1D Inventory

This note explains the two main DP components implemented in `inventory.policies.dp` (see [dp.py](dp.py)):

- `DynamicProgrammingSolver1D`: computes an *optimal finite-horizon ordering plan* for a 1D inventory system.
- `DynamicProgrammingPolicy`: turns the computed plan into a *policy object* that can be simulated inside the course framework.

The goal is to connect the *general DP concept* (backward induction) to the *exact table-based implementation* students see in code.

---

## Problem Setup (What DP is solving)

We consider a finite-horizon inventory problem with:

- Time: $t = 0, 1, \dots, T-1$
- State: inventory level $S_t \in \{0,1,\dots,S_{\max}\}$
- Action: order quantity $x_t \in \{0, dx, 2dx, \dots, x_{\max}\}$
  - Feasibility constraint: $x_t \le S_{\max} - S_t$ (cannot exceed capacity)
- Exogenous demand: $D_{t+1} \sim \text{Poisson}(\lambda_t)$
  - The exogenous model is required to provide `lambda_t(t)`.

One-step dynamics (conceptually):

- On-hand inventory after ordering: $\text{OH} = S_t + x_t$
- Sales: $\min(\text{OH}, D_{t+1})$
- End inventory: $S_{t+1} = \max(\text{OH} - D_{t+1}, 0)$
- Lost sales (backorder-free version): $\max(D_{t+1} - \text{OH}, 0)$

One-step cost components (as implemented):

- Ordering cost: $c\,x_t$
- Holding cost: $h\,S_{t+1}$
- Lost sales penalty: $b\,\max(D_{t+1} - \text{OH}, 0)$
- Sales revenue: $p\,\min(\text{OH}, D_{t+1})$

The implementation uses a cost-minimization convention where revenue is included as a negative cost term:

$$\text{immediate\_cost} = c x_t + h S_{t+1} + b\,\text{lost} - p\,\text{sales}.$$

An optional terminal cost can be added:

$$V_T(S_T) = \text{terminal\_cost\_per\_unit} \cdot S_T.$$

---

## `DynamicProgrammingSolver1D` (Concept + Pseudocode)

### What it produces

The solver returns a `DPSolution1D` containing:

- `V[t, s]`: optimal value-to-go at time $t$ from inventory level $s$
- `x_star[t, s]`: optimal order decision at time $t$ if inventory is $s$

Think of `x_star` as a table that “precomputes” the best action for every state and time.

### Bellman recursion (core DP idea)

For $t = T-1, \dots, 0$ and each inventory level $s$:

$$V_t(s) = \min_{x \in \mathcal{X}(s)} \mathbb{E}\big[\,g(s, x, D_{t+1}) + \gamma V_{t+1}(S_{t+1})\,\big],$$

where:

- $\mathcal{X}(s)$ are feasible actions (respecting capacity and step size)
- $g(\cdot)$ is the one-step cost
- $\gamma$ is the discount factor

### Pseudocode: overall solver

```text
Inputs:
  exog with lambda_t(t)
  horizon T
  capacity S_max
  action grid {0, dx, 2dx, ..., x_max}
  cost params p, c, h, b
  terminal_cost_per_unit
  discount gamma
  expectation_mode in {"truncation", "scenarios"}
  demand approximation parameters (D_max OR K, q_lo, q_hi)

Initialize:
  V[T, s] = terminal_cost_per_unit * s    for s = 0..S_max
  x_star[t, s] = 0                        for all t,s

For t = T-1 down to 0:
  For each inventory s in 0..S_max:
    best_val = +infty
    best_x = 0

    For each action x in {0, dx, 2dx, ..., x_max}:
      If x > S_max - s: continue  (capacity feasibility)

      Compute expectation over demand:
        Q(s,x) = E[ immediate_cost(s,x,D) + gamma * V[t+1, next_inventory(s,x,D)] ]

      If Q(s,x) < best_val:
        best_val = Q(s,x)
        best_x = x

    V[t, s] = best_val
    x_star[t, s] = best_x

Return (V, x_star, meta)
```

### How the expectation is computed (two modes)

In inventory DP, the only randomness is demand, so the expectation is:

$$\mathbb{E}[\cdot] = \sum_d \Pr(D=d)\,(\cdot).$$

But Poisson has infinite support, so we approximate it.

#### Mode A: `expectation_mode="truncation"`

Idea:

- Compute Poisson PMF from $d=0$ up to $D_{\max}$.
- Lump the remaining tail mass into a single “tail event”.

Pseudocode:

```text
Given lambda and a truncation limit D_max:
  pmf[0] = exp(-lambda)
  pmf[d] = pmf[d-1] * lambda / d   for d=1..D_max
  tail = max(0, 1 - sum_{d=0..D_max} pmf[d])

Expectation for a fixed (s,x):
  exp = sum_{d=0..D_max} pmf[d] * cost_to_go(s,x,d)
  exp += tail * cost_to_go(s,x, d_tail)

where d_tail is represented in code as D_max+1.
```

Notes for students:

- The “tail event” is a conservative approximation: it treats *all* demand larger than $D_{\max}$ as if it were $D_{\max}+1$.
- The truncation threshold is chosen automatically when `D_max=None` via a heuristic:
  $D_{\max} \approx \lambda + z\sqrt{\lambda} + \text{extra}$.

#### Mode B: `expectation_mode="scenarios"`

Idea:

- Approximate Poisson by $K$ scenarios of demand.
- Scenarios are chosen by splitting a quantile interval into equal-probability bins.

Pseudocode:

```text
Given lambda, number of scenarios K, and quantile range [q_lo, q_hi]:
  Split [q_lo, q_hi] into K equal bins
  For each bin, take its mid-quantile q_mid
  Scenario demand d_k = PoissonQuantile(lambda, q_mid)
  Scenario weights w_k = (q_hi-q_lo)/K

  Add tails into endpoints:
    w_0 += q_lo
    w_last += (1 - q_hi)

Expectation for (s,x):
  exp = sum_{k=1..K} w_k * cost_to_go(s,x,d_k)
```

Notes:

- This tends to be faster than full truncation when $\lambda$ is large.
- It also makes it easier to “control” the number of demand points you average over.

---

## `DynamicProgrammingPolicy` (How the DP solution becomes a policy)

Once DP computes `x_star[t, s]`, we want a policy object with an `act(state, t)` method.

The class `DynamicProgrammingPolicy` does exactly that:

- It reads the current inventory from `state[0]`.
- It uses time `t` and inventory `s` to look up `x_star[t, s]`.
- It clamps the result for safety:
  - $0 \le x \le x_{\max}$
  - rounds to a multiple of `dx`
  - ensures $x \le S_{\max} - s$ (capacity feasibility)

Pseudocode:

```text
Given x_star[t, s], and parameters S_max, dx, x_max:

act(state, t):
  s = clamp(round(state[0]), 0, S_max)

  if t < 0 or t >= T:
    x = 0
  else:
    x = x_star[t, s]

  x = clamp(x, 0, x_max)
  if dx > 1:
    x = dx * round(x / dx)

  x = min(x, S_max - s)

  return [x] as a length-1 action vector
```

Interpretation:

- The DP policy is a *table policy*: it does not “solve” anything online.
- All the heavy computation is done once by the solver.

---

## Common student questions (quick answers)

### Why does the solver require `lambda_t(t)`?

Because this DP version assumes demand is Poisson with a time-varying mean $\lambda_t$, and `lambda_t(t)` is the interface used to obtain that mean.

### Why is revenue subtracted in a cost-minimization DP?

Because we define the one-step objective as a *net cost* that includes a negative revenue term. The DP still minimizes expected total cost; it just treats revenue as a cost reduction.

### What does `gamma` do?

`gamma` discounts future value-to-go:

- $\gamma = 1$ means no discounting
- smaller values prioritize immediate cost

---

## If you want this doc to match your lecture notation

Tell me (1) whether you present this as “cost minimization” or “profit maximization”, and (2) whether you want the state/action written as $S_t, x_t$ or other symbols. I can align the notation exactly to your slides.
