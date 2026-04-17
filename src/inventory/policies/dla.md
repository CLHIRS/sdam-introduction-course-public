# DLA policies (lookahead via MILP-MPC and MCTS-UCT)

This note explains the DLA (direct lookahead approximation) policies implemented in:

- `inventory.policies.dla_milp` (see [dla_milp.py](dla_milp.py)): `DlaMpcMilpPolicy`
- `inventory.policies.dla_mcts` (see [dla_mcts.py](dla_mcts.py)): `DlaMctsUctPolicy`

Both policies use the **model** to look ahead a few steps and pick an action.
They differ in how they do the lookahead:

- **MILP-MPC**: solve a deterministic planning optimization problem over a short horizon.
- **MCTS-UCT**: simulate many random futures and use a tree search to choose actions.

Throughout we assume a cost-minimization framing.

---

## Common conventions

### State and action

- The state `state` is a 1-D NumPy vector.
- Inventory on hand is always `state[0]`.
- The action is always a length-1 vector `action = [x]` where $x \ge 0$ is the order quantity.

### Discrete action grid (batch size `dx`)

Both policies restrict orders to a discrete set:

$$\mathcal{X} = \{0, dx, 2dx, \dots, x_{max}\}$$

This makes planning/search tractable and matches many “case pack” settings.

### What the model must provide

Both policies store a `model: DynamicSystemMVP` and use:

- `model.exogenous_model.sample(S, X, t, rng) -> W_{t+1}`
- `model.cost_func(S, X, W, t) -> cost`
- `model.transition_func(S, X, W, t) -> S_{t+1}`

---

## `DlaMpcMilpPolicy` (deterministic MPC via MILP)

### Concept

At each time step $t$ we:

1. Build a deterministic forecast mean path $(\mu_0,\dots,\mu_{H-1})$.
2. Solve a deterministic inventory planning MILP over horizon $H$.
3. Execute only the **first** planned order $x_0$ (rolling horizon / MPC).

This is a direct lookahead approximation because it uses a **surrogate planning
problem** (deterministic mean demand) instead of the true stochastic system.

### Forecast interface

This policy can forecast the mean path in two ways:

- If `forecaster` is provided: uses `forecaster.forecast_mean_path(state, t, H, info=...)`.
- Otherwise it falls back to an “expert” mean based on the known exogenous model:
  - `ExogenousPoissonSeasonal`: uses `lambda_t(t+k)`
  - `ExogenousPoissonRegime`: propagates the regime distribution using `P` and computes $E[\lambda]$

### Optimization model (what the MILP represents)

The MILP is a *lost-sales, immediate replenishment* planning model.

Variables per planning step $k=0..H-1$:

- $q_k$ integer order batches, with $x_k = dx \cdot q_k$
- $y_k \in \{0,1\}$ order indicator (fixed cost)
- $s_k \ge 0$ sales
- $\ell_k \ge 0$ lost sales

State variables inside the MILP:

- $I_k \ge 0$ for $k=0..H$ (inventory)

Demand split identity:

$$s_k + \ell_k = \mu_k$$

Availability constraint (sales limited by inventory after ordering):

$$s_k \le I_k + x_k$$

Inventory update:

$$I_{k+1} = I_k + x_k - s_k$$

Optional capacity constraint (pre-demand on-hand after ordering):

$$I_k + x_k \le S_{max}$$

Objective (minimize):

$$\sum_{k=0}^{H-1}\Big(c x_k + K y_k + h I_{k+1} + b\ell_k - p s_k\Big)$$

### Interface

- `act(state, t, info=None) -> [x]`

### Pseudocode: greedy MPC step (`act`)

```text
function ACT_DlaMpcMilpPolicy(state, t, info=None):
    inv0 ← float(state[0])
    H ← self.H

    mu[0..H-1] ← FORECAST_MEAN_PATH(state, t, H, info)

    if SciPy milp not available:
        return [FALLBACK_ORDER(inv0, mu[0])]

    build MILP with variables (q_k, y_k, s_k, l_k, I_k)
    using constraints:
        s_k ≤ I_k + dx*q_k
        s_k + l_k = mu_k
        I_{k+1} = I_k + dx*q_k - s_k
        q_k ≤ q_max*y_k and y_k ≤ q_k
        (optional) I_k + dx*q_k ≤ S_max
        I_0 = inv0

    solve MILP
    if solve fails:
        return [FALLBACK_ORDER(inv0, mu[0])]

    q0 ← optimal q_0
    x0 ← dx * round(q0)

    # enforce practical limits (cap + optional capacity)
    x0 ← clip to [0, x_max]
    if S_max exists:
        x0 ← min(x0, max(0, S_max - inv0))

    return [int(round(x0))]
```

### Notes (implementation details)

- The fallback heuristic is an order-up-to rule using the next mean demand: $x=\max(0,\mu_0-I_0)$,
  then caps/rounds it.
- The MILP uses `scipy.optimize.milp`. If you run on an environment without SciPy,
  you will always get the fallback.

---

## `DlaMctsUctPolicy` (stochastic lookahead via MCTS + UCT)

### Concept

This policy uses Monte Carlo Tree Search (MCTS) to approximate the best action.

- It builds a search tree where nodes correspond to simulated future states.
- At each node it chooses actions using UCT (upper confidence bound applied to trees).
- It estimates action quality by running many simulated rollouts with the model.

Important implementation detail:

- The code stores **rewards** as negative costs: `reward = -cost`.
- It therefore **maximizes** total reward, which is equivalent to minimizing total cost.

### Interface

- `act(state, t, info=None) -> [x]`

### CRN compatibility

When the dynamic system uses strict CRN evaluation, it passes a per-step seed in:

- `info["crn_step_seed"]`

This policy reads that seed and mixes it into its internal planning seed so that
planning is reproducible and comparable across policies.

### Pseudocode: MCTS-UCT action (`act`)

```text
function ACT_DlaMctsUctPolicy(state, t, info=None):
    actions ← {0, dx, ..., x_max}

    root ← new Node(untried=actions, children={}, N=0, W=0)

    step_seed ← 0
    if info contains "crn_step_seed":
        step_seed ← info["crn_step_seed"]

    # Make simulation RNG streams reproducible
    sim_seqs ← SeedSequence([planning_seed, step_seed, t]).spawn(n_simulations)

    for sim in 1..n_simulations:
        rng ← default_rng(sim_seqs[sim].generate_state(1)[0])

        node ← root
        S ← state
        t_sim ← t
        depth ← 0
        path ← [node]
        rewards ← []

        # 1) Selection + Expansion
        while depth < horizon:
            if node has an untried action:
                a ← pop one untried action
                node.children[a] ← new Node(untried=actions)

                S, cost ← STEP_MODEL(S, a, t_sim, rng)
                rewards.append(-cost)

                node ← node.children[a]
                path.append(node)
                depth += 1
                t_sim += 1
                break

            if node has no children:
                break

            a ← UCT_SELECT(node, uct_c)
            S, cost ← STEP_MODEL(S, a, t_sim, rng)
            rewards.append(-cost)

            node ← node.children[a]
            path.append(node)
            depth += 1
            t_sim += 1

        # 2) Rollout (default policy)
        while depth < horizon:
            a ← ROLLOUT_ACTION(S, t_sim)   # heuristic order-up-to mean
            S, cost ← STEP_MODEL(S, a, t_sim, rng)
            rewards.append(-cost)
            depth += 1
            t_sim += 1

        G ← sum(rewards)

        # 3) Backpropagate
        for nd in path:
            nd.N += 1
            nd.W += G

    if root has no children:
        return [0]

    # pick action with most visits
    best_a ← argmax_a root.children[a].N
    best_a ← clip best_a to [0, x_max]

    return [best_a]
```

### Pseudocode: UCT selection (`uct_select`)

Within a node, UCT chooses the action maximizing:

$$\text{UCT}(a) = \bar Q(a) + c\sqrt{\frac{\log N}{N_a}}$$

where:

- $\bar Q(a) = W_a / N_a$ is the mean reward estimate for child $a$.
- $N$ is the parent visit count.
- $N_a$ is the child visit count.
- $c$ is `uct_c`.

### Notes (implementation details)

- Because rewards are negative costs, MCTS is searching for the lowest-cost actions.
- The rollout policy is a simple order-up-to heuristic based on the model’s exogenous
  structure (seasonal/regime Poisson). This keeps rollouts cheap.
- The final decision uses “most visited action” rather than “highest mean value”.

---

## Small naming note (aliases)

The modules define backwards-compatible names used in older notebooks:

- `DLA_MPC_MILP` is an alias for `DlaMpcMilpPolicy`.
- `DLA_MCTS_UCT` is an alias for `DlaMctsUctPolicy`.
