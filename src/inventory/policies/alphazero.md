# AlphaZero-style hybrid policy (PUCT-MCTS + tiny policy/value net)

This note explains the AlphaZero-inspired policy implemented in
`inventory.policies.alphazero` (see [alphazero.py](alphazero.py)).

The key class is:

- `HybridAlphaZeroPolicy`

It is “hybrid” in the sense that it combines:

- **Direct lookahead (DLA)**: a short-horizon search (MCTS)
- **Learning (VFA/PFA flavor)**: a small network that provides
  - a **policy prior** (which actions look promising)
  - a **value estimate** (how good the state looks)

Compared to the MCTS-only DLA policies, the network guidance makes the search
more focused.

## Common conventions

- **State** is a 1-D vector.
  - Inventory on hand is `state[0]`.
  - Some models include a discrete regime indicator in `state[1]`.
- **Action** is a length-1 vector, `action[0] = order_quantity`.
- Costs are minimized, so the MCTS search uses **reward** `r = -cost`.

## High-level idea (concept)

At a decision time $(S_t, t)$, AlphaZero-style planning does:

1) Use a learned network to compute
   - a prior distribution over actions $p_\theta(a\mid S_t, t)$
   - a value estimate $v_\theta(S_t, t)$
2) Run **PUCT-MCTS** simulations to compute improved action preferences.
3) Turn visit counts into an improved policy $\pi(\cdot\mid S_t, t)$.
4) Act greedily (argmax) during evaluation.

Training (optional) alternates between:

- **self-play** (generate data using MCTS) and
- **supervised learning** to fit the network to the MCTS targets.

### Mapping to classic AlphaZero terminology

If you have seen AlphaZero in other settings (Go/Chess), the vocabulary maps
almost directly:

- **policy/value network**: `_TinyMLP_PV` produces `(p, v)`
  - `p` is the prior $p_\theta(a\mid s)$ used in PUCT.
  - `v` is the leaf value used when we stop rollouts at depth `H`.
- **PUCT**: the tree policy `Q + U` with exploration term based on `c_puct` and
  the prior.
- **root Dirichlet noise** (training only): mixes priors with Dirichlet noise to
  encourage exploration early.
- **search policy** `π`: derived from MCTS visit counts (with temperature `tau`).
- **targets for training**:
  - policy target is `π` (from visits)
  - value target is return-to-go `G` (scaled by `value_scale`)

## Strict CRN compatibility (important for this course)

`DynamicSystemMVP` can pass a per-step seed via `info["crn_step_seed"]`.

This policy is written to keep CRN evaluations meaningful:

- The MCTS procedure is **deterministic given** `(state, t)` (it uses an internal
  decision seed derived from state/time).
- Any extra exploration noise (Dirichlet) is used **only during training**, and
  only at the root.
- If you sample from the MCTS policy in `act`, the sampling RNG can optionally
  be seeded using `info["crn_step_seed"]`.

---

## `HybridAlphaZeroPolicy`

### `HybridAlphaZeroPolicy` interface

Constructor (key parameters only):

- `model: DynamicSystemMVP`
- action grid: `actions` (or `x_max` + `dx` to build `0, dx, 2dx, ...`)
- feasibility: `s_max` (optional storage capacity)
- network backend:
  - `net_backend`: `'numpy'` (default tiny net) or `'torch'` (deeper MLP; usually learns better)
  - `device`: e.g. `'cpu'` (and `'cuda'` if available)
- search: `H` (planning depth), `n_sims` (MCTS simulations), `c_puct`
- transposition control: `transposition_key`
  - `'auto'` (default): inventory only for 1D states, inventory + first regime for multi-state problems
  - `'inventory_only'`: coarsest tree abstraction
  - `'inventory_plus_first_regime'`: compromise abstraction for regime problems
  - `'full_state'`: use the full observed state in the tree key
- evaluation temperature: `tau_eval` (often 0)
- training noise: `dirichlet_alpha`, `dirichlet_eps`
- discount: `gamma` (often 1.0 here)

Policy API:

- `act(state, t, info=None) -> action`

Training API:

- `fit_self_play(S0, T, ...) -> history`

Training-specific search knobs (useful in notebooks):

- `n_sims_selfplay`: override the number of MCTS simulations *during self-play only*.
  This lets you keep self-play data generation cheap while still evaluating with a larger `n_sims`.
- `H_selfplay`: override the planning depth *during self-play only*.
  A common trick early in training is to use a larger `H_selfplay` (even `T`) to reduce reliance on an
  initially weak value function.

### `HybridAlphaZeroPolicy` concept

You can think of this as a “search-enhanced greedy policy”:

- If we had a perfect value function $V(S_t)$, we could do one-step lookahead.
- If we had a perfect model, we could do deep lookahead.
- Here we approximate both:
  - MCTS provides the “deep lookahead” part.
  - The network provides (i) priors to guide MCTS and (ii) leaf values to stop
    rollouts early.

---

## Building blocks (implementation-oriented)

### Features: `_featurize(state, t, T)`

The network input keeps the full observed state prefix, but scales it before
feeding it to the network.

- Includes the full observed state `state[:d_state]`.
- Normalizes inventory by `s_max` when available (otherwise by `x_max`).
- Normalizes regime coordinates by their maximum label inferred from the
  exogenous model.
- Adds time features: `frac = t/(T-1)`, plus `sin(2π frac)` and `cos(2π frac)`.

Pseudocode:

```text
function TIME_FEATURES(t, T):
    frac ← 0                     if T ≤ 1
            else t / (T - 1)
    return [ frac,
             sin(2π·frac),
             cos(2π·frac) ]

function FEATURIZE(state, t, T):
  observed ← NORMALIZE(state[:d_state])
  return observed ⧺ TIME_FEATURES(t, T)
```

### Feasibility mask: `_feasible_mask(state)`

If `s_max` is provided, enforce `inv + x ≤ s_max`.

Pseudocode:

```text
function FEASIBLE_MASK(state):
    if s_max is None:
        return [True for each action]

    inv ← float(state[0])
    cap ← max(0, s_max - inv)
    cap_int ← floor(cap)
    return [ actions[a] ≤ cap_int for a in 0..A-1 ]
```

### Policy/value network backends

The implementation supports two backends:

- `_TinyMLP_PV` (NumPy): minimal, lecture-friendly, fast to read.
- `_TorchMLP_PV` (PyTorch): deeper MLP that typically provides a much better value approximation
  during self-play training.

In code this is controlled by the constructor argument `net_backend`.

#### Tiny policy/value network: `_TinyMLP_PV`

The network is a small NumPy MLP with:

- shared trunk (one hidden layer with ReLU)
- policy head producing action probabilities `p`
- value head producing a scalar `v`

Pseudocode (forward pass):

```text
function NET_PREDICT(x):
    h ← ReLU(W1·x + b1)

    logits ← Wp·h + bp
    p ← softmax(logits)

    v ← (Wv·h + bv)              # scalar
    return (p, v)
```

Training step uses a standard AlphaZero-style loss:

- cross entropy on the policy target `π`
- squared error on the value target `z`

$$\mathcal{L}(\theta) = CE(\pi, p_\theta) + \lambda (z - v_\theta)^2 + \text{L2}$$

(Here `lambda` is `value_weight` in the code.)

#### Torch MLP backend: `_TorchMLP_PV`

The Torch backend uses a deeper trunk (two ReLU layers) plus separate policy and value heads.

Why it helps:

- The value function is often the main bottleneck in AlphaZero-style control.
- A deeper network can fit the return-to-go targets better, which makes MCTS backups less noisy.

Usage example:

```python
pol = HybridAlphaZeroPolicy(
  system,
  x_max=480,
  dx=10,
  s_max=480,
  net_backend="torch",
  device="cpu",
  hidden=64,
  H=10,
  n_sims=150,
)
```

---

## PUCT-MCTS planning (core algorithm)

### What is stored in the tree?

- A **node** corresponds to a bucketed representation of `(t, state)`.
- For each action index `a`, the node stores an **edge** with:
  - `N(a)` visit count
  - `W(a)` total return accumulated
  - `Q(a) = W(a)/N(a)` mean return

The node also stores `priors[a]` from the network, masked to feasible actions.

The bucketing detail is controlled by `transposition_key`.

- `inventory_only`: merge all states with the same inventory bucket.
- `inventory_plus_first_regime`: merge by inventory bucket and the first regime coordinate.
- `full_state`: keep the full observed state tuple in the tree key.

In practice, `full_state` is the most faithful abstraction, but it can fragment
the search tree and require more simulations. For multi-regime lecture notebooks,
`inventory_plus_first_regime` is often a better compute/quality compromise.

### Deterministic per-decision seed: `_decision_seed(state, t)`

To keep MCTS deterministic at each decision (important for CRN evaluation), the
policy derives a seed from the same abstraction used for the transposition key,
plus time `t` and the policy’s base `seed`.

Pseudocode:

```text
function DECISION_SEED(state, t):
  sig ← TRANSPOSE_SIGNATURE(state)
  x ← (t * 97531) XOR (seed * 3643)
  for each value in sig:
    x ← MIX(x, value)
    return x mod 2^32
```

### MCTS improvement: `_mcts_improved_pi(state, t, T, tau, training)`

This runs `n_sims` simulations from the root and returns an improved policy `π`.

Main steps:

1) Build a tree keyed by a bucketed state abstraction and time `t`.
2) Repeatedly perform one simulation:
   - selection with PUCT
   - model step (sample exogenous, compute cost, transition)
   - recursion until depth `H` then evaluate the value network
   - backpropagate returns to update `(N, W, Q)`
3) Convert root visit counts into `π` using temperature `tau`.

#### Node creation and masked priors

```text
function MASKED_PRIORS(S, t):
    x ← FEATURIZE(S, t, T)
    (p, v) ← NET_PREDICT(x)

    feas ← FEASIBLE_MASK(S)
    p ← p ⊙ feas
    if sum(p) is ~0:
        p ← uniform over feasible actions
    else:
        p ← p / sum(p)

    return p
```

During training, the root priors are perturbed using Dirichlet noise:

```text
if training and this is the ROOT node:
    noise ← Dirichlet(alpha · 1)
    noise ← normalize(noise ⊙ feas)
    p ← (1 - eps) · p + eps · noise
```

#### PUCT action selection

For each feasible action index `a`:

$$U(a) = c_{puct} \; p(a) \; \frac{\sqrt{N}}{1 + N(a)}$$

Choose `a` maximizing `Q(a) + U(a)`.

##### Tiny worked example (one PUCT decision)

Suppose we are at a node with total visits `node.N = 100`, and we compare two
feasible actions `a=0` and `a=1`.

- `c_puct = 1.5`
- priors: `p(0)=0.60`, `p(1)=0.40`
- edge stats:
  - `N(0)=50`, `Q(0)=1.20`
  - `N(1)=10`, `Q(1)=0.90`

Compute the exploration bonuses:

$$U(a) = c_{puct} \; p(a) \; \frac{\sqrt{N}}{1 + N(a)}$$

Here $\sqrt{N} = \sqrt{100}=10$.

- For action 0:
  $$U(0) = 1.5 \cdot 0.6 \cdot \frac{10}{1+50} \approx 0.176$$
  score $= Q(0)+U(0) \approx 1.20 + 0.176 = 1.376$

- For action 1:
  $$U(1) = 1.5 \cdot 0.4 \cdot \frac{10}{1+10} \approx 0.545$$
  score $= Q(1)+U(1) \approx 0.90 + 0.545 = 1.445$

Even though action 0 has a higher current `Q`, action 1 gets chosen because it
is *less explored* (smaller `N(a)`) so its exploration bonus is larger.

```text
function SELECT_ACTION(node, S):
    feas ← FEASIBLE_MASK(S)
    best_score ← -∞

    for each action index a:
        if not feas[a]:
            continue
        U ← c_puct * priors[a] * sqrt(node.N) / (1 + N(a))
        score ← Q(a) + U
        take the a with highest score

    return best_a
```

#### One simulation / rollout

The environment model is used inside the simulation:

- exogenous sample: `W ~ exogenous_model.sample(S, x, t, rng)`
- cost: `c = cost_func(S, x, W, t)`
- transition: `S' = transition_func(S, x, W, t)`

Terminal logic:

- If we reached planning depth `H`, stop and use `v_theta(S,t)`.
- If we reached time `T`, return 0.

```text
function ROLLOUT(S, t, depth):
    if t ≥ T:
        return 0

    if depth ≥ H:
        x_feat ← FEATURIZE(S, t, T)
        (_, v) ← NET_PREDICT(x_feat)
        return v

    node ← GET_NODE(S, t)
    a ← SELECT_ACTION(node, S)
    x ← actions[a]

    W ← exogenous_sample(S, x, t, rng)
    cost ← cost_func(S, x, W, t)
    S_next ← transition_func(S, x, W, t)

    r ← -cost
    G ← r + gamma * ROLLOUT(S_next, t+1, depth+1)

    # backprop updates for chosen edge
    node.N ← node.N + 1
    N(a) ← N(a) + 1
    W(a) ← W(a) + G
    Q(a) ← W(a) / N(a)

    return G
```

##### Tiny worked trace (one simulation, one backprop update)

This is what a *single* MCTS simulation does at the root (ignoring recursion
details):

```text
Given root node with node.N = 7
and the chosen action edge has (N(a)=2, W(a)=1.0, Q(a)=0.5)

1) Selection:
   a ← argmax_a [ Q(a) + U(a) ]

2) Model step (one environment transition inside the tree):
   x_qty ← actions[a]
   W_exog ← exogenous_model.sample(S, x_qty, t, rng)
   cost ← cost_func(S, x_qty, W_exog, t)
   S_next ← transition_func(S, x_qty, W_exog, t)
   r ← -cost

3) Leaf evaluation / recursion:
   child_return ← 0                    if t+1 ≥ T
                 else v_theta(S_next)  if depth+1 ≥ H
                 else ROLLOUT(S_next, t+1, depth+1)

4) Return for this simulation:
   G ← r + gamma * child_return

5) Backprop update (update counts and averages for the chosen edge):
   node.N ← node.N + 1                       # 7 → 8
   N(a)  ← N(a)  + 1                         # 2 → 3
   W(a)  ← W(a)  + G                         # 1.0 → 1.0 + G
   Q(a)  ← W(a) / N(a)                       # new mean return
```

The important intuition is: **every simulation adds exactly one sample return**
to exactly one edge on each visited node, and `Q(a)` becomes the running average
of those sampled returns.

#### Turning visits into `π`

After running `n_sims` rollouts, compute root visit counts `N(a)` and apply
feasibility masking.

- If `tau = 0`: pick argmax and return a one-hot policy.
- Else: return normalized `N(a)^(1/tau)`.

---

## Acting: `act(state, t, info)`

This method calls MCTS to compute `π` and then either samples or takes argmax.

Key `info` options used:

- `T`: horizon length (otherwise uses `T_default` or 10)
- `deterministic` (default `True`): whether to act greedily
- `det_mode`: if `"sample"` and `deterministic=False`, sample from `π`
- `tau`: temperature to convert visits to `π` (defaults to `tau_eval`)
- `crn_step_seed`: optional seed used only when sampling actions

Pseudocode:

```text
function ACT_HybridAlphaZeroPolicy(state, t, info=None):
    T ← info["T"] if provided else (T_default or 10)

    deterministic ← info.get("deterministic", True)
    det_mode ← info.get("det_mode", "argmax")
    tau ← info.get("tau", tau_eval)

    pi ← MCTS_IMPROVED_PI(state, t, T, tau, training=False)

    if (not deterministic) and det_mode == "sample":
        if info contains crn_step_seed:
            rng ← RNG(seed mixed from (policy.seed, crn_step_seed, t))
        else:
            rng ← RNG(seed = DECISION_SEED(state, t))
        a ← sample(pi)
    else:
        a ← argmax(pi)

    return [ actions[a] ]
```

Notes:

- During evaluation the default is greedy argmax for low variance.
- If you enable sampling, CRN seeding is respected when a `crn_step_seed` is
  provided.

---

## Training: `fit_self_play(...)`

This is a compact, lecture-friendly AlphaZero loop:

1) Self-play with MCTS to fill a replay buffer with triples `(features, π, G)`.
2) Train a candidate network on the buffer.
3) Use a strict-CRN evaluation gate to accept the candidate only if it improves
   mean total cost.

### Self-play data generation

For each episode:

- Start at `S0`.
- For each time `t`:
  - run MCTS with `training=True` and temperature `tau_train`
  - sample an action from `π`
  - step the model and store `(features, π, reward)`

Then compute **return-to-go** targets backward:

$$G_t = r_t + r_{t+1} + \dots + r_{T-1}$$

The code stores `G/value_scale` as the value target to keep magnitudes stable.

Pseudocode:

```text
function SELF_PLAY_EPISODE(S0):
    S ← S0
    traj ← empty list
    total_return ← 0

    for t = 0..T-1:
        pi ← MCTS_IMPROVED_PI(S, t, T, tau=tau_train, training=True)
        a ← sample(pi)                   # training exploration

        x ← actions[a]
        W ← exogenous_sample(S, x, t, ep_rng)
        cost ← cost_func(S, x, W, t)
        S_next ← transition_func(S, x, W, t)

        r ← -cost
        total_return ← total_return + r

        feat ← FEATURIZE(S, t, T)
        append (feat, pi, r) to traj
        S ← S_next

    # build return-to-go targets
    G ← 0
    for (feat, pi, r) in reversed(traj):
        G ← r + G
        buffer.append( feat, pi, (G / value_scale) )

    return total_return
```

### Supervised fit of the candidate network

The policy clones the current network, then does many SGD-like steps on random
minibatches from the buffer.

```text
cand ← clone(current_net)
repeat train_steps times:
    sample a minibatch of indices
    for each (feat, pi, z) in minibatch:
        cand.train_step(feat, target_pi=pi, target_v=z)
```

### Strict-CRN evaluation gate (accept/reject)

The accept/reject gate tries to prevent “learning noise” from overwriting a good
policy.

- Precompute `gate_episodes` CRN seed paths with
  `model.sample_crn_step_seeds(seed, T)`.
- Evaluate mean total cost under strict CRN using
  `model.simulate_crn(policy, S0, step_seeds, info=gate_info)`.
- Accept the candidate only if it strictly reduces mean cost.

Pseudocode:

```text
function EVALUATE_COST(net):
    pol ← new HybridAlphaZeroPolicy(...)
    pol.net ← net
    gate_info.deterministic ← True

    totals ← []
    for each CRN step_seed_path:
        (S, costs, X, info) ← simulate_crn(pol, S0, step_seed_path)
        totals.append(sum(costs))
    return mean(totals)

old ← EVALUATE_COST(current_net)
new ← EVALUATE_COST(candidate_net)
if new < old:
    current_net ← candidate_net
```

---

## Practical notes and “gotchas”

- **State bucketing in the tree**: the tree key uses a rounded inventory (and
  rounded regime, if present). This is a design choice to keep the tree small.
- **Action feasibility** is enforced in two places:
  - priors are masked and renormalized
  - visit counts are masked at the root before creating `π`
- **Depth limit `H`**: MCTS is not run all the way to the horizon; the value
  network is used as a leaf evaluator when `depth >= H`.
- **Reward vs cost**: backprop uses returns of `r = -cost`, so higher is better
  inside MCTS and training returns.
- **Value scaling**: training stores `G/value_scale` and the network predicts on
  the same scale. If you change cost magnitudes, you may want to adjust
  `value_scale`.
