# `policy_automl.py`

## Purpose

`policy_automl.py` is the shared helper layer behind the inventory AutoML notebooks.

Its job is to let a notebook such as `EIC_120_SMAC3_policy_selection.ipynb` express a
student-visible policy menu, hand that menu to SMAC3, and receive back:

- a best-found policy configuration,
- the corresponding runnable policy object,
- and a compact runhistory summary.

The notebook then keeps the familiar SDAM evaluation workflow:

1. define or search for policies,
2. put the resulting policies into a `policies` dictionary,
3. run the usual strict-CRN evaluation via `run_crn_mc(...)`.

So this module is not a replacement for the evaluation workflow. It is an upstream
policy-search layer that feeds into the existing workflow.

---

## Big Picture

The architecture is easiest to read as a seven-step pipeline:

1. **Problem bundling**
   - Collect the dynamic system, initial state, horizon, action grid, costs, and
     optional registries into one `PolicyAutoMLDemoProblem`.

2. **Notebook menu translation**
   - Translate a notebook-facing `policy_menu` into helper-layer SMAC arguments.

3. **ConfigSpace construction**
   - Build a conditional SMAC/ConfigSpace search space from the requested policy
     families and their active hyperparameters.

4. **Config translation**
   - Turn one raw SMAC configuration into one repo-native policy config dict.

5. **Policy materialization**
   - Convert that policy config into a runnable policy object.
   - Depending on the family, this may mean:
     - direct instantiation,
     - training inside the search loop,
     - lookup of a pre-built artifact,
     - or assembly of a composite policy from components.

6. **Strict-CRN objective evaluation**
   - Evaluate the materialized policy on a fixed seed bundle so SMAC compares
     candidates fairly.

7. **SMAC orchestration**
   - Run SMAC3 over that objective and return the incumbent, its score, its policy
     object, and a notebook-friendly runhistory summary.

---

## Core Data Structures

### `PolicyAutoMLDemoProblem`

This dataclass is the central problem bundle.

It stores:

- the simulation model and exogenous model,
- `S0`, `T`, `x_max`, `dx`, `s_max`,
- cost parameters,
- `eval_info` for search-time or hold-out policy evaluation,
- optional `train_info` for train-in-loop families,
- optional `forecaster_registry`,
- optional `prefit_policy_registry`,
- optional `component_policy_registry`.

The important design idea is:

> notebooks define experiment-specific objects, while `policy_automl.py` consumes
> them through one structured bundle.

That keeps the helper reusable across both `EIC` and `EIMR` style notebooks.

### `PolicyAutoMLSearchResult`

This is the result of evaluating one candidate policy under strict CRN:

- label,
- family,
- config,
- mean total cost,
- standard deviation,
- runtime,
- objective score,
- raw totals.

### `PolicyAutoMLSmacResult`

This is the result of a full SMAC run:

- incumbent config,
- incumbent label,
- incumbent score,
- incumbent policy,
- runhistory rows,
- trial counts,
- scenario seed.

The `incumbent_policy` field matters especially for train-in-loop families, because
the notebook can reuse the winning policy directly instead of rebuilding and
retraining it after the search.

---

## Four Search Object Types

The current helper supports four conceptually different kinds of searchable objects.

### 1. Tunable families

These are families that can be instantiated directly from a low-dimensional config.

Examples:

- `order_up_to`
- `pfa_blackbox`
- `cfa_milp`
- `dla_mpc`
- `dla_mcts`

These are the simplest case:

> config -> constructor -> policy

### 2. Trainable families

These are families that must be instantiated and then trained inside the search loop.

Current example:

- `direct_action_pfa`

This is the phase-2A case:

> config -> instantiate training recipe -> fit policy -> evaluate policy

So here SMAC is not only selecting a policy family. It is selecting a small
learning pipeline that produces a policy.

### 3. Composite families

These are policies assembled from existing policy objects playing different roles.

Current example:

- `hybrid_rollout_choice`

This is the phase-2B-lite case:

> config -> choose component-policy roles -> assemble composite policy -> evaluate

In the current implementation, `hybrid_rollout_choice` lets SMAC choose:

- `rollout_policy_key`
- `candidate_policy_key`

while keeping the expensive hybrid planning knobs fixed in the notebook menu.

### 4. Prefit variants

These are already-built policy artifacts selected categorically by name.

Current helper family:

- `prefit_policy`

This means:

> config -> registry lookup -> evaluate existing object

No retraining happens here. This is how notebooks can expose artifacts such as a
pre-trained `DirectActionPFA_Tree` or a pre-built PPO policy.

---

## Notebook-Facing Interface

The preferred notebook interface is `policy_menu`.

Its teaching-facing structure is:

```python
{
    "tunable_families": {...},
    "trainable_families": {...},
    "composite_families": {...},
    "prefit_variants": {...},
}
```

This separation is deliberate because it makes the AutoML search space readable for
students:

- **tunable**: direct hyperparameter search,
- **trainable**: learning inside the search loop,
- **composite**: search over policy roles inside a larger architecture,
- **prefit**: choose among already-built artifacts.

`smac_policy_menu_to_search_space_config(...)` translates that visible notebook menu
into the lower-level kwargs expected by the SMAC helper.

So the notebook stays readable, while `policy_automl.py` keeps the reusable
mechanics.

---

## Search-Time Mechanics

### ConfigSpace layer

`make_policy_automl_smac3_configspace(...)` builds a conditional SMAC search space.

Important properties:

- one categorical `family` variable selects the active branch,
- family-specific hyperparameters are activated only when needed,
- train-in-loop branches can have additional conditional parameters,
- composite branches can expose component-policy choices,
- prefit branches expose artifact names as categorical values.

This is why the module can support a heterogeneous search space in one SMAC run.

### Config translation layer

`smac3_config_to_policy_config(...)` translates one raw SMAC configuration into a
simple repo-native config dictionary.

This is the key normalization step.

Examples:

- `{"family": "order_up_to", ...}` -> `{"label": "OrderUpTo[S=...]", ...}`
- `{"family": "direct_action_pfa", ...}` -> `{"label": "DirectActionPFA[...]", ...}`
- `{"family": "hybrid_rollout_choice", ...}` -> `{"label": "HybridRollout[...]", ...}`

That normalized config is what all downstream steps use.

---

## Policy Materialization Layer

There are two related constructor helpers:

### `build_policy_from_config(...)`

This is the basic family-to-policy constructor interface.

It handles:

- direct family instantiation,
- prefit registry lookup,
- and family-specific special cases.

### `materialize_policy_from_config(...)`

This is the more important function for real AutoML runs.

It adds:

- `train_info` handling,
- `training_seed0`,
- config-based caching,
- train-in-loop policy creation,
- composite policy assembly.

So the best mental model is:

> `build_policy_from_config(...)` is the basic constructor layer, while
> `materialize_policy_from_config(...)` is the search-time orchestration layer.

For repeated SMAC evaluation, notebooks and helper code should conceptually rely on
materialization, not on raw family constructors.

---

## Search Objective and Fairness

`evaluate_policy_config(...)` is the core objective-evaluation function.

It does:

1. materialize the policy,
2. evaluate it with `system.evaluate_policy_crn_mc(...)`,
3. compute mean cost, standard deviation, runtime, and score.

The default score is:

```text
score = mean_total_cost + runtime_penalty * runtime_sec
```

So with `runtime_penalty = 0.0`, SMAC simply minimizes strict-CRN mean total cost.

`make_policy_automl_smac3_target_function(...)` wraps that into the callable that
SMAC sees. The target is deterministic on purpose:

> every candidate is evaluated on the same strict-CRN seed bundle.

This is the main fairness guarantee of the helper.

---

## SMAC Orchestration Layer

`run_policy_automl_smac3(...)` is the top-level entry point that notebooks call.

It is responsible for:

- resolving the notebook menu,
- building the ConfigSpace,
- creating the deterministic target function,
- running SMAC3,
- decoding the incumbent,
- materializing the incumbent policy,
- returning notebook-friendly runhistory rows.

This is the function that lets notebooks stay short while still supporting mixed
search spaces across tunable, trainable, composite, and prefit branches.

---

## Current Extension Pattern

To add a new searchable family, the current architecture expects changes in four
places:

1. **menu translation**
   - add the family to `smac_policy_menu_to_search_space_config(...)`

2. **family resolution and ConfigSpace**
   - validate the family in `_resolve_smac_families(...)`
   - add its parameters in `make_policy_automl_smac3_configspace(...)`

3. **config translation**
   - add a new branch in `smac3_config_to_policy_config(...)`

4. **materialization**
   - add the construction or training logic in
     `build_policy_from_config(...)` or `materialize_policy_from_config(...)`

In practice, the new family usually also determines whether the notebook needs:

- a `forecaster_registry`,
- a `prefit_policy_registry`,
- a `component_policy_registry`,
- or a `train_info` block.

---

## Relation to Benchmarks

`build_exact_dp_benchmark(...)` is intentionally kept outside the SMAC search logic.

Its role is:

- build a strong exact-DP benchmark for supported demo problems,
- let notebooks compare the AutoML incumbent against a principled baseline.

So the benchmark is part of the teaching workflow, but not part of the search-space
machinery.

---

## Practical Teaching Interpretation

A good one-sentence summary for students is:

> `policy_automl.py` separates policy AutoML into four reusable layers:
> problem bundling, search-space construction, policy materialization, and strict-CRN objective evaluation.

For the current `EIC_120` notebook, that architecture now supports:

- direct tuning of simple policy families,
- train-in-loop search for `DirectActionPFA`,
- searchable composition for `HybridDlaRolloutComposablePolicy`,
- and final evaluation through the familiar `run_crn_mc(...)` workflow.

---

## Current Limits

The current helper is deliberately didactic, not fully general.

Important limitations:

- it is designed around small teaching-oriented search spaces,
- expensive inner-loop search should be introduced carefully,
- train-in-loop families still need runtime control through modest budgets,
- full composite-family tuning is intentionally not yet exposed in one step,
- benchmark claims should still be validated in the notebook’s final hold-out run.

So the right teaching message is:

> this helper demonstrates the architecture of policy AutoML in SDAM, while still
> keeping the search space and runtime manageable enough for course notebooks.
