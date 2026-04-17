# Canonical inventory lecture notebooks (source of truth)

This package (`src/inventory/`) is refactored from the **latest** lecture notebooks under `lectures/`.
When there are multiple versions, we treat the newest `*_b_*` notebook as canonical.
Older `*_a_*` notebooks remain as historical references and should not be used as extraction sources.

## Canonical sources

### Core foundations (used by many later lectures)

- Dynamic System MVP + strict-CRN API: `lectures/lecture_01_d_refactor_DynSys_V0.1.ipynb`
  - Implemented in `inventory.core.dynamics.DynamicSystemMVP` (with backwards-compatible constructor aliases `dS/dX/dW`).

- Transition + objective function patterns (inventory examples, capacity-capped variants): `lectures/lecture_03_b_V0.2.ipynb`
  - Implemented in `inventory.problems.inventory_mvp` (`inventory_transition`, `inventory_cost`, and optional `s_max` capped transitions).

- Lecture 11 (PFAs): `lectures/lecture_11_b_V0.1.ipynb`
  - Extracted to `inventory.policies.pfa`:
    - `OrderUpToBlackboxPFA`
    - `OrderUpToStateDependentPFA`
    - `OrderUpToRegimeTablePFA`

- Lecture 12 (CFAs): `lectures/lecture_12_b_V0.1.ipynb`
  - Extracted to:
    - `inventory.policies.cfa_milp`: `OrderMilpCfaPolicy` (alias `Order_MILP_CFA`)
    - `inventory.forecasters.ml`: `SeasonalFeatureAdapter`, `MlDemandForecaster`
    - `inventory.forecasters.naive`: `ExpertDemandForecasterConstant350`

- Lecture 13 (VFAs / FQI): `lectures/lecture_13_b_V0.1.ipynb`
  - Extracted to `inventory.policies.vfa`:
    - `PostDecisionGreedyVfaPolicy` (alias `PostDecision_Greedy_VFA`)
    - `FqiGreedyVfaPolicy` (alias `FQI_Greedy_VFA`)

- Lecture 14 (DLAs: MILP + MCTS): `lectures/lecture_14_b_V0.1.ipynb`
  - Extracted to:
    - `inventory.policies.dla_milp`: `DlaMpcMilpPolicy` (alias `DLA_MPC_MILP`)
    - `inventory.policies.dla_mcts`: `DlaMctsUctPolicy` (alias `DLA_MCTS_UCT`)
    - ML forecaster utilities live in `inventory.forecasters.ml` (shared with Lecture 12)

- Lecture 15 (hybrids): `lectures/lecture_15_b_V0.1.ipynb`
  - Extracted to:
    - `inventory.policies.hybrids`:
      - `HybridDlaCfaBufferedForecastPolicy` (alias `Hybrid_DLA_CFA_BufferedForecast`)
      - `HybridDlaVfaTerminalPolicy` (alias `Hybrid_DLA_VFA_Terminal`)
      - `HybridDlaRolloutBasePfaPolicy` (alias `Hybrid_DLA_Rollout_BasePFA`)
      - `TerminalVfaLinear` (alias `TerminalVFA_Linear`)
    - `inventory.forecasters.path`:
      - `ExogenousMeanPathForecaster`, `SeasonalSinMeanPathForecaster`, `ConstantMeanPathForecaster`

- Lecture 16 (RL / PPO hybrid): `lectures/lecture_16_b_V0.1.ipynb`
  - Extracted to `inventory.policies.ppo`:
    - `HybridPpoPolicy` (alias `Hybrid_PPO`)
    - `PPOHyperParams`
    - `train_ppo_with_eval_gate`
  - Note: This notebook contains multiple PPO class variants; the package matches the in-notebook `# --- LASTEST ---` version, including deterministic evaluation modes via `info={"deterministic": True, "det_mode": "mean"|"argmax", "risk_alpha": ...}`.

- Lecture 17 (AlphaZero hybrid): `lectures/lecture_17_b_V0.1.ipynb`
  - Extracted to `inventory.policies.alphazero`:
    - `HybridAlphaZeroPolicy` (alias `Hybrid_AlphaZero`).
  - Note: This notebook contains an "Old Class" and an "Updated Class"; the package matches the "Updated Class" implementation.

## Smoke tests

Run a quick import+behavior smoke suite for all canonical extractions:

- `poetry run python -m inventory.evaluation.smoke_canonical`

## Evaluation-mode conventions (standardized)

To keep comparisons fair and reproducible across *learning* policies (PPO, AlphaZero, etc.),
we use a small `info` contract passed into `Policy.act(state, t, info)`.

- Deterministic evaluation by default in CRN MC evaluators:
  - `DynamicSystemMVP.evaluate_policy_crn_mc(...)` and `evaluate_policies_crn_mc(...)` inject
    `info["deterministic"] = True` unless you explicitly set it.
  - Training code should pass `info={"deterministic": False, ...}` if it intentionally wants stochastic actions.

- Strict CRN applies to *environment sampling*:
  - Each step injects `info["crn_step_seed"]` and uses it to seed the exogenous RNG.

- Optional CRN for *decision sampling* (only for stochastic policies):
  - If a policy samples actions, it may use `info["crn_step_seed"]` to seed its own sampling so
    decisions are also reproducible under CRN.

Common keys:

- `deterministic: bool` (evaluation=True, training=False)
- `det_mode: str` (e.g. PPO: `"mean"|"argmax"`; AlphaZero: `"argmax"` or `"sample"`)
- `risk_alpha: float` (optional PPO risk-aware argmax)
- `crn_step_seed: int` (injected per-step by the system)
- `last_demand: float` (injected from the previous realized demand; absent at `t=0`)

