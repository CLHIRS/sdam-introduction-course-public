# Use Cases (inventory)

This folder contains the canonical inventory experiment notebooks used across the lectures.
The goal is to give students a clear map of what exists, why it exists, and in which order it makes sense to read or run the material.

## How to read this folder

There are two parallel experiment tracks:

- `constant/`: inventory control under constant Poisson demand.
- `multi_regime/`: inventory control under observable multi-regime Poisson demand, where season/day/weather change over time.

Most experiments appear in both tracks with the same number so students can compare the same policy idea under two problem settings.

Naming convention:

- `EIC` = Experiment Inventory Constant
- `EIMR` = Experiment Inventory Multi-Regime
- Example: `EIC_080` and `EIMR_080` are the constant-demand and multi-regime versions of the PPO experiment.

## Recommended learning path

If you are new to the inventory experiments, read them in this order:

1. `010`: understand the baseline `OrderUpToPolicy`.
2. `020`: study the exact dynamic-programming benchmark.
3. `030` to `061`: move from simple parameterized policies to lookahead / planning methods.
4. `072`: inspect a hybrid rollout-style DLA policy.
5. `080` and `081`: move to PPO and forecast-augmented PPO.
6. `090`: finish with AlphaZero-style search + learning.

Pedagogically, the clean comparison pattern is:

1. Read the constant-demand version first.
2. Then open the multi-regime counterpart.
3. Ask what changed in the state, the information model, and the difficulty of the control problem.

## Quick comparison table

Use this table when you want a fast overview before opening individual notebooks.

| ID | Method family | Constant | Multi-regime | Main teaching point |
| --- | --- | --- | --- | --- |
| `000` | Template | [EIC_000](constant/EIC_000_template.ipynb) | [EIMR_000](multi_regime/EIMR_000_template.ipynb) | Standard experiment structure |
| `010` | Baseline | [EIC_010](constant/EIC_010_OrderUpToPolicy_as_baseline.ipynb) | [EIMR_010](multi_regime/EIMR_010_OrderUpToPolicy_as_baseline.ipynb) | Simple benchmark control |
| `020` | Exact DP | [EIC_020](constant/EIC_020_DynamicProgrammingPolicy_vs_baseline.ipynb) | [EIMR_020](multi_regime/EIMR_020_DynamicProgrammingPolicy_vs_baseline.ipynb) | Optimal benchmark for small problems |
| `030` | PFA | [EIC_030](constant/EIC_030_OrderUpToBlackboxPFA_vs_baseline.ipynb) | [EIMR_030](multi_regime/EIMR_030_OrderUpToBlackboxPFA_vs_baseline.ipynb) | Parameterized policy approximation |
| `040` | CFA / MILP | [EIC_040](constant/EIC_040_OrderMilpCfaPolicy_vs_baseline.ipynb) | [EIMR_040](multi_regime/EIMR_040_OrderMilpCfaPolicy_vs_baseline.ipynb) | Forecast-then-optimize |
| `050` | VFA | [EIC_050](constant/EIC_050_PostDecisionGreedyVfaPolicy_vs_baseline.ipynb) | [EIMR_050](multi_regime/EIMR_050_PostDecisionGreedyVfaPolicy_vs_baseline.ipynb) | Post-decision value approximation |
| `051` | FQI / VFA | [EIC_051](constant/EIC_051_FqiGreedyVfaPolicy_vs_baseline.ipynb) | [EIMR_051](multi_regime/EIMR_051_FqiGreedyVfaPolicy_vs_baseline.ipynb) | Offline fitted value learning |
| `060` | DLA / MPC | [EIC_060](constant/EIC_060_DlaMpcMilpPolicy_vs_baseline.ipynb) | [EIMR_060](multi_regime/EIMR_060_DlaMpcMilpPolicy_vs_baseline.ipynb) | Receding-horizon planning |
| `061` | DLA / MCTS | [EIC_061](constant/EIC_061_DlaMctsUctPolicy_vs_baseline.ipynb) | [EIMR_061](multi_regime/EIMR_061_DlaMctsUctPolicy_vs_baseline.ipynb) | Simulation-based tree search |
| `072` | Hybrid rollout | [EIC_072](constant/EIC_072_HybridDlaRolloutBasePfaPolicy_vs_baseline.ipynb) | [EIMR_072](multi_regime/EIMR_072_HybridDlaRolloutBasePfaPolicy_vs_baseline.ipynb) | Rollout over a base policy |
| `080` | RL / PPO | [EIC_080](constant/EIC_080_HybridPpoPolicy_vs_baseline.ipynb) | [EIMR_080](multi_regime/EIMR_080_HybridPpoPolicy_vs_baseline.ipynb) | Policy-gradient control |
| `081` | RL + forecasts | [EIC_081](constant/EIC_081_ForecastAugmentedHybridPpoPolicy_vs_baseline.ipynb) | [EIMR_081](multi_regime/EIMR_081_ForecastAugmentedHybridPpoPolicy_vs_baseline.ipynb) | Forecast-informed RL |
| `090` | Search + learning | [EIC_090](constant/EIC_090_HybridAlphaZeroPolicy_vs_baseline.ipynb) | [EIMR_090](multi_regime/EIMR_090_HybridAlphaZeroPolicy_vs_baseline.ipynb) | AlphaZero-style planning + learning |

## Canonical experiment list

The canonical experiment set currently consists of the following notebook pairs.

- `000` Template
  - [constant/EIC_000_template.ipynb](constant/EIC_000_template.ipynb)
  - [multi_regime/EIMR_000_template.ipynb](multi_regime/EIMR_000_template.ipynb)
  - Starting point for new lecture-quality experiments.
  - Shows the standard notebook structure: problem, candidate policy, strict-CRN evaluation, plots, and discussion.

- `010` Baseline: Order-up-to policy
  - [constant/EIC_010_OrderUpToPolicy_as_baseline.ipynb](constant/EIC_010_OrderUpToPolicy_as_baseline.ipynb)
  - [multi_regime/EIMR_010_OrderUpToPolicy_as_baseline.ipynb](multi_regime/EIMR_010_OrderUpToPolicy_as_baseline.ipynb)
  - Establishes the simplest benchmark policy.
  - Good first notebook for understanding the inventory state, action, cost, and evaluation workflow.

- `020` Exact benchmark: dynamic programming
  - [constant/EIC_020_DynamicProgrammingPolicy_vs_baseline.ipynb](constant/EIC_020_DynamicProgrammingPolicy_vs_baseline.ipynb)
  - [multi_regime/EIMR_020_DynamicProgrammingPolicy_vs_baseline.ipynb](multi_regime/EIMR_020_DynamicProgrammingPolicy_vs_baseline.ipynb)
  - Provides an exact or near-exact benchmark for small enough problems.
  - Helps students see what “optimal control” looks like before moving to approximations.

- `030` Parameterized policy approximation: black-box order-up-to
  - [constant/EIC_030_OrderUpToBlackboxPFA_vs_baseline.ipynb](constant/EIC_030_OrderUpToBlackboxPFA_vs_baseline.ipynb)
  - [multi_regime/EIMR_030_OrderUpToBlackboxPFA_vs_baseline.ipynb](multi_regime/EIMR_030_OrderUpToBlackboxPFA_vs_baseline.ipynb)
  - Introduces policy function approximation through learned or tuned order-up-to structure.
  - Useful bridge from handcrafted baselines to learned policies.

- `040` Cost function approximation / forecast-then-optimize: MILP CFA
  - [constant/EIC_040_OrderMilpCfaPolicy_vs_baseline.ipynb](constant/EIC_040_OrderMilpCfaPolicy_vs_baseline.ipynb)
  - [multi_regime/EIMR_040_OrderMilpCfaPolicy_vs_baseline.ipynb](multi_regime/EIMR_040_OrderMilpCfaPolicy_vs_baseline.ipynb)
  - Forecast future demand, then optimize replenishment with a mathematical program.
  - This is the clean representative of forecast-then-optimize thinking.

- `050` Value function approximation: post-decision greedy VFA
  - [constant/EIC_050_PostDecisionGreedyVfaPolicy_vs_baseline.ipynb](constant/EIC_050_PostDecisionGreedyVfaPolicy_vs_baseline.ipynb)
  - [multi_regime/EIMR_050_PostDecisionGreedyVfaPolicy_vs_baseline.ipynb](multi_regime/EIMR_050_PostDecisionGreedyVfaPolicy_vs_baseline.ipynb)
  - Introduces post-decision states and approximate value functions.
  - Useful for teaching why value approximation can reduce planning complexity.

- `051` Fitted value learning: FQI-greedy VFA
  - [constant/EIC_051_FqiGreedyVfaPolicy_vs_baseline.ipynb](constant/EIC_051_FqiGreedyVfaPolicy_vs_baseline.ipynb)
  - [multi_regime/EIMR_051_FqiGreedyVfaPolicy_vs_baseline.ipynb](multi_regime/EIMR_051_FqiGreedyVfaPolicy_vs_baseline.ipynb)
  - Extends the VFA story toward fitted Q- or value-iteration style learning.
  - Good notebook for discussing offline approximate dynamic programming.

- `060` Direct lookahead approximation: MPC / MILP
  - [constant/EIC_060_DlaMpcMilpPolicy_vs_baseline.ipynb](constant/EIC_060_DlaMpcMilpPolicy_vs_baseline.ipynb)
  - [multi_regime/EIMR_060_DlaMpcMilpPolicy_vs_baseline.ipynb](multi_regime/EIMR_060_DlaMpcMilpPolicy_vs_baseline.ipynb)
  - Represents receding-horizon planning with an explicit optimization model.
  - Good for contrasting exact dynamic programming vs finite-horizon lookahead.

- `061` Direct lookahead approximation: MCTS / UCT
  - [constant/EIC_061_DlaMctsUctPolicy_vs_baseline.ipynb](constant/EIC_061_DlaMctsUctPolicy_vs_baseline.ipynb)
  - [multi_regime/EIMR_061_DlaMctsUctPolicy_vs_baseline.ipynb](multi_regime/EIMR_061_DlaMctsUctPolicy_vs_baseline.ipynb)
  - Replaces deterministic optimization lookahead with simulation-based tree search.
  - Good for teaching exploration/exploitation in planning.

- `072` Hybrid direct lookahead: rollout over a base PFA
  - [constant/EIC_072_HybridDlaRolloutBasePfaPolicy_vs_baseline.ipynb](constant/EIC_072_HybridDlaRolloutBasePfaPolicy_vs_baseline.ipynb)
  - [multi_regime/EIMR_072_HybridDlaRolloutBasePfaPolicy_vs_baseline.ipynb](multi_regime/EIMR_072_HybridDlaRolloutBasePfaPolicy_vs_baseline.ipynb)
  - Combines a simple base policy with one-step or limited-horizon lookahead.
  - Good for showing how hybrid ADP policies are assembled from reusable parts.

- `080` Reinforcement learning: Hybrid PPO
  - [constant/EIC_080_HybridPpoPolicy_vs_baseline.ipynb](constant/EIC_080_HybridPpoPolicy_vs_baseline.ipynb)
  - [multi_regime/EIMR_080_HybridPpoPolicy_vs_baseline.ipynb](multi_regime/EIMR_080_HybridPpoPolicy_vs_baseline.ipynb)
  - Introduces policy-gradient RL with a masked action space.
  - Good for comparing RL to planning- and approximation-based methods.

- `081` Reinforcement learning with information augmentation: forecast-augmented PPO
  - [constant/EIC_081_ForecastAugmentedHybridPpoPolicy_vs_baseline.ipynb](constant/EIC_081_ForecastAugmentedHybridPpoPolicy_vs_baseline.ipynb)
  - [multi_regime/EIMR_081_ForecastAugmentedHybridPpoPolicy_vs_baseline.ipynb](multi_regime/EIMR_081_ForecastAugmentedHybridPpoPolicy_vs_baseline.ipynb)
  - Uses a forecaster as an observation augmenter rather than as the decision rule.
  - Good for discussing the difference between forecast-then-optimize and forecast-informed RL.

- `090` Search + learning: Hybrid AlphaZero
  - [constant/EIC_090_HybridAlphaZeroPolicy_vs_baseline.ipynb](constant/EIC_090_HybridAlphaZeroPolicy_vs_baseline.ipynb)
  - [multi_regime/EIMR_090_HybridAlphaZeroPolicy_vs_baseline.ipynb](multi_regime/EIMR_090_HybridAlphaZeroPolicy_vs_baseline.ipynb)
  - Combines neural value/policy approximation with MCTS-style search.
  - Useful capstone notebook for unifying planning and learning.

## What students should compare across paired notebooks

When comparing `EIC_xxx` with `EIMR_xxx`, focus on these questions:

1. How does the state change from constant demand to multi-regime demand?
2. Does the policy now need an information model or regime-aware features?
3. Does the baseline remain competitive, or does the richer environment reward more sophisticated policies?
4. Which methods degrade gracefully when the problem becomes more structured and less stationary?

## Common notebook structure

Each inventory notebook should be organized as:

1. Problem setup: define demand model, transition function, cost, and initial state.
2. Baseline policy: include at least one simple benchmark such as `OrderUpToPolicy`.
3. Candidate policy: define the main method being studied.
4. Controlled variation: if useful, add one targeted ablation or modeling variation.
5. Evaluation: strict-CRN comparison under aligned evaluation settings.
6. Discussion: plots, deltas, and short interpretation.

## Evaluation conventions

- Use deterministic evaluation by default:
  - `info = inventory.evaluation.make_eval_info(det_mode=..., T=T)`
- Strict CRN is handled by `DynamicSystemMVP` via `info["crn_step_seed"]`.
- Learning policies may also use `crn_step_seed` for reproducible decision sampling.
- When a learning policy was trained with a specific deterministic evaluation rule, the final CRN evaluation should use the same rule.

## Validation runner

Validate that the inventory notebooks are valid JSON notebooks and that the `inventory` module imports cleanly:

- `poetry run python experiments/inventory/run_all.py`
