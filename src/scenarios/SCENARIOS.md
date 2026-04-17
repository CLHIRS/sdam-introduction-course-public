# Scenarios (queueing)

This folder contains runnable **scenario modules** for the queueing lectures.
The intent is to provide a repeatable workflow:

- define a model/config,
- define baseline policies,
- evaluate fairly (often using strict CRN),
- optionally provide an RL-ready environment demo.

## Quick start

Run all scenarios via the orchestrator:

- `poetry run python -m scenarios.run_all`

Run just one scenario:

- `poetry run python -m scenarios.run_all --only "scenarios.scenario_03_dispatch_within_node"`
- `poetry run python -m scenarios.run_all --only "Scenario 03 — Dispatch within node (FIFO vs EDD)"`

Skip one scenario:

- `poetry run python -m scenarios.run_all --skip "Scenario 01 — M/M/1 basics"`

Notes:
- `--des-seed` controls the discrete-event simulation seed used by scenario entrypoints that accept it.
- `--crn-seed0` controls the base seed used by strict-CRN evaluation entrypoints.
- `--no-env` disables the optional RL-environment episode in Scenario 06.

## Scenario list

The canonical list lives in `scenarios/run_all.py` as `SCENARIOS`.

- **Scenario 01 — M/M/1 basics** ([scenario_01_mm1.py](scenario_01_mm1.py))
  - Minimal M/M/1-style network with no decision points.
  - Demonstrates strict-CRN evaluation harness usage even when policies are trivial.
  - Entrypoint: `run_demo()`.

- **Scenario 02 — Routing to parallel servers** ([scenario_02_parallel.py](scenario_02_parallel.py))
  - Dispatcher routes jobs to parallel servers.
  - Demonstrates routing decision policies (e.g., random masked routing, JSQ).
  - Entrypoint: `run_demo()`.

- **Scenario 03 — Dispatch within node (FIFO vs EDD)** ([scenario_03_dispatch_within_node.py](scenario_03_dispatch_within_node.py))
  - Illustrates dispatch policy choice *within* a node (e.g., FIFO vs EDD).
  - Provides paired entrypoints for DES demo and strict-CRN evaluation.
  - Entrypoints: `run_des_demo(seed=...)`, `run_crn_demo(seed0=...)`.

- **Scenario 04 — Rework loop / quality** ([scenario_04_rework_loop.py](scenario_04_rework_loop.py))
  - Adds quality outcomes and rework loops.
  - Provides paired entrypoints for DES demo and strict-CRN evaluation.
  - Entrypoints: `run_des_demo(seed=...)`, `run_crn_demo(seed0=...)`.

- **Scenario 05 — Fast vs clean + rework** ([scenario_05_fast_vs_clean_rework.py](scenario_05_fast_vs_clean_rework.py))
  - Contrasts different service options (e.g., fast vs clean) coupled with rework/quality effects.
  - Entrypoint: `run_demo()`.

- **Scenario 06 — Final PPO-ready manufacturing** ([scenario_06_final_ppo_ready_manufacturing.py](scenario_06_final_ppo_ready_manufacturing.py))
  - A richer manufacturing network intended to be “RL-ready”.
  - Demonstrates compiling a config, running a DES KPI report, and optionally stepping an RL env.
  - Entrypoints used by the runner: `build_cfg()`, `run_des_kpis(cfg)`, and optionally `run_env_one_episode(cfg, seed=...)`.

## Conventions for new scenarios

See [__init__.py](__init__.py) for the current conventions.

Practical guidance:

- Prefer exposing `run_demo()` for simple scenarios.
- For teaching strict-CRN comparisons, expose `run_crn_demo(seed0=...)`.
- If you have both a quick DES visualization and a strict-CRN evaluation, expose both `run_des_demo(seed=...)` and `run_crn_demo(seed0=...)`.
- If your scenario supports an RL environment, follow the Scenario 06 pattern (`build_cfg()` + `run_des_kpis(cfg)` + optional `run_env_one_episode`).
