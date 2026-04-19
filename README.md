# 👨‍🏫 Sequential Decision Analytics and Modeling (SDAM)

Python framework and lecture materials for the course **“Sequential Decision Analytics and Modeling”**.

This repository accompanies a graduate‑level lecture introducing how to **model, simulate, and optimize sequential decision problems under uncertainty**. Students progressively build a modular Python framework for dynamic systems, decision policies, and stochastic simulation. Thereby they gain algorithmic literacy by learning how theoretical concepts can be applied to solve real world problems.

The course follows the unified framework proposed in:

> [Powell, W. – *Sequential Decision Analytics and Modeling: Modeling with Python* (2026)](https://castle.princeton.edu/wp-content/uploads/2026/02/Powell-Kindle-SDAM-2nd-ed-Feb-9-2026-w_cover.pdf)

Please find an extensive list of resources regarding Sequential Decision Analytics and Modeling provided by Powell here:

> [Sequential Decision Analytics and Modeling](https://castle.princeton.edu/sdalinks/)

---

# 🎯 Learning Objectives

After completing this course, students will be able to:

- model sequential decision problems as dynamic systems,
- design and evaluate decision policies,
- simulate stochastic systems using Monte Carlo methods,
- apply approximate dynamic programming methods, and
- connect RL methods to classical stochastic optimization.

---

# 🧭 Repository Navigation

If you are new to the repository, start here:

| Goal | Location |
|-----|-----|
| Verify that your environment works | `lectures/00_setup.ipynb` |
| Learn the theoretical concepts | `lectures/` |
| Run example experiments | `experiments/` |
| Explore the reusable Python framework | `src/` |
| Access figures and supporting resources | `rsc/` |
| Inspect automated tests | `tests/` |

Typical workflow in this course:

```
Lecture concept → Python implementation → Experiment notebook
```

---

# 🏗️ Sequential Decision Framework

The course is built around Powell’s unified modeling structure for sequential decision problems.

```
                                          Exogenous information
                                                W_{t+1}
                                                    │
                                                    ▼
State S_t ──► Policy π(S_t) ──► Decision X_t ──► Dynamics S^M(S_t, X_t, W_{t+1}) ──► Next State S_{t+1} 
                                                    │
                                                    ▼
                                                Contribution
                                            C(S_t, X_t, W_{t+1})
```

Mathematically, the system evolves according to

```
S_{t+1} = S^M(S_t, X_t, W_{t+1}), with X_t=π(S_t)
```

where

- `S_t` = system state 
- `π(S_t)`= policy 
- `X_t` = decision/action  
- `W_{t+1}` = exogenous information  
- `S^M` = system transition model  
- `C` = contribution to ojective 

and the objective 

$$\max_{\pi}\ \mathbb{E}\left\[ \sum_{t=0}^{T} C\big(S_t, X^{\pi}(S_t), W_{t+1}\big) \right\] $$

---

# 🗺️ Repository Structure

The repository is organized into the following main components:

```
repo(sdam-introduction-course)/
│
├── lectures/
│   Lecture notebooks introducing theory and concepts
│
├── experiments/
│   Reproducible experiments and case studies
│   (inventory models, queueing systems, policy comparisons)
│
├── src/
│   Core Python framework used throughout the course
│   (dynamic systems, policies, simulators)
│
├── rsc/
│   Supporting resources such as figures and datasets
│
├── tests/
│   Unit tests verifying correctness of the framework
│
├── pyproject.toml
│   Project configuration and dependency specification
│
└── README.md
```

---

# 🔬 Lecture Overview

The course follows a notebook-centered format: each lecture introduces one core modeling or policy idea, connects it to the Python implementation, and then asks students to investigate the idea empirically through guided experiments.

| Lecture | Topic | Class Ref. | Experiment Ref. |
|---:|---|---|---|
|  | **Block A** | **Build the World** |  |
| 1 | Introduction to Sequential Decision Problems |  |  |
| 2 | Modeling Dynamic Systems and Policy Evaluation | [`DynamicSystem`](src/inventory/core/dynamics.py) | `EIC_000` `EIMR_000` |
|  | **Block B** | **Solve the World with Policy Classes**  |  |
| 3 | Heuristics and Dynamic Programming | [`OrderUpTo`](src/inventory/policies/baselines.py) · [`OrderRandom`](src/inventory/policies/random.py) · [`DynamicProgrammingSolver`](src/inventory/policies/dp.py) |`EIC_010` `EIMR_010` `EIC_020` `EIMR_020` |
| 4 | Policy Function Approximations (PFA) | [`OrderUpToPFA`](src/inventory/policies/pfa.py) |`EIC_030` `EIMR_030` |
| 5 | Cost Function Approximations (CFA) and Forecasting | [`OrderMilpCFA`](src/inventory/policies/cfa_milp.py) · [`Forecasters`](src/inventory/forecasters/) | `EIC_040` `EIMR_040` |
| 6 | Value Function Approximations (VFA) | [`GreedyVFA`](src/inventory/policies/vfa.py) · [`FqiVFA`](src/inventory/policies/vfa.py)  | `EIC_050` `EIMR_050` `EIC_051` `EIMR_051` |
| 7 | Direct Lookahead Approximations (DLA) | [`MilpDLA`](src/inventory/policies/dla_milp.py) · [`MctsDLA`](src/inventory/policies/dla_mcts.py) | `EIC_060` `EIMR_060` `EIC_061` `EIMR_061`  |
|  | **Block C** | **Modern Synthesis and Transfer**  |  |
| 8 & 9 | Hybrid Policies and Reinforcement Learning | [`HybridDLA`](src/inventory/policies/hybrids.py) · [`HybridPPO`](src/inventory/policies/ppo.py) · [`HybridAlphaZero`](src/inventory/policies/alphazero.py) | `EIC_072` `EIC_080` `EIC_081` `EIC_090` `EIMR_072` `EIMR_080` `EIMR_081` `EIMR_090` |
| 10 | Queueing Systems and Networks | [`QueueingSystem`](src/queueing/sim.py) · [`RoutingPolicy`](src/queueing/)| `Scenario_1` |
|  | **Optional Closing Session** |  |  |
| 11* | Recap, Exam Guidance, and Open Questions |  | |

\* Optional closing session for recap, exam guidance, and open questions.

For a compact index of all inventory notebooks, see [experiments/inventory/EXPERIMENTS.md](experiments/inventory/EXPERIMENTS.md). Detailed explanations are also provided inside the lecture notebooks.

---

# 🚀 Quick Start

Before opening any lecture notebook, verify that your environment works.

1. Install dependencies

```
poetry install
```

2. Start Jupyter

```
poetry run jupyter lab
```

3. Run the setup notebook

```
lectures/00_setup.ipynb
```

This notebook verifies that:

- the correct Python environment is active
- all imports work
- stochastic simulations are reproducible.

---

# 📦 Installation

### Prerequisites

- Git
- Python **>= 3.12 and < 3.15**
- Poetry (dependency manager)

---

### Clone the repository

```
git clone <REPO_URL>
cd sdam_introduction
```

---

### Install Poetry

Recommended installation using `pipx`:

```
python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install poetry
```

Verify:

```
poetry --version
```

---

### Install dependencies

From the repository root:

```
poetry install
```

If Poetry selects the wrong Python version:

```
poetry env use python3.12
poetry install
```

---

# 🚴‍♀️ Running Experiments

Start Jupyter:

```
poetry run jupyter lab
```

Example experiment:

```
experiments/inventory/constant/uc_c_010_OrderUpToPolicy_as_baseline.ipynb
```

These experiments demonstrate how different policies behave under stochastic demand.

# 🐍 Run notebooks in VS Code (alternative)

1. Open the repo folder in VS Code.
2. Open any `.ipynb` under `experiments/inventory/...`.
3. Select the Python kernel that belongs to the Poetry environment.

If VS Code cannot find the Poetry kernel, register it once:

```bash
poetry run python -m ipykernel install --user --name sdam-introduction --display-name SDAM-Poetry
```

Then select the kernel named `SDAM-Poetry`.

---

# 🪄 Reproducibility Guidelines

When comparing policies:

- keep `seed0` fixed
- keep `n_episodes` fixed
- use strict CRN evaluation utilities

Example:

```
DynamicSystemMVP.evaluate_policies_crn_mc()
```

---

# 📚 Recommended Literature

Core reference:

- Powell W. – *Sequential Decision Analytics and Modeling* (2022)

Additional references:

- Sutton & Barto – *Reinforcement Learning: An Introduction*
- Puterman – *Markov Decision Processes*
- Bertsekas – *Dynamic Programming and Optimal Control*
- Kleinrock – *Queueing Systems*
- Chen & Yao – *Fundamentals of Queueing Networks*
- Brunton & Kutz – *Data‑Driven Science and Engineering*

---

# 📋 License

This repository is distributed under the terms of the license provided in `LICENSE`.
