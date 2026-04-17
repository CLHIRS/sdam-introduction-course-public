from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from inventory.core.dynamics import DynamicSystemMVP
from inventory.core.policy import Policy
from inventory.core.types import Action, State
from inventory.problems.demand_models import ExogenousPoissonRegime, ExogenousPoissonSeasonal


@dataclass
class DlaMctsUctPolicy(Policy):
    """Direct Lookahead Approximation via MCTS (UCT).

    Refactor of `DLA_MCTS_UCT` from `lectures/minimal_baseline.py`.

    Notes:
    - Uses the system model to simulate rollouts.
    - For strict-CRN reproducibility, this policy consumes `info['crn_step_seed']` when present.
    """

    model: DynamicSystemMVP
    horizon: int = 5
    n_simulations: int = 250
    uct_c: float = 1.4

    x_max: float = 480.0
    dx: int = 10

    planning_seed: int = 0

    def __post_init__(self) -> None:
        self.horizon = int(self.horizon)
        self.n_simulations = int(self.n_simulations)
        self.uct_c = float(self.uct_c)
        self.x_max = float(self.x_max)
        self.dx = int(self.dx)
        self.planning_seed = int(self.planning_seed)

        if self.horizon <= 0:
            raise ValueError("horizon must be >= 1")
        if self.n_simulations <= 0:
            raise ValueError("n_simulations must be >= 1")
        if self.dx <= 0:
            raise ValueError("dx must be positive")

    def _candidate_orders(self) -> List[float]:
        xs = np.arange(0, int(np.floor(self.x_max)) + 1, int(self.dx), dtype=int)
        return [float(int(x)) for x in xs]

    def _expert_mean_demand_one_step(self, state: State, t: int) -> float:
        exo = self.model.exogenous_model

        if isinstance(exo, ExogenousPoissonSeasonal):
            return float(exo.lambda_t(t))

        if isinstance(exo, ExogenousPoissonRegime):
            r_t = int(np.round(float(state[exo.regime_index])))
            r_t = max(0, min(r_t, len(exo.lam_by_regime) - 1))
            dist = np.zeros(len(exo.lam_by_regime), dtype=float)
            dist[r_t] = 1.0
            dist = dist @ exo.P
            return float(dist @ exo.lam_by_regime)

        return 0.0

    def _rollout_action(self, state: State, t: int) -> float:
        inv = float(state[0])
        mu = self._expert_mean_demand_one_step(state, t)
        x = max(0.0, mu - inv)
        x = min(x, self.x_max)
        if self.dx > 1:
            x = self.dx * np.round(x / self.dx)
        return float(int(np.round(max(0.0, x))))

    def _step_model(self, state: State, x: float, t: int, rng: np.random.Generator) -> Tuple[State, float]:
        action = np.array([float(int(np.round(x)))], dtype=float)
        exog = self.model.exogenous_model.sample(state, action, t, rng)
        cost = float(self.model.cost_func(state, action, exog, t))
        next_state = np.asarray(self.model.transition_func(state, action, exog, t), dtype=float)
        return next_state, cost

    class _Node:
        __slots__ = ("N", "W", "children", "untried")

        def __init__(self, actions: List[float]):
            self.N = 0
            self.W = 0.0
            self.children: Dict[float, "DlaMctsUctPolicy._Node"] = {}
            self.untried = list(actions)

        def uct_select(self, c: float) -> float:
            logN = np.log(max(1, self.N))
            best_a = None
            best_val = -float("inf")
            for a, ch in self.children.items():
                if ch.N <= 0:
                    val = float("inf")
                else:
                    q = ch.W / ch.N
                    val = q + c * np.sqrt(logN / ch.N)
                if val > best_val:
                    best_val = val
                    best_a = a
            return float(best_a)

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        actions = self._candidate_orders()
        root = self._Node(actions)

        step_seed = 0
        if info is not None and "crn_step_seed" in info:
            step_seed = int(info["crn_step_seed"])

        ss = np.random.SeedSequence([self.planning_seed, int(step_seed), int(t)])
        sim_seqs = ss.spawn(self.n_simulations)

        for sim_i in range(self.n_simulations):
            rng = np.random.default_rng(int(sim_seqs[sim_i].generate_state(1)[0]))

            node = root
            state_sim = np.asarray(state, dtype=float).copy()
            t_sim = int(t)
            depth = 0
            path: List[DlaMctsUctPolicy._Node] = [node]
            rewards: List[float] = []

            # Selection / expansion
            while depth < self.horizon:
                if node.untried:
                    a = float(node.untried.pop(0))
                    child = self._Node(actions)
                    node.children[a] = child

                    state_sim, c = self._step_model(state_sim, a, t_sim, rng)
                    rewards.append(-float(c))
                    node = child
                    path.append(node)
                    depth += 1
                    t_sim += 1
                    break

                if not node.children:
                    break

                a = node.uct_select(self.uct_c)
                state_sim, c = self._step_model(state_sim, a, t_sim, rng)
                rewards.append(-float(c))
                node = node.children[a]
                path.append(node)
                depth += 1
                t_sim += 1

            # Rollout
            while depth < self.horizon:
                a = self._rollout_action(state_sim, t_sim)
                state_sim, c = self._step_model(state_sim, a, t_sim, rng)
                rewards.append(-float(c))
                depth += 1
                t_sim += 1

            G = float(np.sum(rewards)) if rewards else 0.0

            # Backprop
            for nd in path:
                nd.N += 1
                nd.W += G

        if not root.children:
            return np.array([0.0], dtype=float)

        best_a = max(root.children.items(), key=lambda kv: kv[1].N)[0]
        best_a = float(int(np.round(best_a)))
        best_a = max(0.0, min(best_a, self.x_max))
        return np.array([best_a], dtype=float)


DLA_MCTS_UCT = DlaMctsUctPolicy


__all__ = ["DlaMctsUctPolicy", "DLA_MCTS_UCT"]
