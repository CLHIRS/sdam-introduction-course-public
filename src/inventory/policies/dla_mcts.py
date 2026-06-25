from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from inventory.core.dynamics import DynamicSystemMVP
from inventory.core.policy import Policy
from inventory.core.types import Action, State
from inventory.problems.demand_models import (
    ExogenousPoissonConstant,
    ExogenousPoissonMultiRegime,
    ExogenousPoissonRegime,
    ExogenousPoissonSeasonal,
)


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

    def __repr__(self) -> str:
        return (
            "DlaMctsUctPolicy("
            f"horizon={self.horizon}, n_simulations={self.n_simulations}, "
            f"uct_c={self.uct_c}, x_max={self.x_max}, dx={self.dx}, "
            f"planning_seed={self.planning_seed})"
        )

    def _candidate_orders(self) -> List[float]:
        # Builds the discrete action set  X = {0, dx, 2*dx, ..., x_max}.
        xs = np.arange(0, int(np.floor(self.x_max)) + 1, int(self.dx), dtype=int)
        return [float(int(x)) for x in xs]

    def _expert_mean_demand_one_step(self, state: State, t: int) -> float:
        exo = self.model.exogenous_model

        if isinstance(exo, ExogenousPoissonConstant):
            return float(exo.lambda_t(t))

        if isinstance(exo, ExogenousPoissonSeasonal):
            return float(exo.lambda_t(t))

        if isinstance(exo, ExogenousPoissonRegime):
            r_t = int(np.round(float(state[exo.regime_index])))
            r_t = max(0, min(r_t, len(exo.lam_by_regime) - 1))
            dist = np.zeros(len(exo.lam_by_regime), dtype=float)
            dist[r_t] = 1.0
            dist = dist @ exo.P
            return float(dist @ exo.lam_by_regime)
        
        if isinstance(exo, ExogenousPoissonMultiRegime):
            S = np.asarray(state, dtype=float).reshape(-1)

            season = int(np.round(float(S[exo.season_index])))
            day = int(np.round(float(S[exo.day_index])))
            weather = int(np.round(float(S[exo.weather_index])))

            season = int(np.clip(season, 0, exo.P_season.shape[0] - 1))
            day = int(np.clip(day, 0, exo.P_day.shape[0] - 1))
            weather = int(np.clip(weather, 0, exo.P_weather.shape[0] - 1))

            pi_season = np.zeros(exo.P_season.shape[0], dtype=float)
            pi_day = np.zeros(exo.P_day.shape[0], dtype=float)
            pi_weather = np.zeros(exo.P_weather.shape[0], dtype=float)
            pi_season[season] = 1.0
            pi_day[day] = 1.0
            pi_weather[weather] = 1.0

            if int(getattr(exo, "season_period", 0)) <= 0 or ((int(t) + 1) % int(exo.season_period) == 0):
                pi_season = pi_season @ exo.P_season

            pi_day = pi_day @ exo.P_day
            pi_weather = pi_weather @ exo.P_weather

            base = float(pi_season @ exo.lambda_base_season)
            coeff_day = float(pi_day @ exo.lambda_coeff_day)
            coeff_weather = float(pi_weather @ exo.lambda_coeff_weather)
            lam = base * (1.0 + coeff_day + coeff_weather)
            return float(max(getattr(exo, "lam_min", 0.0), lam))

        return 0.0

    def _rollout_action(self, state: State, t: int) -> float:
        # Rollout heuristic: order up to expected demand.
        #   x = max(0, mu_t - I_t),  then clip to x_max and round to nearest dx.
        inv = float(state[0])
        mu = self._expert_mean_demand_one_step(state, t)
        x = max(0.0, mu - inv)   # order-up-to formula
        return self._sanitize_action_scalar(x)

    def _sanitize_action_scalar(self, action: Action | float) -> float:
        arr = np.asarray(action, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError("action must contain at least one scalar")

        x = float(arr[0])
        x = max(0.0, min(x, self.x_max))
        if self.dx > 1:
            x = self.dx * np.round(x / self.dx)
        x = max(0.0, min(float(x), self.x_max))
        return float(int(np.round(x)))

    def _step_model_full(
        self,
        state: State,
        x: float,
        t: int,
        rng: np.random.Generator,
    ) -> Tuple[State, float, np.ndarray]:
        action = np.array([float(int(np.round(x)))], dtype=float)
        exog = self.model.exogenous_model.sample(state, action, t, rng)
        cost = float(self.model.cost_func(state, action, exog, t))
        next_state = np.asarray(self.model.transition_func(state, action, exog, t), dtype=float)
        return next_state, cost, np.asarray(exog, dtype=float)

    def _step_model(self, state: State, x: float, t: int, rng: np.random.Generator) -> Tuple[State, float]:
        next_state, cost, _exog = self._step_model_full(state, x, t, rng)
        return next_state, cost

    def _rollout_history_window(self) -> int:
        override = int(getattr(self, "rollout_history_window", 0))
        if override > 0:
            return override
        return int(getattr(self.model, "demand_history_window", 0))

    def _init_rollout_info(self, info: Optional[dict]) -> dict:
        rollout_info = {} if info is None else dict(info)
        hist = np.asarray(rollout_info.get("demand_history", []), dtype=float).reshape(-1)
        hist_window = self._rollout_history_window()
        if hist_window > 0 and hist.size > hist_window:
            hist = hist[-hist_window:]
        if hist.size > 0:
            rollout_info["demand_history"] = hist.astype(float, copy=False)
        else:
            rollout_info.pop("demand_history", None)
        return rollout_info

    def _update_rollout_info(
        self,
        rollout_info: dict,
        exog: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        exog_arr = np.asarray(exog, dtype=float).reshape(-1)
        if exog_arr.size >= 1:
            demand = float(exog_arr[0])
            rollout_info["last_demand"] = demand

            hist = np.asarray(rollout_info.get("demand_history", []), dtype=float).reshape(-1)
            hist_window = self._rollout_history_window()
            if hist.size > 0 or hist_window > 0:
                hist = np.concatenate([hist, np.array([demand], dtype=float)])
                if hist_window > 0 and hist.size > hist_window:
                    hist = hist[-hist_window:]
                if hist.size > 0:
                    rollout_info["demand_history"] = hist

        rollout_info["crn_step_seed"] = int(
            rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)
        )

    def _select_rollout_action(self, state: State, t: int, info: Optional[dict] = None) -> float:
        return self._rollout_action(state, t)

    class _Node:
        __slots__ = ("N", "W", "children", "untried")

        def __init__(self, actions: List[float]):
            self.N = 0
            self.W = 0.0
            self.children: Dict[float, "DlaMctsUctPolicy._Node"] = {}
            self.untried = list(actions)

        def uct_select(self, c: float) -> float:
            # UCT selection rule:
            #   UCT(a) = Q_bar(a) + c * sqrt( log(N) / N(a) )
            # where Q_bar(a) = W(a) / N(a)  is the empirical mean return.
            # Unvisited children (N(a) = 0) are scored +inf so they are tried first.
            logN = np.log(max(1, self.N))   # log N(parent)
            best_a = None
            best_val = -float("inf")
            for a, ch in self.children.items():
                if ch.N <= 0:
                    val = float("inf")   # force exploration of unvisited children
                else:
                    q = ch.W / ch.N          # Q_bar(a) = W(a) / N(a)
                    val = q + c * np.sqrt(logN / ch.N)   # UCT score
                if val > best_val:
                    best_val = val
                    best_a = a
            return float(best_a)

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        # Objective: approximate  a* = argmin_{a in X} E[ sum_{k=0}^{H-1} C(S_{t+k}, A_{t+k}, W_{t+k+1}) | S_t, A_t=a ]
        # Internally rewards are stored as R = -C, so the search maximises G = sum_{k} (-C_k).
        actions = self._candidate_orders()   # X = {0, dx, 2*dx, ..., x_max}
        root = self._Node(actions)

        # CRN seed mixing: combine planning_seed, crn_step_seed (if given), and t
        # so that each decision epoch gets independent but reproducible random streams.
        step_seed = 0
        if info is not None and "crn_step_seed" in info:
            step_seed = int(info["crn_step_seed"])

        ss = np.random.SeedSequence([self.planning_seed, int(step_seed), int(t)])
        sim_seqs = ss.spawn(self.n_simulations)   # one independent RNG per simulation

        for sim_i in range(self.n_simulations):
            rng = np.random.default_rng(int(sim_seqs[sim_i].generate_state(1)[0]))

            node = root
            state_sim = np.asarray(state, dtype=float).copy()
            rollout_info = self._init_rollout_info(info)
            t_sim = int(t)
            depth = 0
            path: List[DlaMctsUctPolicy._Node] = [node]
            rewards: List[float] = []

            # Phase 1 — Selection / Expansion
            # Descend the tree until depth H, choosing actions by UCT or expanding a new child.
            while depth < self.horizon:
                if node.untried:
                    # Expansion: pick the first untried action and add a new node.
                    a = float(node.untried.pop(0))
                    child = self._Node(actions)
                    node.children[a] = child

                    state_sim, c, exog = self._step_model_full(state_sim, a, t_sim, rng)
                    rewards.append(-float(c))   # R = -C (convert cost to reward)
                    self._update_rollout_info(rollout_info, exog, rng)
                    node = child
                    path.append(node)
                    depth += 1
                    t_sim += 1
                    break   # switch to rollout after one expansion

                if not node.children:
                    break

                # Selection: follow UCT(a) = Q_bar(a) + c*sqrt(log N / N(a))
                a = node.uct_select(self.uct_c)
                state_sim, c, exog = self._step_model_full(state_sim, a, t_sim, rng)
                rewards.append(-float(c))   # R = -C
                self._update_rollout_info(rollout_info, exog, rng)
                node = node.children[a]
                path.append(node)
                depth += 1
                t_sim += 1

            # Phase 2 — Rollout
            # Simulate remaining steps to horizon H using the order-up-to heuristic.
            while depth < self.horizon:
                a = self._select_rollout_action(state_sim, t_sim, info=rollout_info)
                state_sim, c, exog = self._step_model_full(state_sim, a, t_sim, rng)
                rewards.append(-float(c))   # R = -C
                self._update_rollout_info(rollout_info, exog, rng)
                depth += 1
                t_sim += 1

            # G_m = sum_{k=0}^{H-1} R_k  = -sum C_k  (total return for this simulation)
            G = float(np.sum(rewards)) if rewards else 0.0

            # Phase 3 — Backpropagation
            # Update every node on the visited path:  N <- N+1,  W <- W + G_m
            for nd in path:
                nd.N += 1
                nd.W += G

        if not root.children:
            return np.array([0.0], dtype=float)

        # Final decision: most-visited child  a_hat = argmax_{a in X} N(root, a).
        # Visit count is a robust statistic for finite simulation budgets
        # (preferred over noisy mean-value ranking).
        best_a = max(root.children.items(), key=lambda kv: kv[1].N)[0]
        best_a = float(int(np.round(best_a)))
        best_a = max(0.0, min(best_a, self.x_max))   # clip to feasible range
        return np.array([best_a], dtype=float)


@dataclass
class DlaMctsUctRolloutPolicy(DlaMctsUctPolicy):
    """UCT search with an explicit rollout policy for the default tail."""

    rollout_policy: Optional[Policy] = None
    rollout_history_window: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rollout_history_window = int(self.rollout_history_window)
        if self.rollout_policy is None:
            raise ValueError("rollout_policy must be provided")
        if self.rollout_history_window < 0:
            raise ValueError("rollout_history_window must be nonnegative")

    def __repr__(self) -> str:
        rollout_name = type(self.rollout_policy).__name__ if self.rollout_policy is not None else "None"
        return (
            "DlaMctsUctRolloutPolicy("
            f"horizon={self.horizon}, n_simulations={self.n_simulations}, "
            f"uct_c={self.uct_c}, x_max={self.x_max}, dx={self.dx}, "
            f"planning_seed={self.planning_seed}, rollout_policy={rollout_name}, "
            f"rollout_history_window={self.rollout_history_window})"
        )

    def _select_rollout_action(self, state: State, t: int, info: Optional[dict] = None) -> float:
        action = self.rollout_policy.act(np.asarray(state, dtype=float).copy(), int(t), info=info)
        return self._sanitize_action_scalar(action)


DLA_MCTS_UCT = DlaMctsUctPolicy
DLA_MCTS_UCT_Rollout = DlaMctsUctRolloutPolicy


__all__ = [
    "DlaMctsUctPolicy",
    "DlaMctsUctRolloutPolicy",
    "DLA_MCTS_UCT",
    "DLA_MCTS_UCT_Rollout",
]
