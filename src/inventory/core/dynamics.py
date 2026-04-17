from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from inventory.core.exogenous import ExogenousModel
from inventory.core.policy import Policy
from inventory.core.types import Action, Exog, State


@dataclass(frozen=True)
class CRNEvaluationResult:
    mean: float
    std: float
    totals: np.ndarray
    runtime_sec: float = 0.0


class DynamicSystemMVP:
    """Minimal dynamic system for DP / PFA / CFA / VFA / DLA (vector conventions).

    Model:
      W_{t+1} ~ exogenous(S_t, X_t, t)      (sampled by ExogenousModel)
      S_{t+1} = transition(S_t, X_t, W_{t+1}, t)
      C_t     = cost(S_t, X_t, W_{t+1}, t)

    Conventions:
      - S, X, W are ALWAYS 1-D np.ndarrays (vectors), even if length 1.
      - strict CRN uses shared per-step RNG seeds (not shared realized W),
        which remains valid even when W depends on (S, X, t).

    Public API:
      - sample_crn_step_seeds(episode_seed, T) -> (T,) int64
      - simulate(policy, S0, T=10, seed=None) -> (traj, costs, actions, step_seeds)
      - simulate_crn(policy, S0, step_seeds) -> (traj, costs, actions, step_seeds)
      - evaluate_policy_crn_mc(policy, S0, T=10, n_episodes=200, seed0=1234) -> totals
      - evaluate_policies_crn_mc(policies, S0, T=10, n_episodes=200, seed0=1234)
    """

    def __init__(
        self,
        transition_func: Callable[[State, Action, Exog, int], State],
        cost_func: Callable[[State, Action, Exog, int], float],
        exogenous_model: ExogenousModel,
        *,
        sim_seed: int = 42,
        d_s: Optional[int] = None,
        d_x: Optional[int] = None,
        d_w: Optional[int] = None,
        demand_history_window: int = 0,
        # Notebook-API aliases (Lecture 01_d / Lecture 03_b)
        dS: Optional[int] = None,
        dX: Optional[int] = None,
        dW: Optional[int] = None,
    ):
        if d_s is not None and dS is not None:
            raise TypeError("Specify only one of d_s or dS")
        if d_x is not None and dX is not None:
            raise TypeError("Specify only one of d_x or dX")
        if d_w is not None and dW is not None:
            raise TypeError("Specify only one of d_w or dW")

        self.transition_func = transition_func
        self.cost_func = cost_func
        self.exogenous_model = exogenous_model

        self.sim_seed = int(sim_seed)
        self._sim_root = np.random.SeedSequence(self.sim_seed)

        self.demand_history_window = int(demand_history_window)
        if self.demand_history_window < 0:
            raise ValueError("demand_history_window must be nonnegative")

        # Optional dimension hints (lets you assert shapes)
        self.d_s = d_s if d_s is not None else dS
        self.d_x = d_x if d_x is not None else dX
        self.d_w = d_w if d_w is not None else dW

    def initial_state(self, *values: float) -> State:
        """Convenience helper for notebooks.

        - If `values` are provided, returns them as a 1-D float vector.
        - Otherwise, returns a zero vector of length `d_s` (requires `d_s` to be set).
        """

        if values:
            return np.asarray(values, dtype=float)
        if self.d_s is None:
            raise ValueError("initial_state() requires d_s to be set on the system (or pass explicit values).")
        return np.zeros(int(self.d_s), dtype=float)

    # -----------------------------
    # Shape / type normalization
    # -----------------------------
    @staticmethod
    def _as_1d_float_array(x: Union[Sequence[float], np.ndarray], name: str) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1-D vector. Got shape {arr.shape}.")
        return arr

    @staticmethod
    def _check_dim(arr: np.ndarray, expected: Optional[int], name: str) -> None:
        if expected is not None and arr.shape[0] != expected:
            raise ValueError(f"{name} has dim {arr.shape[0]} but expected {expected}.")

    # -----------------------------
    # CRN primitive
    # -----------------------------
    def sample_crn_step_seeds(self, episode_seed: int, T: int) -> np.ndarray:
        """Deterministically produce per-step seeds (length T) from an episode seed."""
        T = int(T)
        ss = np.random.SeedSequence(int(episode_seed))
        child = ss.spawn(T)
        return np.array([int(c.generate_state(1)[0]) for c in child], dtype=np.int64)

    # -----------------------------
    # Unified rollout engine (internal)
    # -----------------------------
    def _rollout(
        self,
        policy: Policy,
        S0: State,
        T: int,
        *,
        step_seeds: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        info: Optional[dict] = None,
        return_exog: bool = False,
    ):
        """One rollout engine. Provide ONE of step_seeds (strict CRN) or seed."""
        T = int(T)

        S0 = self._as_1d_float_array(S0, "S0")
        self._check_dim(S0, self.d_s, "S0")
        d_s = int(S0.shape[0])

        # Determine step seed stream
        if step_seeds is not None:
            step_seeds = np.asarray(step_seeds, dtype=np.int64)
            if step_seeds.ndim != 1 or len(step_seeds) != T:
                raise ValueError("step_seeds must be a 1-D array of length T.")
        else:
            if seed is not None:
                step_seeds = self.sample_crn_step_seeds(int(seed), T)
            else:
                step_seeds = np.array(
                    [int(self._sim_root.spawn(1)[0].generate_state(1)[0]) for _ in range(T)],
                    dtype=np.int64,
                )

        traj = np.empty((T + 1, d_s), dtype=float)
        costs = np.empty(T, dtype=float)
        traj[0] = S0

        actions_arr: Optional[np.ndarray] = None
        exogs_arr: Optional[np.ndarray] = None
        last_demand: Optional[float] = None
        demand_history: deque[float] = deque(maxlen=self.demand_history_window) if self.demand_history_window > 0 else deque()

        for t in range(T):
            S_t = traj[t].copy()

            step_info = {} if info is None else dict(info)
            step_info["crn_step_seed"] = int(step_seeds[t])
            if last_demand is not None:
                step_info["last_demand"] = float(last_demand)
            if demand_history:
                step_info["demand_history"] = np.asarray(demand_history, dtype=float)

            X_t = policy.act(S_t, t, info=step_info)
            X_t = self._as_1d_float_array(X_t, "X_t")
            self._check_dim(X_t, self.d_x, "X_t")
            if actions_arr is None:
                d_x = int(X_t.shape[0])
                actions_arr = np.empty((T, d_x), dtype=float)
            actions_arr[t] = X_t

            rng_t = np.random.default_rng(int(step_seeds[t]))
            W_tp1 = self.exogenous_model.sample(S_t, X_t, t, rng_t)
            W_tp1 = self._as_1d_float_array(W_tp1, "W_{t+1}")
            self._check_dim(W_tp1, self.d_w, "W_{t+1}")
            if W_tp1.size >= 1:
                last_demand = float(W_tp1.reshape(-1)[0])
                if self.demand_history_window > 0:
                    demand_history.append(last_demand)
            if return_exog:
                if exogs_arr is None:
                    d_w = int(W_tp1.shape[0])
                    exogs_arr = np.empty((T, d_w), dtype=float)
                exogs_arr[t] = W_tp1

            C_t = float(self.cost_func(S_t, X_t, W_tp1, t))
            S_tp1 = self.transition_func(S_t, X_t, W_tp1, t)

            costs[t] = C_t

            S_tp1 = self._as_1d_float_array(S_tp1, "S_{t+1}")
            self._check_dim(S_tp1, self.d_s, "S_{t+1}")
            traj[t + 1] = S_tp1

        if actions_arr is None:
            actions_arr = np.empty((0, self.d_x or 0), dtype=float)

        if return_exog:
            if exogs_arr is None:
                exogs_arr = np.empty((0, self.d_w or 0), dtype=float)
            return traj, costs, actions_arr, step_seeds, exogs_arr

        return traj, costs, actions_arr, step_seeds

    # -----------------------------
    # Public simulation methods
    # -----------------------------
    def simulate(
        self,
        policy: Policy,
        S0: State,
        *,
        T: int = 10,
        seed: Optional[int] = None,
        info: Optional[dict] = None,
    ):
        return self._rollout(policy, S0, int(T), seed=seed, info=info, return_exog=False)

    def simulate_crn(self, policy: Policy, S0: State, step_seeds: np.ndarray, *, info: Optional[dict] = None):
        T = len(step_seeds)
        return self._rollout(policy, S0, int(T), step_seeds=step_seeds, info=info, return_exog=False)

    def collect_policies_crn_rollouts_mc(
        self,
        policies: Dict[str, Policy],
        S0: State,
        *,
        T: int = 10,
        n_episodes: int = 50,
        seed0: int = 1234,
        info: Optional[dict] = None,
        store_step_seeds: bool = False,
    ) -> Dict[str, List[dict]]:
        """Collect full episode rollouts under strict CRN for plotting/diagnostics.

        This complements `evaluate_policies_crn_mc`, which returns totals plus ONE
        reference rollout per policy.

        Returns
        -------
        rollouts_by_policy : dict[str, list[dict]]
            Mapping policy name -> list of episode dicts:
              {
                "traj": (T+1, dS) array,
                "actions": (T, dX) array,
                "costs": (T,) array,
                "ws": (T, dW) array,
                "step_seeds": (T,) int64 array   (optional)
              }
        """

        # Evaluation convention: deterministic by default.
        if info is None:
            info = {"deterministic": True}
        elif isinstance(info, dict) and "deterministic" not in info:
            info = dict(info)
            info["deterministic"] = True

        T = int(T)
        n_episodes = int(n_episodes)

        if n_episodes < 0:
            raise ValueError("n_episodes must be >= 0")

        rng = np.random.default_rng(int(seed0))
        episode_seeds = [int(s) for s in rng.integers(1, 2**31 - 1, size=n_episodes)]
        step_seed_paths = [self.sample_crn_step_seeds(ep_seed, T) for ep_seed in episode_seeds]

        rollouts_by_policy: Dict[str, List[dict]] = {name: [] for name in policies.keys()}
        for name, pol in policies.items():
            eps: List[dict] = []
            for step_seeds in step_seed_paths:
                traj, costs, actions, step_seeds_out, ws = self._rollout(
                    pol,
                    S0,
                    T,
                    step_seeds=step_seeds,
                    info=info,
                    return_exog=True,
                )
                ep = {
                    "traj": traj,
                    "actions": actions,
                    "costs": costs,
                    "ws": ws,
                }
                if store_step_seeds:
                    ep["step_seeds"] = step_seeds_out
                eps.append(ep)
            rollouts_by_policy[name] = eps

        return rollouts_by_policy

    # -----------------------------
    # Monte Carlo evaluation with strict CRN
    # -----------------------------
    def evaluate_policy_crn_mc(
        self,
        policy: Policy,
        S0: State,
        *,
        T: int = 10,
        n_episodes: int = 200,
        seed0: int = 1234,
        info: Optional[dict] = None,
    ) -> np.ndarray:
        # Evaluation convention: deterministic by default.
        # (Training code should explicitly pass deterministic=False if desired.)
        if info is None:
            info = {"deterministic": True}
        elif isinstance(info, dict) and "deterministic" not in info:
            info = dict(info)
            info["deterministic"] = True

        T = int(T)
        n_episodes = int(n_episodes)

        rng = np.random.default_rng(int(seed0))
        episode_seeds = [int(s) for s in rng.integers(1, 2**31 - 1, size=n_episodes)]

        totals = np.empty(n_episodes, dtype=float)
        for i, ep_seed in enumerate(episode_seeds):
            step_seeds = self.sample_crn_step_seeds(ep_seed, T)
            _, costs, _, _ = self.simulate_crn(policy, S0, step_seeds, info=info)
            totals[i] = float(costs.sum())
        return totals

    def evaluate_policies_crn_mc(
        self,
        policies: Dict[str, Policy],
        S0: State,
        *,
        T: int = 10,
        n_episodes: int = 200,
        seed0: int = 1234,
        info: Optional[dict] = None,
    ):
        # Evaluation convention: deterministic by default.
        if info is None:
            info = {"deterministic": True}
        elif isinstance(info, dict) and "deterministic" not in info:
            info = dict(info)
            info["deterministic"] = True

        T = int(T)
        n_episodes = int(n_episodes)

        rng = np.random.default_rng(int(seed0))
        episode_seeds = [int(s) for s in rng.integers(1, 2**31 - 1, size=n_episodes)]
        step_seed_paths = [self.sample_crn_step_seeds(ep_seed, T) for ep_seed in episode_seeds]

        results: Dict[str, CRNEvaluationResult] = {}
        runtime_by_policy: Dict[str, float] = {}
        for name, pol in policies.items():
            t0 = perf_counter()
            totals = np.empty(n_episodes, dtype=float)
            for i, step_seeds in enumerate(step_seed_paths):
                _, costs, _, _ = self.simulate_crn(pol, S0, step_seeds, info=info)
                totals[i] = float(costs.sum())
            runtime_by_policy[name] = float(perf_counter() - t0)
            results[name] = CRNEvaluationResult(
                mean=float(totals.mean()),
                std=float(totals.std(ddof=1)) if len(totals) > 1 else 0.0,
                totals=totals,
                runtime_sec=runtime_by_policy[name],
            )

        # One common rollout for plotting/debugging
        rollouts = {}
        if n_episodes > 0:
            ref_step_seeds = step_seed_paths[0]
            for name, pol in policies.items():
                t0 = perf_counter()
                traj, costs, actions, _, ws = self._rollout(
                    pol,
                    S0,
                    T,
                    step_seeds=ref_step_seeds,
                    info=info,
                    return_exog=True,
                )
                runtime_by_policy[name] = runtime_by_policy.get(name, 0.0) + float(perf_counter() - t0)
                res = results[name]
                results[name] = CRNEvaluationResult(
                    mean=res.mean,
                    std=res.std,
                    totals=res.totals,
                    runtime_sec=runtime_by_policy[name],
                )
                rollouts[name] = {"traj": traj, "costs": costs, "actions": actions, "ws": ws}

        return results, rollouts


# Backwards-compatible rename: prefer `DynamicSystem` going forward, but keep
# `DynamicSystemMVP` as the canonical implementation name for older notebooks.
DynamicSystem = DynamicSystemMVP


__all__ = ["DynamicSystem", "DynamicSystemMVP", "CRNEvaluationResult"]
