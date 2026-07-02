from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Union, runtime_checkable

import numpy as np
from inventory.core.policy import Policy
from inventory.core.types import Action, State


PolicyLike = Union[Policy, Callable[[State, int], Action]]


@runtime_checkable
class TerminalValueProvider(Protocol):
    """Interface for terminal continuation value in deterministic lookahead.

    Convention:
    - Returns a continuation *value* where larger is better.
    - The deterministic DP minimizes cost, so it subtracts this value as
      negative terminal cost at the horizon boundary.
    """

    def value_terminal(self, state: State, t: int) -> float: ...


@dataclass(frozen=True)
class ZeroTerminalValueProvider:
    """Terminal continuation provider that contributes zero value."""

    def value_terminal(self, state: State, t: int) -> float:
        _ = state, t
        return 0.0


@dataclass(frozen=True)
class InventoryLinearTerminalValue:
    """Linear terminal continuation value based only on inventory."""

    theta0: float = 0.0
    theta1: float = 0.0

    def value_terminal(self, state: State, t: int) -> float:
        _ = t
        S = np.asarray(state, dtype=float).reshape(-1)
        inv = float(S[0]) if S.size >= 1 else 0.0
        return float(self.theta0 + self.theta1 * inv)


@dataclass(frozen=True)
class LegacyTerminalVfaAdapter:
    """Adapter from the legacy `TerminalVFA.V(inv, t)` interface."""

    terminal_vfa: object

    def value_terminal(self, state: State, t: int) -> float:
        S = np.asarray(state, dtype=float).reshape(-1)
        if not hasattr(self.terminal_vfa, "V"):
            raise TypeError("legacy terminal_vfa must expose V(inv, t)")
        return float(self.terminal_vfa.V(float(S[0]), int(t)))  # type: ignore[attr-defined]


def _policy_act(policy_like: PolicyLike, state: State, t: int, info: Optional[dict]) -> np.ndarray:
    if hasattr(policy_like, "act"):
        return np.asarray(policy_like.act(state, t, info=info), dtype=float).reshape(-1)  # type: ignore[attr-defined]
    return np.asarray(policy_like(state, t), dtype=float).reshape(-1)


class HybridDlaDeterministicComposablePolicy(Policy):
    """Composable deterministic-lookahead hybrid DLA policy.

    Roles:
    - `forecast_provider`: produces a deterministic mean path and optionally a
      standard-deviation path.
    - `terminal_value_provider`: supplies a continuation value at the horizon end.
    - `candidate_policy`: optionally narrows the *root* action set around a
      proposed action while leaving the interior DP recursion untouched.
    """

    def __init__(
        self,
        system,
        forecast_provider: object,
        *,
        terminal_value_provider: Optional[TerminalValueProvider] = None,
        candidate_policy: Optional[PolicyLike] = None,
        terminal_state_builder: Optional[Callable[[float, np.ndarray, int], np.ndarray]] = None,
        H: int = 5,
        buffer_k: float = 0.0,
        dx: int = 10,
        x_max: int = 500,
        s_max: int = 1000,
        s_grid_step: Optional[int] = None,
        candidate_radius_steps: int = 0,
        include_full_action_grid: bool = True,
        p: float = 2.0,
        c: float = 0.5,
        h: float = 0.03,
        b: float = 1.0,
        K: float = 0.0,
    ):
        self.system = system
        self.forecast_provider = forecast_provider
        self.terminal_value_provider = terminal_value_provider or ZeroTerminalValueProvider()
        self.candidate_policy = candidate_policy
        self.terminal_state_builder = terminal_state_builder

        self.H = int(H)
        self.buffer_k = float(buffer_k)
        self.dx = int(dx)
        self.x_max = int(x_max)
        self.s_max = int(s_max)
        self.s_grid_step = int(s_grid_step) if s_grid_step is not None else int(dx)
        self.candidate_radius_steps = int(candidate_radius_steps)
        self.include_full_action_grid = bool(include_full_action_grid)

        self.p = float(p)
        self.c = float(c)
        self.h = float(h)
        self.b = float(b)
        self.K = float(K)

        if self.H <= 0:
            raise ValueError("H must be >= 1")
        if self.dx <= 0:
            raise ValueError("dx must be positive")
        if self.s_grid_step <= 0:
            raise ValueError("s_grid_step must be positive")
        if self.candidate_radius_steps < 0:
            raise ValueError("candidate_radius_steps must be >= 0")
        if not hasattr(self.forecast_provider, "forecast_mean_path"):
            raise TypeError("forecast_provider must expose forecast_mean_path(state, t, H, info=None)")
        if not isinstance(self.terminal_value_provider, TerminalValueProvider):
            raise TypeError("terminal_value_provider must satisfy the TerminalValueProvider protocol")

        self._A = np.arange(0, self.x_max + self.dx, self.dx, dtype=float)
        self._Sgrid = np.arange(0, self.s_max + self.s_grid_step, self.s_grid_step, dtype=float)

    def __repr__(self) -> str:
        forecast_name = type(self.forecast_provider).__name__
        terminal_name = type(self.terminal_value_provider).__name__
        candidate_name = None
        if self.candidate_policy is not None:
            candidate_name = (
                type(self.candidate_policy).__name__
                if hasattr(self.candidate_policy, "__class__")
                else getattr(self.candidate_policy, "__name__", None)
            )
        return (
            "HybridDlaDeterministicComposablePolicy("
            f"forecast_provider={forecast_name!r}, terminal_value_provider={terminal_name!r}, "
            f"candidate_policy={candidate_name!r}, H={self.H}, buffer_k={self.buffer_k}, "
            f"dx={self.dx}, x_max={self.x_max}, s_max={self.s_max}, s_grid_step={self.s_grid_step}, "
            f"candidate_radius_steps={self.candidate_radius_steps}, "
            f"include_full_action_grid={self.include_full_action_grid}, "
            f"p={self.p}, c={self.c}, h={self.h}, b={self.b}, K={self.K})"
        )

    def _pack_action(self, x: float) -> Action:
        x = float(np.clip(x, 0.0, self.x_max))
        if self.dx > 1:
            x = self.dx * np.round(x / self.dx)
        return np.array([float(int(np.round(x)))], dtype=float)

    def _scalar_from_action_like(self, action: Action | float) -> float:
        arr = np.asarray(action, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError("action must contain at least one scalar")
        x = float(arr[0])
        x = float(np.clip(x, 0.0, self.x_max))
        if self.dx > 1:
            x = self.dx * np.round(x / self.dx)
        return float(int(np.round(x)))

    def _forecast_mean_path(self, state: State, t: int, info: Optional[dict]) -> np.ndarray:
        mu = np.asarray(
            self.forecast_provider.forecast_mean_path(state, int(t), self.H, info=info),  # type: ignore[attr-defined]
            dtype=float,
        ).reshape(-1)
        if mu.shape[0] != self.H:
            raise ValueError(f"forecast_mean_path returned shape {mu.shape}; expected ({self.H},)")
        return mu

    def _forecast_std_path(self, state: State, t: int, info: Optional[dict]) -> np.ndarray:
        if hasattr(self.forecast_provider, "forecast_std_path"):
            sig = np.asarray(
                self.forecast_provider.forecast_std_path(state, int(t), self.H, info=info),  # type: ignore[attr-defined]
                dtype=float,
            ).reshape(-1)
            if sig.shape[0] != self.H:
                raise ValueError(f"forecast_std_path returned shape {sig.shape}; expected ({self.H},)")
            return np.maximum(0.0, sig)
        return np.zeros(self.H, dtype=float)

    def _demand_hat_path(self, state: State, t: int, info: Optional[dict]) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(-1)
        mu = self._forecast_mean_path(state, t, info)
        sig = self._forecast_std_path(state, t, info)
        d = mu + self.buffer_k * sig
        return np.maximum(0.0, d).astype(float)

    def _terminal_state_from_inventory(self, inv: float, ref_state: np.ndarray, t: int) -> np.ndarray:
        if self.terminal_state_builder is not None:
            return np.asarray(self.terminal_state_builder(float(inv), np.asarray(ref_state, dtype=float).copy(), int(t)), dtype=float).reshape(-1)
        ref = np.asarray(ref_state, dtype=float).reshape(-1)
        if ref.size <= 1:
            return np.array([float(inv)], dtype=float)
        return np.concatenate(([float(inv)], ref[1:].copy()))

    def _terminal_cost_from_inventory(self, inv: float, ref_state: np.ndarray, t: int) -> float:
        state_term = self._terminal_state_from_inventory(inv, ref_state, t)
        value = float(self.terminal_value_provider.value_terminal(state_term, int(t)))
        return -value

    def _root_candidate_actions(self, state: np.ndarray, t: int, info: dict) -> np.ndarray:
        if self.include_full_action_grid or self.candidate_policy is None:
            return self._A.copy()

        x_base = self._scalar_from_action_like(_policy_act(self.candidate_policy, state, int(t), info))
        if self.candidate_radius_steps <= 0:
            return np.array([float(x_base)], dtype=float)

        offsets = np.arange(-self.candidate_radius_steps, self.candidate_radius_steps + 1) * float(self.dx)
        cand = x_base + offsets
        cand = np.clip(cand, 0.0, float(self.x_max))
        cand = np.unique(cand.astype(float))
        return np.sort(cand)

    def _solve_root_action(self, state: np.ndarray, t: int, demand_hat_path: np.ndarray, root_actions: np.ndarray) -> float:
        Sg = self._Sgrid
        A = self._A

        t_terminal = int(t) + self.H
        J_next = np.array([self._terminal_cost_from_inventory(float(s), state, t_terminal) for s in Sg], dtype=float)

        for k in range(self.H - 1, -1, -1):
            d = float(demand_hat_path[k])

            # Mirror the capped simulator semantics: received stock is clipped
            # pre-demand at s_max, but purchase/setup cost still uses the actual
            # order quantity A just as the capped cost functions do.
            on_hand = np.minimum(Sg[:, None] + A[None, :], float(self.s_max))
            inv_next = np.maximum(0.0, on_hand - d)
            setup = self.K * (A[None, :] > 0.0).astype(float)

            sales = np.minimum(on_hand, d)
            lost = np.maximum(0.0, d - on_hand)
            cost = setup + (self.c * A[None, :]) + (self.h * inv_next) + (self.b * lost) - (self.p * sales)

            idx = np.searchsorted(Sg, inv_next, side="left")
            idx = np.clip(idx, 0, len(Sg) - 1)
            idx0 = np.clip(idx - 1, 0, len(Sg) - 1)
            idx1 = idx
            s0 = Sg[idx0]
            s1 = Sg[idx1]
            denom = s1 - s0

            w = np.zeros_like(inv_next)
            mask = denom > 0
            w[mask] = (inv_next[mask] - s0[mask]) / denom[mask]

            J_interp = (1.0 - w) * J_next[idx0] + w * J_next[idx1]
            Q = cost + J_interp
            J_next = np.min(Q, axis=1)

        inv0 = float(np.clip(float(state[0]), 0.0, float(self.s_max)))
        d0 = float(demand_hat_path[0])
        root_actions = np.asarray(root_actions, dtype=float).reshape(-1)

        # Root step uses the same pre-demand cap semantics as the interior DP.
        on_hand0 = np.minimum(inv0 + root_actions, float(self.s_max))
        inv_next0 = np.maximum(0.0, on_hand0 - d0)
        setup0 = self.K * (root_actions > 0.0).astype(float)

        sales0 = np.minimum(on_hand0, d0)
        lost0 = np.maximum(0.0, d0 - on_hand0)
        cost0 = setup0 + (self.c * root_actions) + (self.h * inv_next0) + (self.b * lost0) - (self.p * sales0)

        idx = np.searchsorted(Sg, inv_next0, side="left")
        idx = np.clip(idx, 0, len(Sg) - 1)
        idx0 = np.clip(idx - 1, 0, len(Sg) - 1)
        idx1 = idx
        s0 = Sg[idx0]
        s1 = Sg[idx1]
        denom = s1 - s0

        w = np.zeros_like(inv_next0)
        mask = denom > 0
        w[mask] = (inv_next0[mask] - s0[mask]) / denom[mask]

        J_interp0 = (1.0 - w) * J_next[idx0] + w * J_next[idx1]
        Q0 = cost0 + J_interp0
        a_idx = int(np.argmin(Q0))
        return float(root_actions[a_idx])

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        S = np.asarray(state, dtype=float).reshape(-1)
        t = int(t)
        info = {} if info is None else dict(info)

        demand_hat_path = self._demand_hat_path(S, t, info)
        root_actions = self._root_candidate_actions(S, t, info)
        best_a = self._solve_root_action(S, t, demand_hat_path, root_actions)
        return self._pack_action(best_a)


class HybridDlaRolloutComposablePolicy(Policy):
    """Composable rollout-based hybrid DLA policy.

    Roles:
    - `rollout_policy`: continuation policy used after the first candidate action.
    - `candidate_policy`: optional proposal policy used to narrow the root action set.

    This class preserves uncertainty explicitly by evaluating candidate first
    actions across multiple sampled futures, then choosing the candidate with the
    lowest average rollout cost.
    """

    def __init__(
        self,
        system,
        rollout_policy: PolicyLike,
        *,
        candidate_policy: Optional[PolicyLike] = None,
        H: int = 5,
        n_rollouts: int = 30,
        dx: int = 10,
        x_max: int = 500,
        decision_seed: int = 2026,
        candidate_radius_steps: int = 5,
        include_full_action_grid: bool = False,
        adaptive_candidate_radius: bool = False,
        adaptive_radius_near: int = 2,
        adaptive_radius_far: int = 8,
        adaptive_gap_threshold: float = 40.0,
        rollout_history_window: int = 0,
    ):
        self.system = system
        self.rollout_policy = rollout_policy
        self.candidate_policy = candidate_policy

        self.H = int(H)
        self.n_rollouts = int(n_rollouts)
        self.dx = int(dx)
        self.x_max = int(x_max)
        self.decision_seed = int(decision_seed)
        self.candidate_radius_steps = int(candidate_radius_steps)
        self.include_full_action_grid = bool(include_full_action_grid)
        self.adaptive_candidate_radius = bool(adaptive_candidate_radius)
        self.adaptive_radius_near = int(adaptive_radius_near)
        self.adaptive_radius_far = int(adaptive_radius_far)
        self.adaptive_gap_threshold = float(adaptive_gap_threshold)
        self.rollout_history_window = int(rollout_history_window)

        if self.H <= 0:
            raise ValueError("H must be >= 1")
        if self.n_rollouts <= 0:
            raise ValueError("n_rollouts must be >= 1")
        if self.dx <= 0:
            raise ValueError("dx must be positive")
        if self.candidate_radius_steps < 0:
            raise ValueError("candidate_radius_steps must be >= 0")
        if self.adaptive_radius_near < 0 or self.adaptive_radius_far < 0:
            raise ValueError("adaptive radii must be >= 0")
        if self.adaptive_gap_threshold < 0.0:
            raise ValueError("adaptive_gap_threshold must be >= 0")
        if self.rollout_history_window < 0:
            raise ValueError("rollout_history_window must be >= 0")

        self._A = np.arange(0, self.x_max + self.dx, self.dx, dtype=float)

    def __repr__(self) -> str:
        rollout_name = getattr(self.rollout_policy, "__name__", type(self.rollout_policy).__name__)
        candidate_obj = self.candidate_policy or self.rollout_policy
        candidate_name = getattr(candidate_obj, "__name__", type(candidate_obj).__name__)
        return (
            "HybridDlaRolloutComposablePolicy("
            f"rollout_policy={rollout_name!r}, candidate_policy={candidate_name!r}, "
            f"H={self.H}, n_rollouts={self.n_rollouts}, dx={self.dx}, x_max={self.x_max}, "
            f"decision_seed={self.decision_seed}, candidate_radius_steps={self.candidate_radius_steps}, "
            f"include_full_action_grid={self.include_full_action_grid}, "
            f"adaptive_candidate_radius={self.adaptive_candidate_radius}, "
            f"adaptive_radius_near={self.adaptive_radius_near}, adaptive_radius_far={self.adaptive_radius_far}, "
            f"adaptive_gap_threshold={self.adaptive_gap_threshold}, "
            f"rollout_history_window={self.rollout_history_window})"
        )

    def _pack_action(self, x: float) -> Action:
        x = float(np.clip(x, 0.0, self.x_max))
        if self.dx > 1:
            x = self.dx * np.round(x / self.dx)
        return np.array([float(int(np.round(x)))], dtype=float)

    def _scalar_from_action_like(self, action: Action | float) -> float:
        arr = np.asarray(action, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError("action must contain at least one scalar")
        x = float(arr[0])
        x = float(np.clip(x, 0.0, self.x_max))
        if self.dx > 1:
            x = self.dx * np.round(x / self.dx)
        return float(int(np.round(x)))

    def _proposal_policy(self) -> PolicyLike:
        return self.candidate_policy if self.candidate_policy is not None else self.rollout_policy

    def _candidate_radius_for_state(self, state: np.ndarray, t: int, info: dict) -> int:
        if not self.adaptive_candidate_radius:
            return self.candidate_radius_steps

        proposal = self._proposal_policy()
        target = None
        if hasattr(proposal, "target_level"):
            target = float(proposal.target_level)  # type: ignore[attr-defined]
        elif hasattr(proposal, "S_bar"):
            target = float(proposal.S_bar)  # type: ignore[attr-defined]

        inv = float(np.asarray(state, dtype=float).reshape(-1)[0])
        if target is not None:
            gap = abs(target - inv)
        else:
            x_base = self._scalar_from_action_like(_policy_act(proposal, state, t, info))
            gap = abs(x_base)

        return self.adaptive_radius_near if gap <= self.adaptive_gap_threshold else self.adaptive_radius_far

    def _candidate_actions(self, state: np.ndarray, t: int, info: dict) -> np.ndarray:
        if self.include_full_action_grid:
            return self._A.copy()

        proposal = self._proposal_policy()
        radius_steps = self._candidate_radius_for_state(state, t, info)
        x_base = self._scalar_from_action_like(_policy_act(proposal, state, t, info))
        if radius_steps <= 0:
            return np.array([float(x_base)], dtype=float)

        offsets = np.arange(-radius_steps, radius_steps + 1) * float(self.dx)
        cand = x_base + offsets
        cand = np.clip(cand, 0.0, float(self.x_max))
        cand = np.unique(cand.astype(float))
        return np.sort(cand)

    def _history_window(self) -> int:
        if self.rollout_history_window > 0:
            return self.rollout_history_window
        return int(getattr(self.system, "demand_history_window", 0))

    def _init_rollout_info(self, info: Optional[dict]) -> dict:
        rollout_info = {} if info is None else dict(info)
        hist = np.asarray(rollout_info.get("demand_history", []), dtype=float).reshape(-1)
        hist_window = self._history_window()
        if hist_window > 0 and hist.size > hist_window:
            hist = hist[-hist_window:]
        if hist.size > 0:
            rollout_info["demand_history"] = hist
        else:
            rollout_info.pop("demand_history", None)
        return rollout_info

    def _update_rollout_info(self, rollout_info: dict, exog: np.ndarray, rng: np.random.Generator) -> None:
        exog_arr = np.asarray(exog, dtype=float).reshape(-1)
        if exog_arr.size >= 1:
            demand = float(exog_arr[0])
            rollout_info["last_demand"] = demand

            hist = np.asarray(rollout_info.get("demand_history", []), dtype=float).reshape(-1)
            hist_window = self._history_window()
            if hist.size > 0 or hist_window > 0:
                hist = np.concatenate([hist, np.array([demand], dtype=float)])
                if hist_window > 0 and hist.size > hist_window:
                    hist = hist[-hist_window:]
                if hist.size > 0:
                    rollout_info["demand_history"] = hist

        rollout_info["crn_step_seed"] = int(
            rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)
        )

    def _rollout_cost(self, state0: State, t0: int, x0: Action, rng: np.random.Generator, info: dict) -> float:
        state = np.asarray(state0, dtype=float).reshape(-1).copy()
        rollout_info = self._init_rollout_info(info)
        total = 0.0

        for k in range(self.H):
            if k == 0:
                action = np.asarray(x0, dtype=float).reshape(-1)
            else:
                action = _policy_act(self.rollout_policy, state, int(t0) + k, rollout_info)

            exog = np.asarray(
                self.system.exogenous_model.sample(state, action, int(t0) + k, rng),
                dtype=float,
            ).reshape(-1)
            total += float(self.system.cost_func(state, action, exog, int(t0) + k))
            state = np.asarray(
                self.system.transition_func(state, action, exog, int(t0) + k),
                dtype=float,
            ).reshape(-1)
            self._update_rollout_info(rollout_info, exog, rng)

        return float(total)

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        S = np.asarray(state, dtype=float).reshape(-1)
        t = int(t)
        info = {} if info is None else dict(info)

        crn_step_seed = info.get("crn_step_seed", 0)
        ss = np.random.SeedSequence([int(self.decision_seed), int(crn_step_seed), int(t)])
        rng = np.random.default_rng(ss)

        cand = self._candidate_actions(S, t, info)
        rollout_seeds = rng.integers(1, 2**31 - 1, size=self.n_rollouts, dtype=np.int64)

        best_x = self._pack_action(float(cand[0]) if len(cand) else 0.0)
        best_val = np.inf

        for x0 in cand:
            x0_vec = self._pack_action(float(x0))
            vals = []
            for seed in rollout_seeds:
                rng_r = np.random.default_rng(int(seed))
                vals.append(self._rollout_cost(S, t, x0_vec, rng_r, info))
            v = float(np.mean(vals))
            if v < best_val:
                best_val = v
                best_x = x0_vec

        return best_x


__all__ = [
    "TerminalValueProvider",
    "ZeroTerminalValueProvider",
    "InventoryLinearTerminalValue",
    "LegacyTerminalVfaAdapter",
    "HybridDlaDeterministicComposablePolicy",
    "HybridDlaRolloutComposablePolicy",
]
