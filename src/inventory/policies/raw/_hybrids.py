from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
from inventory.core.policy import Policy
from inventory.core.types import Action, State
from inventory.forecasters.path import DemandPathForecaster


class TerminalVFA:
    def V(self, inv: float, t: int) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class TerminalVfaLinear(TerminalVFA):
    """V-hat: V(inv,t) = theta0 + theta1*inv."""

    theta0: float = 0.0
    theta1: float = 0.0

    def V(self, inv: float, t: int) -> float:
        _ = t
        return float(self.theta0 + self.theta1 * float(inv))


class HybridDlaCfaBufferedForecastPolicy(Policy):
    """Hybrid #1 (DLA + CFA): deterministic lookahead with buffered demand forecast.

    Uses a path forecast and constructs a deterministic demand path:
      d_hat[k] = mu[k] + buffer_k * std[k]

    Then solves an H-step deterministic DP on discretized grids and returns the first action.
    """

    def __init__(
        self,
        system,
        forecast: DemandPathForecaster,
        *,
        H: int = 5,
        buffer_k: float = 0.0,
        dx: int = 10,
        x_max: int = 500,
        s_max: int = 1000,
        s_grid_step: Optional[int] = None,
        # cost params (match inventory_cost defaults)
        p: float = 2.0,
        c: float = 0.5,
        h: float = 0.03,
        b: float = 1.0,
    ):
        self.system = system
        self.forecast = forecast

        self.H = int(H)
        self.buffer_k = float(buffer_k)
        self.dx = int(dx)
        self.x_max = int(x_max)
        self.s_max = int(s_max)
        self.s_grid_step = int(s_grid_step) if s_grid_step is not None else int(dx)

        self.p = float(p)
        self.c = float(c)
        self.h = float(h)
        self.b = float(b)

        if self.H <= 0:
            raise ValueError("H must be >= 1")
        if self.dx <= 0:
            raise ValueError("dx must be positive")
        if self.s_grid_step <= 0:
            raise ValueError("s_grid_step must be positive")

        self._A = np.arange(0, self.x_max + self.dx, self.dx, dtype=float)
        self._Sgrid = np.arange(0, self.s_max + self.s_grid_step, self.s_grid_step, dtype=float)

    def _pack_action(self, x: float) -> Action:
        x = float(np.clip(x, 0.0, self.x_max))
        if self.dx > 1:
            x = self.dx * np.round(x / self.dx)
        return np.array([float(int(np.round(x)))], dtype=float)

    def _demand_hat_path(self, state: State, t: int, info: Optional[dict]) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(-1)
        mu = np.asarray(self.forecast.forecast_mean_path(state, int(t), self.H, info=info), dtype=float).reshape(-1)
        sig = np.asarray(self.forecast.forecast_std_path(state, int(t), self.H, info=info), dtype=float).reshape(-1)
        if mu.shape[0] != self.H or sig.shape[0] != self.H:
            raise ValueError("Forecaster returned wrong shape; expected length H")
        d = mu + self.buffer_k * sig
        return np.maximum(0.0, d).astype(float)

    def _fast_path_available(self) -> bool:
        try:
            return (
                getattr(self.system, "cost_func", None) is not None
                and getattr(self.system, "transition_func", None) is not None
                and getattr(self.system.cost_func, "__name__", "") == "inventory_cost"
                and getattr(self.system.transition_func, "__name__", "") == "inventory_transition"
            )
        except Exception:
            return False

    def _dp_vectorized(self, inv0: float, demand_hat_path: np.ndarray) -> float:
        Sg = self._Sgrid
        A = self._A

        J_next = np.zeros(Sg.shape[0], dtype=float)

        for k in range(self.H - 1, -1, -1):
            d = float(demand_hat_path[k])

            on_hand = Sg[:, None] + A[None, :]
            inv_next = np.maximum(0.0, on_hand - d)

            sales = np.minimum(on_hand, d)
            lost = np.maximum(0.0, d - on_hand)
            cost = (self.c * A[None, :]) + (self.h * inv_next) + (self.b * lost) - (self.p * sales)

            idx = np.searchsorted(Sg, inv_next, side="left")
            idx = np.clip(idx, 0, len(Sg) - 1)
            idx0 = np.clip(idx - 1, 0, len(Sg) - 1)
            idx1 = idx
            s0 = Sg[idx0]
            s1 = Sg[idx1]
            denom = (s1 - s0)

            w = np.zeros_like(inv_next)
            mask = denom > 0
            w[mask] = (inv_next[mask] - s0[mask]) / denom[mask]

            J_interp = (1.0 - w) * J_next[idx0] + w * J_next[idx1]
            Q = cost + J_interp

            J_next = np.min(Q, axis=1)

        inv0 = float(np.clip(inv0, 0.0, float(self.s_max)))
        d0 = float(demand_hat_path[0])

        on_hand0 = inv0 + A
        inv_next0 = np.maximum(0.0, on_hand0 - d0)

        sales0 = np.minimum(on_hand0, d0)
        lost0 = np.maximum(0.0, d0 - on_hand0)
        cost0 = (self.c * A) + (self.h * inv_next0) + (self.b * lost0) - (self.p * sales0)

        idx = np.searchsorted(Sg, inv_next0, side="left")
        idx = np.clip(idx, 0, len(Sg) - 1)
        idx0 = np.clip(idx - 1, 0, len(Sg) - 1)
        idx1 = idx
        s0 = Sg[idx0]
        s1 = Sg[idx1]
        denom = (s1 - s0)

        w = np.zeros_like(inv_next0)
        mask = denom > 0
        w[mask] = (inv_next0[mask] - s0[mask]) / denom[mask]

        J_interp0 = (1.0 - w) * J_next[idx0] + w * J_next[idx1]
        Q0 = cost0 + J_interp0
        a_idx = int(np.argmin(Q0))
        return float(A[a_idx])

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        state = np.asarray(state, dtype=float).reshape(-1)
        inv0 = float(state[0])
        t = int(t)
        info = info or {}

        demand_hat_path = self._demand_hat_path(state, t, info)

        if self._fast_path_available():
            x0 = self._dp_vectorized(inv0, demand_hat_path)
            return self._pack_action(x0)

        # Generic fallback: loop over the same discretization grids.
        Sg = self._Sgrid
        A = self._A

        J_next = np.zeros_like(Sg, dtype=float)
        for k in range(self.H - 1, -1, -1):
            d = float(demand_hat_path[k])
            J = np.zeros_like(Sg, dtype=float)
            for i, s in enumerate(Sg):
                best = np.inf
                for a in A:
                    S_vec = np.array([float(s)], dtype=float)
                    X_vec = np.array([float(a)], dtype=float)
                    W_vec = np.array([float(d)], dtype=float)
                    c0 = float(self.system.cost_func(S_vec, X_vec, W_vec, t + k))
                    s_next = float(self.system.transition_func(S_vec, X_vec, W_vec, t + k)[0])
                    jn = float(np.interp(s_next, Sg, J_next))
                    best = min(best, c0 + jn)
                J[i] = best
            J_next = J

        best_a = 0.0
        best = np.inf
        d0 = float(demand_hat_path[0])
        for a in A:
            S_vec = np.array([inv0], dtype=float)
            X_vec = np.array([float(a)], dtype=float)
            W_vec = np.array([float(d0)], dtype=float)
            c0 = float(self.system.cost_func(S_vec, X_vec, W_vec, t))
            s_next = float(self.system.transition_func(S_vec, X_vec, W_vec, t)[0])
            jn = float(np.interp(s_next, Sg, J_next))
            v = c0 + jn
            if v < best:
                best = v
                best_a = float(a)

        return self._pack_action(best_a)


class HybridDlaVfaTerminalPolicy(Policy):
    """Hybrid #2 (DLA + VFA): deterministic lookahead + terminal value approximation."""

    def __init__(
        self,
        system,
        forecast: DemandPathForecaster,
        terminal_vfa: TerminalVFA,
        *,
        H: int = 5,
        buffer_k: float = 0.0,
        dx: int = 10,
        x_max: int = 500,
        s_max: int = 1000,
        s_grid_step: Optional[int] = None,
        p: float = 2.0,
        c: float = 0.5,
        h: float = 0.03,
        b: float = 1.0,
    ):
        self.system = system
        self.forecast = forecast
        self.terminal_vfa = terminal_vfa

        self.H = int(H)
        self.buffer_k = float(buffer_k)
        self.dx = int(dx)
        self.x_max = int(x_max)
        self.s_max = int(s_max)
        self.s_grid_step = int(s_grid_step) if s_grid_step is not None else int(dx)

        self.p = float(p)
        self.c = float(c)
        self.h = float(h)
        self.b = float(b)

        if self.H <= 0:
            raise ValueError("H must be >= 1")
        if self.dx <= 0:
            raise ValueError("dx must be positive")
        if self.s_grid_step <= 0:
            raise ValueError("s_grid_step must be positive")

        self._A = np.arange(0, self.x_max + self.dx, self.dx, dtype=float)
        self._Sgrid = np.arange(0, self.s_max + self.s_grid_step, self.s_grid_step, dtype=float)

    def _pack_action(self, x: float) -> Action:
        x = float(np.clip(x, 0.0, self.x_max))
        if self.dx > 1:
            x = self.dx * np.round(x / self.dx)
        return np.array([float(int(np.round(x)))], dtype=float)

    def _demand_hat_path(self, state: State, t: int, info: Optional[dict]) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(-1)
        mu = np.asarray(self.forecast.forecast_mean_path(state, int(t), self.H, info=info), dtype=float).reshape(-1)
        sig = np.asarray(self.forecast.forecast_std_path(state, int(t), self.H, info=info), dtype=float).reshape(-1)
        if mu.shape[0] != self.H or sig.shape[0] != self.H:
            raise ValueError("Forecaster returned wrong shape; expected length H")
        d = mu + self.buffer_k * sig
        return np.maximum(0.0, d).astype(float)

    def _fast_path_available(self) -> bool:
        try:
            return (
                getattr(self.system, "cost_func", None) is not None
                and getattr(self.system, "transition_func", None) is not None
                and getattr(self.system.cost_func, "__name__", "") == "inventory_cost"
                and getattr(self.system.transition_func, "__name__", "") == "inventory_transition"
            )
        except Exception:
            return False

    def _dp_vectorized(self, inv0: float, t0: int, demand_hat_path: np.ndarray) -> float:
        Sg = self._Sgrid
        A = self._A

        J_next = np.array([-float(self.terminal_vfa.V(float(s), int(t0) + self.H)) for s in Sg], dtype=float)

        for k in range(self.H - 1, -1, -1):
            d = float(demand_hat_path[k])

            on_hand = Sg[:, None] + A[None, :]
            inv_next = np.maximum(0.0, on_hand - d)

            sales = np.minimum(on_hand, d)
            lost = np.maximum(0.0, d - on_hand)
            cost = (self.c * A[None, :]) + (self.h * inv_next) + (self.b * lost) - (self.p * sales)

            idx = np.searchsorted(Sg, inv_next, side="left")
            idx = np.clip(idx, 0, len(Sg) - 1)
            idx0 = np.clip(idx - 1, 0, len(Sg) - 1)
            idx1 = idx
            s0 = Sg[idx0]
            s1 = Sg[idx1]
            denom = (s1 - s0)

            w = np.zeros_like(inv_next)
            mask = denom > 0
            w[mask] = (inv_next[mask] - s0[mask]) / denom[mask]

            J_interp = (1.0 - w) * J_next[idx0] + w * J_next[idx1]
            Q = cost + J_interp

            J_next = np.min(Q, axis=1)

        inv0 = float(np.clip(inv0, 0.0, float(self.s_max)))
        d0 = float(demand_hat_path[0])

        on_hand0 = inv0 + A
        inv_next0 = np.maximum(0.0, on_hand0 - d0)

        sales0 = np.minimum(on_hand0, d0)
        lost0 = np.maximum(0.0, d0 - on_hand0)
        cost0 = (self.c * A) + (self.h * inv_next0) + (self.b * lost0) - (self.p * sales0)

        idx = np.searchsorted(Sg, inv_next0, side="left")
        idx = np.clip(idx, 0, len(Sg) - 1)
        idx0 = np.clip(idx - 1, 0, len(Sg) - 1)
        idx1 = idx
        s0 = Sg[idx0]
        s1 = Sg[idx1]
        denom = (s1 - s0)

        w = np.zeros_like(inv_next0)
        mask = denom > 0
        w[mask] = (inv_next0[mask] - s0[mask]) / denom[mask]

        J_interp0 = (1.0 - w) * J_next[idx0] + w * J_next[idx1]
        Q0 = cost0 + J_interp0
        a_idx = int(np.argmin(Q0))
        return float(A[a_idx])

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        state = np.asarray(state, dtype=float).reshape(-1)
        inv0 = float(state[0])
        t = int(t)
        info = info or {}

        demand_hat_path = self._demand_hat_path(state, t, info)

        if self._fast_path_available():
            x0 = self._dp_vectorized(inv0, t, demand_hat_path)
            return self._pack_action(x0)

        Sg = self._Sgrid
        A = self._A

        J_next = np.array([-float(self.terminal_vfa.V(float(s), t + self.H)) for s in Sg], dtype=float)

        for k in range(self.H - 1, -1, -1):
            d = float(demand_hat_path[k])
            J = np.zeros_like(Sg, dtype=float)
            for i, s in enumerate(Sg):
                best = np.inf
                for a in A:
                    S_vec = np.array([float(s)], dtype=float)
                    X_vec = np.array([float(a)], dtype=float)
                    W_vec = np.array([float(d)], dtype=float)
                    c0 = float(self.system.cost_func(S_vec, X_vec, W_vec, t + k))
                    s_next = float(self.system.transition_func(S_vec, X_vec, W_vec, t + k)[0])
                    jn = float(np.interp(s_next, Sg, J_next))
                    best = min(best, c0 + jn)
                J[i] = best
            J_next = J

        best_a = 0.0
        best = np.inf
        d0 = float(demand_hat_path[0])
        for a in A:
            S_vec = np.array([inv0], dtype=float)
            X_vec = np.array([float(a)], dtype=float)
            W_vec = np.array([float(d0)], dtype=float)
            c0 = float(self.system.cost_func(S_vec, X_vec, W_vec, t))
            s_next = float(self.system.transition_func(S_vec, X_vec, W_vec, t)[0])
            jn = float(np.interp(s_next, Sg, J_next))
            v = c0 + jn
            if v < best:
                best = v
                best_a = float(a)

        return self._pack_action(best_a)


class HybridDlaRolloutBasePfaPolicy(Policy):
    """Hybrid #3 (DLA + PFA): rollout policy using a base policy after the first step."""

    def __init__(
        self,
        system,
        base_policy: Union[Policy, Callable[[State, int], Action]],
        *,
        H: int = 5,
        n_rollouts: int = 30,
        dx: int = 10,
        x_max: int = 500,
        decision_seed: int = 2026,
        candidate_radius_steps: int = 5,
        include_full_action_grid: bool = False,
        # cost params (match inventory_cost defaults)
        p: float = 2.0,
        c: float = 0.5,
        h: float = 0.03,
        b: float = 1.0,
    ):
        self.system = system
        self.base_policy = base_policy

        self.H = int(H)
        self.n_rollouts = int(n_rollouts)
        self.dx = int(dx)
        self.x_max = int(x_max)
        self.decision_seed = int(decision_seed)
        self.candidate_radius_steps = int(candidate_radius_steps)
        self.include_full_action_grid = bool(include_full_action_grid)

        self.p = float(p)
        self.c = float(c)
        self.h = float(h)
        self.b = float(b)

        if self.H <= 0:
            raise ValueError("H must be >= 1")
        if self.n_rollouts <= 0:
            raise ValueError("n_rollouts must be >= 1")
        if self.dx <= 0:
            raise ValueError("dx must be positive")
        if self.candidate_radius_steps < 0:
            raise ValueError("candidate_radius_steps must be >= 0")

        self._A = np.arange(0, self.x_max + self.dx, self.dx, dtype=float)

    def _pack_action(self, x: float) -> Action:
        x = float(np.clip(x, 0.0, self.x_max))
        if self.dx > 1:
            x = self.dx * np.round(x / self.dx)
        return np.array([float(int(np.round(x)))], dtype=float)

    def _base_act(self, state: State, t: int, info: Optional[dict]) -> Action:
        if hasattr(self.base_policy, "act"):
            return np.asarray(self.base_policy.act(state, t, info=info), dtype=float).reshape(-1)  # type: ignore[attr-defined]
        return np.asarray(self.base_policy(state, t), dtype=float).reshape(-1)

    def _fast_path_available(self) -> bool:
        try:
            return (
                getattr(self.system, "cost_func", None) is not None
                and getattr(self.system, "transition_func", None) is not None
                and getattr(self.system.cost_func, "__name__", "") == "inventory_cost"
                and getattr(self.system.transition_func, "__name__", "") == "inventory_transition"
                and hasattr(getattr(self.system, "exogenous_model", None), "lambda_t")
            )
        except Exception:
            return False

    def _candidate_actions(self, state: State, t: int, info: dict) -> np.ndarray:
        if self.include_full_action_grid or self.candidate_radius_steps == 0:
            return self._A.copy()

        x_base = float(self._base_act(state, t, info).reshape(-1)[0])
        x_base = float(np.clip(x_base, 0.0, self.x_max))
        if self.dx > 1:
            x_base = float(self.dx * np.round(x_base / self.dx))

        offsets = np.arange(-self.candidate_radius_steps, self.candidate_radius_steps + 1) * float(self.dx)
        cand = x_base + offsets
        cand = np.clip(cand, 0.0, float(self.x_max))
        cand = np.unique(cand.astype(float))
        return np.sort(cand)

    def _rollout_cost_generic(self, state0: State, t0: int, x0: Action, rng: np.random.Generator, info: dict) -> float:
        state = np.asarray(state0, dtype=float).reshape(-1).copy()
        total = 0.0

        for k in range(self.H):
            if k == 0:
                action = np.asarray(x0, dtype=float).reshape(-1)
            else:
                action = self._base_act(state, int(t0) + k, info)

            exog = self.system.exogenous_model.sample(state, action, int(t0) + k, rng)
            exog = np.asarray(exog, dtype=float).reshape(-1)

            total += float(self.system.cost_func(state, action, exog, int(t0) + k))
            state = np.asarray(self.system.transition_func(state, action, exog, int(t0) + k), dtype=float).reshape(-1)

        return float(total)

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        state = np.asarray(state, dtype=float).reshape(-1)
        t = int(t)
        info = info or {}

        crn_step_seed = info.get("crn_step_seed", 0)
        ss = np.random.SeedSequence([int(self.decision_seed), int(crn_step_seed), int(t)])
        rng = np.random.default_rng(ss)

        cand = self._candidate_actions(state, t, info)

        if self._fast_path_available():
            lam_by_k = np.array([float(self.system.exogenous_model.lambda_t(t + k)) for k in range(self.H)], dtype=float)
            D = np.stack([rng.poisson(lam, size=self.n_rollouts) for lam in lam_by_k], axis=1).astype(float)

            inv0 = float(state[0])

            can_vector_base = hasattr(self.base_policy, "target_level") or hasattr(self.base_policy, "S_bar")
            target = None
            x_max_base = None
            dx_base = None

            if hasattr(self.base_policy, "target_level"):
                target = float(self.base_policy.target_level)  # type: ignore[attr-defined]
                x_max_base = getattr(self.base_policy, "x_max", None)  # type: ignore[attr-defined]
                x_max_base = None if x_max_base is None else float(x_max_base)
                dx_base = getattr(self.base_policy, "dx", None)  # type: ignore[attr-defined]
                dx_base = None if dx_base is None else int(dx_base)
            elif hasattr(self.base_policy, "S_bar"):
                target = float(self.base_policy.S_bar)  # type: ignore[attr-defined]
                x_max_base = float(getattr(self.base_policy, "x_max", self.x_max))  # type: ignore[attr-defined]
                dx_base = int(getattr(self.base_policy, "dx", self.dx))  # type: ignore[attr-defined]

            def base_order(inv_vec: np.ndarray) -> np.ndarray:
                if not can_vector_base or target is None:
                    out = np.empty_like(inv_vec)
                    for i, inv_i in enumerate(inv_vec):
                        out[i] = float(self._base_act(np.array([inv_i], dtype=float), t, info)[0])
                    return out

                x = np.maximum(0.0, target - inv_vec)
                if x_max_base is not None:
                    x = np.minimum(x, x_max_base)
                if dx_base is not None and dx_base > 1:
                    x = dx_base * np.round(x / dx_base)
                    x = np.maximum(0.0, x)
                    if x_max_base is not None:
                        x = np.minimum(x, x_max_base)
                return x.astype(float)

            best_x = self._pack_action(0.0)
            best_val = np.inf

            for x0 in cand:
                inv = np.full(self.n_rollouts, inv0, dtype=float)
                total = np.zeros(self.n_rollouts, dtype=float)

                for k in range(self.H):
                    if k == 0:
                        order_vec = np.full(self.n_rollouts, float(x0), dtype=float)
                    else:
                        order_vec = base_order(inv)

                    demand = D[:, k]
                    on_hand = inv + order_vec

                    sales = np.minimum(on_hand, demand)
                    lost = np.maximum(0.0, demand - on_hand)
                    inv = np.maximum(0.0, on_hand - demand)

                    total += (self.c * order_vec) + (self.h * inv) + (self.b * lost) - (self.p * sales)

                v = float(total.mean())
                if v < best_val:
                    best_val = v
                    best_x = self._pack_action(float(x0))

            return best_x

        # Generic fallback: use CRN across candidates by reusing rollout seeds.
        rollout_seeds = rng.integers(1, 2**31 - 1, size=self.n_rollouts, dtype=np.int64)

        best_x = self._pack_action(float(cand[0]) if len(cand) else 0.0)
        best_val = np.inf

        for x0 in cand:
            x0_vec = self._pack_action(float(x0))
            vals = []
            for seed in rollout_seeds:
                rng_r = np.random.default_rng(int(seed))
                vals.append(self._rollout_cost_generic(state, t, x0_vec, rng_r, info))
            v = float(np.mean(vals))
            if v < best_val:
                best_val = v
                best_x = x0_vec

        return best_x


# Notebook-compatibility aliases (Lecture 15 naming)
Hybrid_DLA_CFA_BufferedForecast = HybridDlaCfaBufferedForecastPolicy
Hybrid_DLA_VFA_Terminal = HybridDlaVfaTerminalPolicy
Hybrid_DLA_Rollout_BasePFA = HybridDlaRolloutBasePfaPolicy
TerminalVFA_Linear = TerminalVfaLinear


__all__ = [
    "TerminalVFA",
    "TerminalVfaLinear",
    "TerminalVFA_Linear",
    "HybridDlaCfaBufferedForecastPolicy",
    "HybridDlaVfaTerminalPolicy",
    "HybridDlaRolloutBasePfaPolicy",
    "Hybrid_DLA_CFA_BufferedForecast",
    "Hybrid_DLA_VFA_Terminal",
    "Hybrid_DLA_Rollout_BasePFA",
]
