from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import numpy as np
from inventory.core.policy import Policy
from inventory.core.types import Action, State


def poisson_pmf_truncated(lam: float, D_max: int) -> Tuple[np.ndarray, float]:
    """Compute a truncated Poisson PMF plus the remaining tail mass.

    Returns:
      - pmf[d] for d=0..D_max
      - tail_prob = P(D > D_max)

    This is a dependency-free implementation (no SciPy).
    """

    lam = float(max(0.0, lam))
    D_max = int(D_max)

    pmf = np.empty(D_max + 1, dtype=float)
    pmf[0] = math.exp(-lam)
    for d in range(1, D_max + 1):
        pmf[d] = pmf[d - 1] * (lam / d)

    tail = max(0.0, 1.0 - float(pmf.sum()))
    return pmf, float(tail)


def auto_D_max_from_lam(lam: float, z: float = 6.0, extra: int = 10) -> int:
    """Conservative truncation heuristic: D_max ≈ lam + z*sqrt(lam) + extra."""

    lam = float(max(0.0, lam))
    return int(math.ceil(lam + float(z) * math.sqrt(max(lam, 1e-9)) + int(extra)))


def poisson_ppf(lam: float, q: float, d_max_cap: int = 10000) -> int:
    """Dependency-free Poisson quantile: smallest d with P(D<=d) >= q."""

    lam = float(max(0.0, lam))
    q = float(min(max(q, 0.0), 1.0))
    d_max_cap = int(d_max_cap)

    if q <= 0.0:
        return 0
    if q >= 1.0:
        return int(min(d_max_cap, math.ceil(lam + 10.0 * math.sqrt(max(lam, 1e-9)) + 50.0)))

    p = math.exp(-lam)  # P(D=0)
    cdf = p
    d = 0
    while cdf < q and d < d_max_cap:
        d += 1
        p *= lam / d
        cdf += p
    return int(d)


def poisson_scenarios_from_quantile_bins(
    lam: float,
    K: int,
    q_lo: float = 1e-4,
    q_hi: float = 1.0 - 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scenario approximation for Poisson demand.

    - Split [q_lo,q_hi] into K equal-probability bins
    - Pick the mid-quantile in each bin as representative demand
    - Use equal weights within [q_lo,q_hi]; push tails into the first/last bin
    """

    K = int(K)
    if K <= 0:
        raise ValueError("K must be >= 1")
    if not (0.0 <= q_lo < q_hi <= 1.0):
        raise ValueError("Require 0 <= q_lo < q_hi <= 1")

    edges = np.linspace(float(q_lo), float(q_hi), K + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])

    d = np.array([poisson_ppf(lam, float(q)) for q in mids], dtype=int)

    w = np.full(K, (float(q_hi) - float(q_lo)) / float(K), dtype=float)
    w[0] += float(q_lo)
    w[-1] += (1.0 - float(q_hi))
    w /= float(w.sum())

    return d, w


class SeasonalLambda(Protocol):
    def lambda_t(self, t: int) -> float:  # pragma: no cover
        ...


class RegimeLambda(Protocol):
    P: np.ndarray
    R: int

    def lambda_t_regime(self, t: int, r: int) -> float:  # pragma: no cover
        ...


class DynamicProgrammingPolicy(Policy):
    """Policy derived from an optimal action table x_star[t, inv]."""

    def __init__(self, x_star: np.ndarray, *, S_max: int, dx: int, x_max: int):
        self.x_star = np.asarray(x_star, dtype=int)  # (T, S_max+1)
        self.T = int(self.x_star.shape[0])
        self.S_max = int(S_max)
        self.dx = int(dx)
        self.x_max = int(x_max)

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        inv = int(np.clip(int(round(float(state[0]))), 0, self.S_max))
        t = int(t)
        x = 0 if (t < 0 or t >= self.T) else int(self.x_star[t, inv])

        # safety clamps
        x = max(0, min(int(x), self.x_max))
        if self.dx > 1:
            x = self.dx * int(round(x / self.dx))
        x = min(x, self.S_max - inv)
        return np.array([float(x)], dtype=float)


@dataclass(frozen=True)
class DPSolution1D:
    V: np.ndarray  # (T+1, S_max+1)
    x_star: np.ndarray  # (T, S_max+1)
    meta: Dict


class DynamicProgrammingSolver1D:
    """Finite-horizon DP for 1D inventory with capacity and batched actions.

    expectation_mode:
      - "truncation": sum_{d=0..Dmax} + tail lump
      - "scenarios":  K scenario demands with weights

    Exogenous model must provide `lambda_t(t)`.
    """

    def __init__(
        self,
        exog_seasonal: SeasonalLambda,
        *,
        T: int,
        S_max: int,
        x_max: int,
        dx: int,
        p: float,
        c: float,
        h: float,
        b: float,
        terminal_cost_per_unit: float,
        gamma: float = 1.0,
        expectation_mode: str = "truncation",
        D_max: Optional[int] = None,
        K: int = 15,
        q_lo: float = 1e-4,
        q_hi: float = 1.0 - 1e-4,
    ):
        if not hasattr(exog_seasonal, "lambda_t"):
            raise ValueError("exogenous_model must have lambda_t(t).")

        self.exog = exog_seasonal
        self.T = int(T)
        self.S_max = int(S_max)
        self.x_max = int(x_max)
        self.dx = int(dx)

        self.p, self.c, self.h, self.b = float(p), float(c), float(h), float(b)
        self.terminal_cost_per_unit = float(terminal_cost_per_unit)
        self.gamma = float(gamma)

        self.expectation_mode = str(expectation_mode).lower().strip()
        if self.expectation_mode not in {"truncation", "scenarios"}:
            raise ValueError("expectation_mode must be 'truncation' or 'scenarios'.")

        self.D_max = D_max
        self.K = int(K)
        self.q_lo = float(q_lo)
        self.q_hi = float(q_hi)

        self.inv = np.arange(self.S_max + 1, dtype=int)
        self.actions = np.arange(0, self.x_max + 1, self.dx, dtype=int)

        # cache demand support per t
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}

    def _demand_support(self, t: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Returns (d, w, tail) for time t.

        - truncation: d=0..Dmax, w=pmf, tail>0
        - scenarios: d=scenarios, w=weights, tail=0
        """

        t = int(t)
        if t in self._cache:
            return self._cache[t]

        lam = float(self.exog.lambda_t(t))

        if self.expectation_mode == "truncation":
            Dmax = int(self.D_max) if self.D_max is not None else auto_D_max_from_lam(lam)
            pmf, tail = poisson_pmf_truncated(lam, Dmax)
            d = np.arange(Dmax + 1, dtype=int)
            w = pmf.astype(float)
            self._cache[t] = (d, w, float(tail))
            return d, w, float(tail)

        d, w = poisson_scenarios_from_quantile_bins(lam, K=self.K, q_lo=self.q_lo, q_hi=self.q_hi)
        self._cache[t] = (d, w, 0.0)
        return d, w, 0.0

    def solve(self) -> DPSolution1D:
        T, S_max = self.T, self.S_max
        V = np.zeros((T + 1, S_max + 1), dtype=float)
        x_star = np.zeros((T, S_max + 1), dtype=int)

        if self.terminal_cost_per_unit != 0.0:
            V[T, :] = self.terminal_cost_per_unit * self.inv.astype(float)

        inv = self.inv

        for t in reversed(range(T)):
            Vnext = V[t + 1, :]

            d, w, tail = self._demand_support(t)
            d_row = d[None, :]
            w_row = w[None, :]

            best_val = np.full(S_max + 1, np.inf, dtype=float)
            best_x = np.zeros(S_max + 1, dtype=int)

            for x in self.actions:
                feas = x <= (S_max - inv)
                if not np.any(feas):
                    continue

                on_hand = inv + x
                oh = on_hand[:, None]

                sales = np.minimum(oh, d_row).astype(float)
                inv_end = np.maximum(oh - d_row, 0.0).astype(int)
                inv_end = np.clip(inv_end, 0, S_max)
                lost = np.maximum(d_row - oh, 0.0).astype(float)

                immediate = (self.c * float(x)) + (self.h * inv_end.astype(float)) + (self.b * lost) - (self.p * sales)
                cont = self.gamma * Vnext[inv_end]
                exp = np.sum((immediate + cont) * w_row, axis=1)

                if tail > 0.0 and self.expectation_mode == "truncation":
                    Dmax = int(d[-1])
                    d_tail = Dmax + 1

                    sales_t = np.minimum(on_hand, d_tail).astype(float)
                    inv_end_t = np.maximum(on_hand - d_tail, 0).astype(int)
                    inv_end_t = np.clip(inv_end_t, 0, S_max)
                    lost_t = np.maximum(d_tail - on_hand, 0.0).astype(float)

                    immediate_t = (self.c * float(x)) + (self.h * inv_end_t.astype(float)) + (self.b * lost_t) - (self.p * sales_t)
                    cont_t = self.gamma * Vnext[inv_end_t]
                    exp = exp + float(tail) * (immediate_t + cont_t)

                exp = np.where(feas, exp, np.inf)
                improve = exp < best_val
                best_val[improve] = exp[improve]
                best_x[improve] = int(x)

            V[t, :] = best_val
            x_star[t, :] = best_x

        meta = dict(
            expectation_mode=self.expectation_mode,
            D_max=self.D_max,
            K=self.K,
            q_lo=self.q_lo,
            q_hi=self.q_hi,
            T=self.T,
            S_max=self.S_max,
            x_max=self.x_max,
            dx=self.dx,
        )
        return DPSolution1D(V=V, x_star=x_star, meta=meta)

    def to_policy(self, sol: DPSolution1D) -> DynamicProgrammingPolicy:
        return DynamicProgrammingPolicy(sol.x_star, S_max=self.S_max, dx=self.dx, x_max=self.x_max)


class DynamicProgrammingPolicyRegime(Policy):
    """Policy derived from an optimal action table x_star[t, inv, r]."""

    def __init__(self, x_star: np.ndarray, *, S_max: int, R: int, dx: int, x_max: int):
        self.x_star = np.asarray(x_star, dtype=int)  # (T, S_max+1, R)
        self.T = int(self.x_star.shape[0])
        self.S_max = int(S_max)
        self.R = int(R)
        self.dx = int(dx)
        self.x_max = int(x_max)

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        inv = int(np.clip(int(round(float(state[0]))), 0, self.S_max))
        r = int(np.clip(int(round(float(state[1]))), 0, self.R - 1))
        t = int(t)
        x = 0 if (t < 0 or t >= self.T) else int(self.x_star[t, inv, r])

        x = max(0, min(int(x), self.x_max))
        if self.dx > 1:
            x = self.dx * int(round(x / self.dx))
        x = min(x, self.S_max - inv)
        return np.array([float(x)], dtype=float)


@dataclass(frozen=True)
class DPSolutionRegime:
    V: np.ndarray  # (T+1, S_max+1, R)
    x_star: np.ndarray  # (T, S_max+1, R)


class DPSolverRegimeScenarios:
    """DP on (inventory, regime) with scenario approximation of Poisson demand.

    Exogenous model must provide:
      - `P` transition matrix (R,R)
      - `R` number of regimes
      - `lambda_t_regime(t, r)`
    """

    def __init__(
        self,
        exog_reg: RegimeLambda,
        *,
        T: int,
        S_max: int,
        x_max: int,
        dx: int,
        p: float,
        c: float,
        h: float,
        b: float,
        terminal_cost_per_unit: float,
        gamma: float = 1.0,
        K: int = 15,
        q_lo: float = 1e-4,
        q_hi: float = 1.0 - 1e-4,
    ):
        self.exog = exog_reg
        self.P = np.asarray(exog_reg.P, dtype=float)
        self.R = int(exog_reg.R)

        self.T = int(T)
        self.S_max = int(S_max)
        self.x_max = int(x_max)
        self.dx = int(dx)
        self.p, self.c, self.h, self.b = float(p), float(c), float(h), float(b)
        self.gamma = float(gamma)
        self.terminal_cost_per_unit = float(terminal_cost_per_unit)

        self.K = int(K)
        self.q_lo = float(q_lo)
        self.q_hi = float(q_hi)

        self.inv = np.arange(self.S_max + 1, dtype=int)
        self.actions = np.arange(0, self.x_max + 1, self.dx, dtype=int)

        self._cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    def _scenarios(self, t: int, r: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (int(t), int(r))
        if key in self._cache:
            return self._cache[key]
        lam = float(self.exog.lambda_t_regime(t, r))
        d, w = poisson_scenarios_from_quantile_bins(lam, K=self.K, q_lo=self.q_lo, q_hi=self.q_hi)
        self._cache[key] = (d, w)
        return d, w

    def solve(self) -> DPSolutionRegime:
        T, S_max, R = self.T, self.S_max, self.R
        V = np.zeros((T + 1, S_max + 1, R), dtype=float)
        x_star = np.zeros((T, S_max + 1, R), dtype=int)

        if self.terminal_cost_per_unit != 0.0:
            V[T, :, :] = self.terminal_cost_per_unit * self.inv.astype(float)[:, None]

        inv = self.inv

        for t in reversed(range(T)):
            Vnext = V[t + 1, :, :]  # (S,R)

            for r in range(R):
                Vbar = Vnext @ self.P[r, :]  # (S,)

                d, w = self._scenarios(t, r)
                d_row = d[None, :]
                w_row = w[None, :]

                best_val = np.full(S_max + 1, np.inf, dtype=float)
                best_x = np.zeros(S_max + 1, dtype=int)

                for x in self.actions:
                    feas = x <= (S_max - inv)
                    if not np.any(feas):
                        continue

                    on_hand = inv + x
                    oh = on_hand[:, None]

                    sales = np.minimum(oh, d_row).astype(float)
                    inv_end = np.maximum(oh - d_row, 0.0).astype(int)
                    inv_end = np.clip(inv_end, 0, S_max)
                    lost = np.maximum(d_row - oh, 0.0).astype(float)

                    immediate = (self.c * float(x)) + (self.h * inv_end.astype(float)) + (self.b * lost) - (self.p * sales)
                    cont = self.gamma * Vbar[inv_end]
                    exp = np.sum((immediate + cont) * w_row, axis=1)

                    exp = np.where(feas, exp, np.inf)
                    improve = exp < best_val
                    best_val[improve] = exp[improve]
                    best_x[improve] = int(x)

                V[t, :, r] = best_val
                x_star[t, :, r] = best_x

        return DPSolutionRegime(V=V, x_star=x_star)

    def to_policy(self, sol: DPSolutionRegime) -> DynamicProgrammingPolicyRegime:
        return DynamicProgrammingPolicyRegime(sol.x_star, S_max=self.S_max, R=self.R, dx=self.dx, x_max=self.x_max)


__all__ = [
    # Poisson expectation helpers
    "poisson_pmf_truncated",
    "auto_D_max_from_lam",
    "poisson_ppf",
    "poisson_scenarios_from_quantile_bins",
    # 1D DP
    "DynamicProgrammingPolicy",
    "DPSolution1D",
    "DynamicProgrammingSolver1D",
    # Regime DP (scenario approximation)
    "DynamicProgrammingPolicyRegime",
    "DPSolutionRegime",
    "DPSolverRegimeScenarios",
]
