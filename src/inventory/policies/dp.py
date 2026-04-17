from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import numpy as np
from inventory.core.policy import Policy
from inventory.core.types import Action, State


def _one_hot(i: int, K: int) -> np.ndarray:
    v = np.zeros(int(K), dtype=float)
    v[int(np.clip(int(i), 0, int(K) - 1))] = 1.0
    return v


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
    """Table-based DP policy for the 1D inventory problem.

        This policy is the *output format* of :class:`DynamicProgrammingSolver1D`.
        The solver computes an optimal action table `x_star[t, inv]`, and this policy
        simply looks up the action for the current time and inventory.

        Notes for students:
        - This is an *offline* policy: the optimization is done once (in DP), then
            the policy is just a fast table lookup.
        - The implementation includes safety clamps so that even if a table entry is
            out-of-range, the returned action remains feasible.
        """

    def __init__(self, x_star: np.ndarray, *, S_max: int, dx: int, x_max: int):
        self.x_star = np.asarray(x_star, dtype=int)  # (T, S_max+1)
        self.T = int(self.x_star.shape[0])
        self.S_max = int(S_max)
        self.dx = int(dx)
        self.x_max = int(x_max)

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        """Return the (batched) order quantity action.

        Args:
          state: expected to contain inventory at index 0.
          t: time index.
          info: unused here; included for `Policy` interface compatibility.
        """

        # inventory is stored in state[0] for these teaching problems
        inv = int(np.clip(int(round(float(state[0]))), 0, self.S_max))
        t = int(t)

        # If t is out of bounds, do nothing (x=0). Otherwise, lookup the action.
        x = 0 if (t < 0 or t >= self.T) else int(self.x_star[t, inv])

        # Safety clamps (keep action feasible and aligned with dx-grid)
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
    """Finite-horizon dynamic programming (DP) solver for 1D inventory.

        This solver computes:
            - a value table `V[t, inv]` for t=0..T and inv=0..S_max
            - an optimal action table `x_star[t, inv]` for t=0..T-1

        The objective is *cost minimization*.

        Expectation over Poisson demand is approximated in one of two ways:
            - expectation_mode="truncation": exact sum over d=0..D_max plus one tail term
            - expectation_mode="scenarios": quantile-bin scenario approximation with K scenarios

        Required exogenous interface:
            - `lambda_t(t) -> float` (Poisson mean at time t)

        Implementation note:
        - The inner loops are vectorized over all inventory levels `inv=0..S_max`.
            For each action x, we evaluate its expected cost for *every* inventory
            level at once, then take an argmin.
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
        # Discrete action grid: 0, dx, 2dx, ..., up to x_max
        self.actions = np.arange(0, self.x_max + 1, self.dx, dtype=int)

        # cache demand support per t
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}

    def __repr__(self) -> str:
        exog_name = type(self.exog).__name__ if self.exog is not None else None
        return (
            "DynamicProgrammingSolver1D("
            f"exog_seasonal={exog_name!r}, T={self.T}, S_max={self.S_max}, "
            f"x_max={self.x_max}, dx={self.dx}, p={self.p}, c={self.c}, "
            f"h={self.h}, b={self.b}, terminal_cost_per_unit={self.terminal_cost_per_unit}, "
            f"gamma={self.gamma}, expectation_mode={self.expectation_mode!r}, "
            f"D_max={self.D_max}, K={self.K})"
        )

    def _demand_support(self, t: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Returns (d, w, tail) for time t.

        - truncation: d=0..Dmax, w=pmf, tail>0
        - scenarios: d=scenarios, w=weights, tail=0

        This method is cached per time t because demand support depends only on
        lambda_t(t), not on inventory or action.
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
        """Run backward induction and return DP tables.

        Returns:
            `DPSolution1D` with:
                - V: shape (T+1, S_max+1)
                - x_star: shape (T, S_max+1)
        """

        T, S_max = self.T, self.S_max
        V = np.zeros((T + 1, S_max + 1), dtype=float)
        x_star = np.zeros((T, S_max + 1), dtype=int)

        if self.terminal_cost_per_unit != 0.0:
            V[T, :] = self.terminal_cost_per_unit * self.inv.astype(float)

        inv = self.inv  # vector of all inventory levels 0..S_max

        for t in reversed(range(T)):
            Vnext = V[t + 1, :]

            d, w, tail = self._demand_support(t)
            d_row = d[None, :]
            w_row = w[None, :]

            # For each inventory level, track the best value and argmin action.
            best_val = np.full(S_max + 1, np.inf, dtype=float)
            best_x = np.zeros(S_max + 1, dtype=int)

            for x in self.actions:
                # `feas[i]` says whether action x is feasible at inventory inv[i]
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

    This is the natural extension of :class:`DynamicProgrammingSolver1D` when demand
    is driven by a Markov regime.

    State:
        - inventory level ``inv`` in ``{0,1,...,S_max}``
        - regime index ``r`` in ``{0,1,...,R-1}``

    The Bellman recursion is cost-minimizing.

    Expectation is split into two parts:
        1) Demand uncertainty: approximated with ``K`` Poisson scenarios.
        2) Regime uncertainty: exact expectation using the transition matrix ``P``.

    Required exogenous interface:
        - ``P`` transition matrix of shape (R, R), where ``P[r, r_next]`` is the
          probability of transitioning to ``r_next`` given current regime ``r``.
        - ``R`` number of regimes.
        - ``lambda_t_regime(t, r)`` returning the Poisson mean at time ``t`` in regime ``r``.
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
        """Return scenario demands and weights for a specific (t, r).

        This is cached because it depends only on ``lambda_t_regime(t, r)``.

        Returns:
            (d, w) where:
              - d: integer demands, shape (K,)
              - w: probabilities/weights, shape (K,), sum to 1
        """

        key = (int(t), int(r))
        if key in self._cache:
            return self._cache[key]
        lam = float(self.exog.lambda_t_regime(t, r))
        d, w = poisson_scenarios_from_quantile_bins(lam, K=self.K, q_lo=self.q_lo, q_hi=self.q_hi)
        self._cache[key] = (d, w)
        return d, w

    def solve(self) -> DPSolutionRegime:
        """Run backward induction over time, inventory, and regimes.

        Returns:
            A :class:`DPSolutionRegime` containing:
              - ``V`` value table with shape (T+1, S_max+1, R)
              - ``x_star`` optimal action table with shape (T, S_max+1, R)
        """

        T, S_max, R = self.T, self.S_max, self.R
        V = np.zeros((T + 1, S_max + 1, R), dtype=float)
        x_star = np.zeros((T, S_max + 1, R), dtype=int)

        if self.terminal_cost_per_unit != 0.0:
            V[T, :, :] = self.terminal_cost_per_unit * self.inv.astype(float)[:, None]

        inv = self.inv

        for t in reversed(range(T)):
            Vnext = V[t + 1, :, :]  # (S,R)

            for r in range(R):
                # Expected next-step value given current regime r:
                #   Vbar[s] = E[ Vnext[s, r_next] | r ] = sum_{r_next} Vnext[s, r_next] * P[r, r_next]
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
        """Wrap an optimal action table as a `Policy` object."""
        return DynamicProgrammingPolicyRegime(sol.x_star, S_max=self.S_max, R=self.R, dx=self.dx, x_max=self.x_max)


class DynamicProgrammingPolicyMultiRegime(Policy):
    """DP policy for the multi-regime inventory model.

    State convention (default):
      S = [inventory, season, day, weather]

    The DP table is stored over a *collapsed* regime index:
      r = (season * Kd + day) * Kw + weather
    """

    def __init__(
        self,
        x_star: np.ndarray,
        *,
        S_max: int,
        Ks: int,
        Kd: int,
        Kw: int,
        dx: int,
        x_max: int,
        season_index: int = 1,
        day_index: int = 2,
        weather_index: int = 3,
    ):
        self.x_star = np.asarray(x_star, dtype=int)  # (T, S_max+1, R)
        self.T = int(self.x_star.shape[0])
        self.S_max = int(S_max)
        self.Ks = int(Ks)
        self.Kd = int(Kd)
        self.Kw = int(Kw)
        self.R = int(self.Ks * self.Kd * self.Kw)
        self.dx = int(dx)
        self.x_max = int(x_max)
        self.season_index = int(season_index)
        self.day_index = int(day_index)
        self.weather_index = int(weather_index)

        if self.x_star.ndim != 3 or self.x_star.shape[1] != (self.S_max + 1) or self.x_star.shape[2] != self.R:
            raise ValueError(
                f"x_star must have shape (T, S_max+1, R)=(*,{self.S_max+1},{self.R}). Got {self.x_star.shape}."
            )

    def __repr__(self) -> str:
        return (
            "DynamicProgrammingPolicyMultiRegime("
            f"T={self.T}, S_max={self.S_max}, Ks={self.Ks}, Kd={self.Kd}, Kw={self.Kw}, "
            f"R={self.R}, dx={self.dx}, x_max={self.x_max}, season_index={self.season_index}, "
            f"day_index={self.day_index}, weather_index={self.weather_index})"
        )

    def _collapse_regime(self, state: State) -> int:
        S = np.asarray(state, dtype=float).reshape(-1)
        season = int(np.round(float(S[self.season_index])))
        day = int(np.round(float(S[self.day_index])))
        weather = int(np.round(float(S[self.weather_index])))

        season = int(np.clip(season, 0, self.Ks - 1))
        day = int(np.clip(day, 0, self.Kd - 1))
        weather = int(np.clip(weather, 0, self.Kw - 1))
        return int((season * self.Kd + day) * self.Kw + weather)

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        inv = int(np.clip(int(round(float(np.asarray(state, dtype=float).reshape(-1)[0]))), 0, self.S_max))
        r = int(self._collapse_regime(state))
        t = int(t)

        x = 0 if (t < 0 or t >= self.T) else int(self.x_star[t, inv, r])

        x = max(0, min(int(x), self.x_max))
        if self.dx > 1:
            x = self.dx * int(round(x / self.dx))
        x = min(x, self.S_max - inv)
        return np.array([float(x)], dtype=float)


class DPSolverMultiRegimeScenarios:
    """DP on (inventory, season, day, weather) via collapsed regime index.

    This solver is designed for `PoissonMultiRegimeDemand` / `ExogenousPoissonMultiRegime`.

    Key point vs `DPSolverRegimeScenarios`:
    - The season transition can be time-dependent (season updates only every `season_period` steps).
      We incorporate this exactly in the regime expectation step.

    Demand model used in DP (mean-matching):
      D_{t+1} ~ Poisson(lam_eff),
      lam_eff = E[ lambda_base_season[R_season,t+1] ] * (1 + E[coeff_day[R_day,t+1]] + E[coeff_weather[R_weather,t+1]])
    where each expectation is taken over the *next* regime distribution conditional on current regimes.
    """

    def __init__(
        self,
        exo_multi,
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
        # Import here to avoid a hard dependency at module import time.
        from inventory.problems.demand_models import PoissonMultiRegimeDemand

        if not isinstance(exo_multi, PoissonMultiRegimeDemand):
            raise ValueError("exo_multi must be a PoissonMultiRegimeDemand / ExogenousPoissonMultiRegime instance")

        self.exo = exo_multi
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

        self.P_season = np.asarray(self.exo.P_season, dtype=float)
        self.P_day = np.asarray(self.exo.P_day, dtype=float)
        self.P_weather = np.asarray(self.exo.P_weather, dtype=float)
        self.base = np.asarray(self.exo.lambda_base_season, dtype=float).reshape(-1)
        self.coeff_day = np.asarray(self.exo.lambda_coeff_day, dtype=float).reshape(-1)
        self.coeff_weather = np.asarray(self.exo.lambda_coeff_weather, dtype=float).reshape(-1)
        self.lam_min = float(getattr(self.exo, "lam_min", 0.0))

        self.Ks = int(self.P_season.shape[0])
        self.Kd = int(self.P_day.shape[0])
        self.Kw = int(self.P_weather.shape[0])
        self.R = int(self.Ks * self.Kd * self.Kw)

        self.season_index = int(getattr(self.exo, "season_index", 1))
        self.day_index = int(getattr(self.exo, "day_index", 2))
        self.weather_index = int(getattr(self.exo, "weather_index", 3))

        self.season_period = int(getattr(self.exo, "season_period", 0))

        self.inv = np.arange(self.S_max + 1, dtype=int)
        self.actions = np.arange(0, self.x_max + 1, self.dx, dtype=int)

        # Precompute conditional expectations for day/weather next regime effects.
        self._E_cd = (self.P_day @ self.coeff_day).astype(float)  # (Kd,)
        self._E_cw = (self.P_weather @ self.coeff_weather).astype(float)  # (Kw,)
        self._E_base_update = (self.P_season @ self.base).astype(float)  # (Ks,)

        # Cache Poisson scenarios per (t, r)
        self._cache_scenarios: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    def __repr__(self) -> str:
        exog_name = type(self.exo).__name__ if self.exo is not None else None
        return (
            "DPSolverMultiRegimeScenarios("
            f"exo_multi={exog_name!r}, T={self.T}, S_max={self.S_max}, "
            f"x_max={self.x_max}, dx={self.dx}, p={self.p}, c={self.c}, "
            f"h={self.h}, b={self.b}, terminal_cost_per_unit={self.terminal_cost_per_unit}, "
            f"gamma={self.gamma}, K={self.K}, q_lo={self.q_lo}, q_hi={self.q_hi}, "
            f"season_period={self.season_period}, R={self.R})"
        )

    def _season_updates_at_next_step(self, t: int) -> bool:
        t = int(t)
        if self.season_period <= 0:
            return True
        return ((t + 1) % self.season_period) == 0

    def _unpack_r(self, r: int) -> tuple[int, int, int]:
        r = int(np.clip(int(r), 0, self.R - 1))
        season = r // (self.Kd * self.Kw)
        rem = r % (self.Kd * self.Kw)
        day = rem // self.Kw
        weather = rem % self.Kw
        return int(season), int(day), int(weather)

    def _lambda_mean(self, t: int, r: int) -> float:
        season, day, weather = self._unpack_r(r)

        if self._season_updates_at_next_step(t):
            E_base = float(self._E_base_update[season])
        else:
            E_base = float(self.base[season])

        lam = E_base * (1.0 + float(self._E_cd[day]) + float(self._E_cw[weather]))
        return float(max(self.lam_min, lam))

    def _scenarios(self, t: int, r: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (int(t), int(r))
        if key in self._cache_scenarios:
            return self._cache_scenarios[key]

        lam = float(self._lambda_mean(int(t), int(r)))
        d, w = poisson_scenarios_from_quantile_bins(lam, K=self.K, q_lo=self.q_lo, q_hi=self.q_hi)
        self._cache_scenarios[key] = (d, w)
        return d, w

    def _expected_next_value_all_regimes(self, Vnext: np.ndarray, t: int) -> np.ndarray:
        """Compute Vbar[s, r] = E[ Vnext[s, r_next] | current regime r ] for all r.

        Uses the independence structure (season/day/weather) to avoid O(R^2) work.
        """

        Vnext = np.asarray(Vnext, dtype=float)
        if Vnext.shape != (self.S_max + 1, self.R):
            raise ValueError(f"Vnext must have shape ({self.S_max+1},{self.R}). Got {Vnext.shape}.")

        V4 = Vnext.reshape(self.S_max + 1, self.Ks, self.Kd, self.Kw)

        # Weather: output is indexed by current weather; sum over next weather
        Vw = np.tensordot(V4, self.P_weather.T, axes=([3], [0]))  # (S, Ks, Kd, Kw)

        # Day: output indexed by current day; sum over next day
        Vd = np.tensordot(Vw, self.P_day.T, axes=([2], [0]))  # (S, Ks, Kw, Kd)
        Vd = np.transpose(Vd, (0, 1, 3, 2))  # (S, Ks, Kd, Kw)

        # Season: possibly time-dependent update rule
        if self._season_updates_at_next_step(t):
            Vs = np.tensordot(Vd, self.P_season.T, axes=([1], [0]))  # (S, Kd, Kw, Ks)
            Vs = np.transpose(Vs, (0, 3, 1, 2))  # (S, Ks, Kd, Kw)
        else:
            Vs = Vd

        return Vs.reshape(self.S_max + 1, self.R)

    def solve(self) -> DPSolutionRegime:
        T, S_max, R = self.T, self.S_max, self.R
        V = np.zeros((T + 1, S_max + 1, R), dtype=float)
        x_star = np.zeros((T, S_max + 1, R), dtype=int)

        if self.terminal_cost_per_unit != 0.0:
            V[T, :, :] = self.terminal_cost_per_unit * self.inv.astype(float)[:, None]

        inv = self.inv

        for t in reversed(range(T)):
            Vnext = V[t + 1, :, :]  # (S,R)
            Vbar_all = self._expected_next_value_all_regimes(Vnext, t)  # (S,R)

            for r in range(R):
                Vbar = Vbar_all[:, r]  # (S,)

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

    def to_policy(self, sol: DPSolutionRegime) -> DynamicProgrammingPolicyMultiRegime:
        return DynamicProgrammingPolicyMultiRegime(
            sol.x_star,
            S_max=self.S_max,
            Ks=self.Ks,
            Kd=self.Kd,
            Kw=self.Kw,
            dx=self.dx,
            x_max=self.x_max,
            season_index=self.season_index,
            day_index=self.day_index,
            weather_index=self.weather_index,
        )


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
    # Multi-regime DP (season/day/weather) with time-dependent season transitions
    "DynamicProgrammingPolicyMultiRegime",
    "DPSolverMultiRegimeScenarios",
]
