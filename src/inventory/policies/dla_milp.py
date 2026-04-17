from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from inventory.core.dynamics import DynamicSystemMVP
from inventory.core.policy import Policy
from inventory.core.types import Action, State
from inventory.forecasters.base import DemandForecaster
from inventory.problems.demand_models import ExogenousPoissonRegime, ExogenousPoissonSeasonal


@dataclass
class DlaMpcMilpPolicy(Policy):
    """Direct Lookahead Approximation (DLA) via a deterministic surrogate MILP (MPC style).

    At each time t:
      1) Build a deterministic demand mean path mu[t:t+H)
      2) Solve a small MILP for orders over the horizon (lost-sales, immediate delivery)
      3) Execute ONLY the first action x_0

    This is a refactor of `DLA_MPC_MILP` from `minimal_baseline.py`.

    Notes:
    - Uses `scipy.optimize.milp` if available; otherwise falls back to a simple heuristic.
    - Action is a length-1 vector (order quantity).
    """

    model: DynamicSystemMVP
    H: int = 3
    x_max: float = 480.0
    dx: int = 1
    S_max: Optional[float] = None

    # economics
    p: float = 2.0
    c: float = 0.5
    h: float = 0.03
    b: float = 1.0
    K: float = 0.0

    forecaster: Optional[DemandForecaster] = None

    def __post_init__(self) -> None:
        self.H = int(self.H)
        if self.H <= 0:
            raise ValueError("H must be >= 1")

        self.x_max = float(self.x_max)
        self.dx = int(self.dx)
        if self.dx <= 0:
            raise ValueError("dx must be positive")

        if self.S_max is not None:
            self.S_max = float(self.S_max)

        self.p = float(self.p)
        self.c = float(self.c)
        self.h = float(self.h)
        self.b = float(self.b)
        self.K = float(self.K)

        self._q_max = int(np.floor(self.x_max / float(self.dx)))

    def __repr__(self) -> str:
        forecaster_name = type(self.forecaster).__name__ if self.forecaster is not None else None
        return (
            "DlaMpcMilpPolicy("
            f"H={self.H}, x_max={self.x_max}, dx={self.dx}, "
            f"S_max={self.S_max}, p={self.p}, c={self.c}, h={self.h}, "
            f"b={self.b}, K={self.K}, forecaster={forecaster_name!r})"
        )

    # -----------------
    # mean-path forecast
    # -----------------
    def _expert_mean_path(self, state: State, t: int, H: int) -> np.ndarray:
        exo = self.model.exogenous_model

        if isinstance(exo, ExogenousPoissonSeasonal):
            return np.array([float(exo.lambda_t(t + k)) for k in range(H)], dtype=float)

        if isinstance(exo, ExogenousPoissonRegime):
            r_t = int(np.round(float(state[exo.regime_index])))
            r_t = max(0, min(r_t, len(exo.lam_by_regime) - 1))
            dist = np.zeros(len(exo.lam_by_regime), dtype=float)
            dist[r_t] = 1.0

            mu = np.empty(H, dtype=float)
            for k in range(H):
                dist = dist @ exo.P
                mu[k] = float(dist @ exo.lam_by_regime)
            return mu

        return np.zeros(H, dtype=float)

    def _forecast_mean_path(self, state: State, t: int, H: int, info: Optional[dict]) -> np.ndarray:
        if self.forecaster is not None:
            mu = np.asarray(self.forecaster.forecast_mean_path(state, t, H, info=info), dtype=float).reshape(-1)
        else:
            mu = self._expert_mean_path(state, t, H)
        if mu.shape[0] != H:
            raise ValueError(f"Forecaster returned shape {mu.shape}; expected ({H},).")
        return np.maximum(mu, 0.0)

    # -----------------
    # fallback heuristic
    # -----------------
    def _fallback_order(self, inv0: float, mu0: float) -> float:
        x = max(0.0, float(mu0) - float(inv0))
        x = min(x, self.x_max)
        if self.S_max is not None:
            x = min(x, max(0.0, self.S_max - inv0))
        if self.dx > 1:
            x = self.dx * np.round(x / self.dx)
        return float(int(np.round(max(0.0, x))))

    # -----------------
    # Policy.act
    # -----------------
    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        inv0 = float(state[0])
        H = int(self.H)
        mu = self._forecast_mean_path(state, int(t), H, info)

        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
        except Exception:
            x0 = self._fallback_order(inv0, float(mu[0]) if len(mu) else 0.0)
            return np.array([x0], dtype=float)

        def idx_q(k: int) -> int:
            return 4 * k + 0

        def idx_y(k: int) -> int:
            return 4 * k + 1

        def idx_s(k: int) -> int:
            return 4 * k + 2

        def idx_l(k: int) -> int:
            return 4 * k + 3

        def idx_I(k: int) -> int:
            return 4 * H + k  # k=0..H

        n = 5 * H + 1

        # objective vector (minimize)
        cvec = np.zeros(n, dtype=float)
        for k in range(H):
            cvec[idx_q(k)] = self.c * float(self.dx)
            cvec[idx_y(k)] = self.K
            cvec[idx_s(k)] = -self.p
            cvec[idx_l(k)] = self.b
            cvec[idx_I(k + 1)] = self.h

        # integrality for q and y
        integrality = np.zeros(n, dtype=int)
        for k in range(H):
            integrality[idx_q(k)] = 1
            integrality[idx_y(k)] = 1

        # bounds
        lb = np.zeros(n, dtype=float)
        ub = np.full(n, np.inf, dtype=float)
        for k in range(H):
            lb[idx_q(k)] = 0.0
            ub[idx_q(k)] = float(self._q_max)

            lb[idx_y(k)] = 0.0
            ub[idx_y(k)] = 1.0

            lb[idx_s(k)] = 0.0
            ub[idx_s(k)] = float(mu[k])

            lb[idx_l(k)] = 0.0
            ub[idx_l(k)] = float(mu[k])

        for k in range(H + 1):
            lb[idx_I(k)] = 0.0

        # fix initial inventory
        lb[idx_I(0)] = inv0
        ub[idx_I(0)] = inv0

        bounds = Bounds(lb=lb, ub=ub)

        A_rows = []
        lb_con = []
        ub_con = []

        # (C1) s_k <= I_k + dx*q_k
        for k in range(H):
            row = np.zeros(n, dtype=float)
            row[idx_s(k)] = 1.0
            row[idx_I(k)] = -1.0
            row[idx_q(k)] = -float(self.dx)
            A_rows.append(row)
            lb_con.append(-np.inf)
            ub_con.append(0.0)

        # (C2) s_k + l_k = mu_k
        for k in range(H):
            row = np.zeros(n, dtype=float)
            row[idx_s(k)] = 1.0
            row[idx_l(k)] = 1.0
            rhs = float(mu[k])
            A_rows.append(row)
            lb_con.append(rhs)
            ub_con.append(rhs)

        # (C3) I_{k+1} = I_k + dx*q_k - s_k
        for k in range(H):
            row = np.zeros(n, dtype=float)
            row[idx_I(k + 1)] = 1.0
            row[idx_I(k)] = -1.0
            row[idx_q(k)] = -float(self.dx)
            row[idx_s(k)] = 1.0
            A_rows.append(row)
            lb_con.append(0.0)
            ub_con.append(0.0)

        # (C4) link q and y
        for k in range(H):
            row = np.zeros(n, dtype=float)
            row[idx_q(k)] = 1.0
            row[idx_y(k)] = -float(self._q_max)
            A_rows.append(row)
            lb_con.append(-np.inf)
            ub_con.append(0.0)

            row = np.zeros(n, dtype=float)
            row[idx_y(k)] = 1.0
            row[idx_q(k)] = -1.0
            A_rows.append(row)
            lb_con.append(-np.inf)
            ub_con.append(0.0)

        # (C5) optional capacity (pre-demand on-hand): I_k + dx*q_k <= S_max
        if self.S_max is not None:
            for k in range(H):
                row = np.zeros(n, dtype=float)
                row[idx_I(k)] = 1.0
                row[idx_q(k)] = float(self.dx)
                A_rows.append(row)
                lb_con.append(-np.inf)
                ub_con.append(float(self.S_max))

        A = np.vstack(A_rows) if A_rows else np.zeros((0, n), dtype=float)
        constraints = LinearConstraint(A, np.array(lb_con, dtype=float), np.array(ub_con, dtype=float))

        res = milp(c=cvec, integrality=integrality, bounds=bounds, constraints=constraints)
        if (not getattr(res, "success", False)) or (getattr(res, "x", None) is None):
            x0 = self._fallback_order(inv0, float(mu[0]) if len(mu) else 0.0)
            return np.array([x0], dtype=float)

        q0 = float(res.x[idx_q(0)])
        x0 = float(self.dx) * float(int(np.round(q0)))
        x0 = min(x0, self.x_max)
        if self.S_max is not None:
            x0 = min(x0, max(0.0, self.S_max - inv0))
        x0 = float(int(np.round(max(0.0, x0))))
        return np.array([x0], dtype=float)


# Backwards-compatible alias (from minimal_baseline naming)
DLA_MPC_MILP = DlaMpcMilpPolicy


__all__ = ["DlaMpcMilpPolicy", "DLA_MPC_MILP"]
