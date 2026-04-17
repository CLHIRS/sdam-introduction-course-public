from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from inventory.core.policy import Policy
from inventory.core.types import Action, State


@dataclass
class OrderMilpCfaPolicy(Policy):
    """CFA policy: compute the order quantity by solving a small MILP each time step.

    This is a forecast-then-optimize controller:
      1) Ask a forecaster for demand forecasts over a horizon H.
      2) Solve a deterministic MILP that chooses orders x_0..x_{H-1}.
      3) Execute ONLY the first decision x_0 (rolling horizon / MPC style).

    Model assumptions aligned with the baseline inventory MVP model:
      - Immediate replenishment (on-hand after ordering is I_k + x_k)
      - Lost sales (no backorders)
      - Inventory cannot go negative

    Notes:
      - Uses `scipy.optimize.milp` when available; otherwise falls back to an order-up-to rule.
      - Forecaster interface:
          - preferred: `forecaster.forecast(state, t, H, info=...)` returning an object with `.mean`.
          - fallback:  `forecaster.forecast_mean_path(state, t, H, info=...)`.
    """

    forecaster: object

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

    def __post_init__(self) -> None:
        self.H = int(self.H)
        if not (1 <= self.H <= 5):
            raise ValueError("Use H in [1,5] for the lecture MILP CFA demo.")

        self.x_max = float(self.x_max)
        self.dx = int(self.dx)
        if self.dx <= 0:
            raise ValueError("dx must be positive")

        self.S_max = None if self.S_max is None else float(self.S_max)

        self.p = float(self.p)
        self.c = float(self.c)
        self.h = float(self.h)
        self.b = float(self.b)
        self.K = float(self.K)

        self._q_max = int(np.floor(self.x_max / max(1, self.dx)))

    def __repr__(self) -> str:
        forecaster_name = type(self.forecaster).__name__ if self.forecaster is not None else None
        return (
            "OrderMilpCfaPolicy("
            f"H={self.H}, x_max={self.x_max}, dx={self.dx}, "
            f"S_max={self.S_max}, p={self.p}, c={self.c}, h={self.h}, "
            f"b={self.b}, K={self.K}, forecaster={forecaster_name!r})"
        )

    def _fallback_order(self, inv0: float, mu0: float) -> float:
        x = max(0.0, float(mu0) - float(inv0))
        x = min(x, self.x_max)
        if self.S_max is not None:
            x = min(x, max(0.0, self.S_max - inv0))
        if self.dx > 1:
            x = self.dx * np.round(x / self.dx)
        return float(int(np.round(max(0.0, x))))

    def _forecast_mean_path(self, state: State, t: int, H: int, info: Optional[dict]) -> np.ndarray:
        f = self.forecaster

        if hasattr(f, "forecast"):
            bundle = f.forecast(state, int(t), int(H), info=info)
            mu = np.asarray(getattr(bundle, "mean"), dtype=float).reshape(-1)
        else:
            # Allow forecasters that don't accept info
            try:
                mu = np.asarray(f.forecast_mean_path(state, int(t), int(H), info=info), dtype=float).reshape(-1)
            except TypeError:
                mu = np.asarray(f.forecast_mean_path(state, int(t), int(H)), dtype=float).reshape(-1)

        if mu.shape[0] != H:
            raise ValueError(f"Forecaster returned shape {mu.shape}; expected ({H},).")
        return np.maximum(mu, 0.0)

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

        # objective (minimize)
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

        lb[idx_I(0)] = inv0
        ub[idx_I(0)] = inv0

        bounds = Bounds(lb=lb, ub=ub)

        A_rows: list[np.ndarray] = []
        lb_con: list[float] = []
        ub_con: list[float] = []

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

        # (C5) optional capacity cap: I_k + dx*q_k <= S_max
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


# Backwards-compatible alias (from lecture_12_b naming)
Order_MILP_CFA = OrderMilpCfaPolicy


@dataclass
class OrderMilpCfaLeadTimePolicy(Policy):
    """CFA policy (MILP) with lead time via an explicit pipeline state.

    This is extracted from the Lecture 12_b notebook and adapted to the core `Policy` API.

    State convention:
      S_t = [I_t, p1_t, ..., pL_t]
        - I_t is on-hand inventory (>=0)
        - p_j_t is the amount scheduled to arrive in j steps (pipeline)

    Action convention:
      X_t = [x_t] where x_t enters the pipeline tail and arrives after L steps.

    Forecaster interface:
      - preferred: `forecaster.forecast(state, t, H, info=...)` returning an object with `.mean` (shape (H,))
      - fallback:  `forecaster.forecast_mean_path(state, t, H, info=...)` (shape (H,))

    MILP variables per step k=0..H-1:
      - q_k integer batches, x_k = dx*q_k
      - y_k 0/1 indicator (only relevant if K>0)
      - s_k continuous sales
      - l_k continuous lost sales

    Inside-MILP state variables:
      - I_k for k=0..H
      - P_{k,j} for k=0..H, j=1..L  (pipeline)

    Objective (sum over k):
      c*x_k + K*y_k + h*I_{k+1} + b*l_k - p*s_k
    """

    forecaster: object

    H: int = 4
    L: int = 2
    x_max: float = 480.0
    dx: int = 1
    S_max: Optional[float] = None  # optional pre-demand cap on on_hand = I_k + P_{k,1}

    # economics
    p: float = 2.0
    c: float = 0.5
    h: float = 0.03
    b: float = 1.0
    K: float = 0.0

    def __post_init__(self) -> None:
        self.H = int(self.H)
        self.L = int(self.L)
        if self.H <= 0:
            raise ValueError("H must be positive")
        if self.L < 0:
            raise ValueError("L must be >= 0")

        self.x_max = float(self.x_max)
        self.dx = int(self.dx)
        if self.dx <= 0:
            raise ValueError("dx must be positive")

        self.S_max = None if self.S_max is None else float(self.S_max)

        self.p = float(self.p)
        self.c = float(self.c)
        self.h = float(self.h)
        self.b = float(self.b)
        self.K = float(self.K)

        self._q_max = int(np.floor(self.x_max / max(1, self.dx)))

    def _forecast_mean_path(self, state: State, t: int, H: int, info: Optional[dict]) -> np.ndarray:
        f = self.forecaster

        if hasattr(f, "forecast"):
            bundle = f.forecast(state, int(t), int(H), info=info)
            mu = np.asarray(getattr(bundle, "mean"), dtype=float).reshape(-1)
        else:
            try:
                mu = np.asarray(f.forecast_mean_path(state, int(t), int(H), info=info), dtype=float).reshape(-1)
            except TypeError:
                mu = np.asarray(f.forecast_mean_path(state, int(t), int(H)), dtype=float).reshape(-1)

        if mu.shape[0] != H:
            raise ValueError(f"Forecaster returned shape {mu.shape}; expected ({H},).")
        return np.maximum(mu, 0.0)

    def _fallback_order(self, state: State, mu0: float) -> Action:
        """Lead-time-aware fallback: order-up-to on Inventory Position using target=mu0."""

        inv0 = float(state[0])
        if self.L > 0:
            if state.shape[0] < 1 + self.L:
                raise ValueError(f"State must include L pipeline entries; got {state.shape[0]-1} for L={self.L}.")
            pipe = np.asarray(state[1 : 1 + self.L], dtype=float)
        else:
            pipe = np.zeros(0, dtype=float)

        ip = inv0 + float(np.sum(pipe))
        x = max(0.0, float(mu0) - ip)
        x = min(x, self.x_max)
        if self.dx > 1:
            x = self.dx * np.round(x / self.dx)
        return np.array([float(int(np.round(max(0.0, x))))], dtype=float)

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        H, L = int(self.H), int(self.L)
        mu = self._forecast_mean_path(state, int(t), H, info)

        # SciPy MILP (fallback if unavailable)
        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
        except Exception:
            return self._fallback_order(state, mu0=float(mu[0]) if H > 0 else 0.0)

        # ----------------------------
        # Variable indexing
        # ----------------------------
        # Per step k=0..H-1: q_k, y_k, s_k, l_k  (4H vars)
        # Inventory I_k for k=0..H             (H+1 vars)
        # Pipeline P_{k,j} for k=0..H, j=1..L  ((H+1)*L vars)
        def idx_q(k: int) -> int:
            return 4 * k + 0

        def idx_y(k: int) -> int:
            return 4 * k + 1

        def idx_s(k: int) -> int:
            return 4 * k + 2

        def idx_l(k: int) -> int:
            return 4 * k + 3

        base_I = 4 * H

        def idx_I(k: int) -> int:
            return base_I + k  # k=0..H

        base_P = base_I + (H + 1)

        def idx_P(k: int, j: int) -> int:
            # j=1..L
            return base_P + k * L + (j - 1)

        n = 4 * H + (H + 1) + (H + 1) * L

        # ----------------------------
        # Objective (minimize)
        # ----------------------------
        cvec = np.zeros(n, dtype=float)
        for k in range(H):
            cvec[idx_q(k)] = self.c * float(self.dx)
            cvec[idx_y(k)] = self.K
            cvec[idx_s(k)] = -self.p
            cvec[idx_l(k)] = self.b
            cvec[idx_I(k + 1)] = self.h

        integrality = np.zeros(n, dtype=int)
        for k in range(H):
            integrality[idx_q(k)] = 1
            integrality[idx_y(k)] = 1

        # ----------------------------
        # Bounds
        # ----------------------------
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

        for k in range(H + 1):
            for j in range(1, L + 1):
                lb[idx_P(k, j)] = 0.0

        # Fix initial I_0 and pipeline P_0 from state
        inv0 = float(state[0])
        lb[idx_I(0)] = inv0
        ub[idx_I(0)] = inv0

        if L > 0:
            if state.shape[0] < 1 + L:
                raise ValueError(f"State must include L pipeline entries; got {state.shape[0]-1} for L={L}.")
            p0 = np.asarray(state[1 : 1 + L], dtype=float).reshape(-1)
            for j in range(1, L + 1):
                lb[idx_P(0, j)] = float(p0[j - 1])
                ub[idx_P(0, j)] = float(p0[j - 1])

        bounds = Bounds(lb=lb, ub=ub)

        # ----------------------------
        # Constraints (Linear)
        # ----------------------------
        A_rows: list[np.ndarray] = []
        lb_con: list[float] = []
        ub_con: list[float] = []

        # (C1) Demand split: s_k + l_k = mu_k
        for k in range(H):
            row = np.zeros(n, dtype=float)
            row[idx_s(k)] = 1.0
            row[idx_l(k)] = 1.0
            rhs = float(mu[k])
            A_rows.append(row)
            lb_con.append(rhs)
            ub_con.append(rhs)

        # (C2) Sales cannot exceed available inventory after arrival: s_k <= I_k + P_{k,1}
        for k in range(H):
            row = np.zeros(n, dtype=float)
            row[idx_s(k)] = 1.0
            row[idx_I(k)] = -1.0
            if L > 0:
                row[idx_P(k, 1)] = -1.0
            A_rows.append(row)
            lb_con.append(-np.inf)
            ub_con.append(0.0)

        # (C3) Inventory update with lost sales: I_{k+1} = I_k + P_{k,1} - s_k
        for k in range(H):
            row = np.zeros(n, dtype=float)
            row[idx_I(k + 1)] = 1.0
            row[idx_I(k)] = -1.0
            if L > 0:
                row[idx_P(k, 1)] = -1.0
            row[idx_s(k)] = 1.0
            A_rows.append(row)
            lb_con.append(0.0)
            ub_con.append(0.0)

        # (C4) Pipeline shift: P_{k+1,j} = P_{k,j+1} for j=1..L-1
        for k in range(H):
            for j in range(1, L):
                row = np.zeros(n, dtype=float)
                row[idx_P(k + 1, j)] = 1.0
                row[idx_P(k, j + 1)] = -1.0
                A_rows.append(row)
                lb_con.append(0.0)
                ub_con.append(0.0)

        # (C5) Pipeline tail equals new order: P_{k+1,L} = dx*q_k
        if L > 0:
            for k in range(H):
                row = np.zeros(n, dtype=float)
                row[idx_P(k + 1, L)] = 1.0
                row[idx_q(k)] = -float(self.dx)
                A_rows.append(row)
                lb_con.append(0.0)
                ub_con.append(0.0)

        # (C6) Link y_k to q_k (useful if K>0): q_k <= q_max*y_k and y_k <= q_k
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

        # (C7) Optional capacity on pre-demand on_hand = I_k + P_{k,1}
        if self.S_max is not None:
            for k in range(H):
                row = np.zeros(n, dtype=float)
                row[idx_I(k)] = 1.0
                if L > 0:
                    row[idx_P(k, 1)] = 1.0
                A_rows.append(row)
                lb_con.append(-np.inf)
                ub_con.append(float(self.S_max))

        A = np.vstack(A_rows) if A_rows else np.zeros((0, n), dtype=float)
        constraints = LinearConstraint(A, np.array(lb_con, dtype=float), np.array(ub_con, dtype=float))

        res = milp(c=cvec, integrality=integrality, bounds=bounds, constraints=constraints)
        if (not getattr(res, "success", False)) or (getattr(res, "x", None) is None):
            return self._fallback_order(state, mu0=float(mu[0]) if H > 0 else 0.0)

        q0 = float(res.x[idx_q(0)])
        x0 = float(self.dx) * float(int(np.round(q0)))
        x0 = min(x0, self.x_max)
        x0 = float(int(np.round(max(0.0, x0))))
        return np.array([x0], dtype=float)


# Backwards-compatible alias (from lecture_12_b naming)
Order_MILP_CFA_LeadTime = OrderMilpCfaLeadTimePolicy


__all__ = [
    "OrderMilpCfaPolicy",
    "Order_MILP_CFA",
    "OrderMilpCfaLeadTimePolicy",
    "Order_MILP_CFA_LeadTime",
]
