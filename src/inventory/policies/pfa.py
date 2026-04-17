from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Sequence, Tuple

import numpy as np
from inventory.core.dynamics import DynamicSystemMVP
from inventory.core.policy import Policy
from inventory.core.types import Action, State
from inventory.policies.baselines import OrderUpToPolicy


class Regressor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray):  # pragma: no cover
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        ...


@dataclass
class _OneStepOverridePolicy(Policy):
    """Take a fixed action at t=0, then follow base_policy."""

    base_policy: Policy
    x0: float

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        if int(t) == 0:
            return np.array([float(self.x0)], dtype=float)
        return self.base_policy.act(state, t, info=info)


def _round_to_grid(x: float, dx: int) -> float:
    if dx <= 1:
        return float(x)
    return float(dx * np.round(float(x) / float(dx)))


# ------------------------
# Blackbox (1-parameter) PFA
# ------------------------
@dataclass
class OrderUpToBlackboxPFA(Policy):
    """Black-box 1-parameter PFA (policy function approximation).

    Student view (the *idea*):
    - We choose a very small policy family: "order up to a target level S".
    - The target level is the single parameter: ``S = theta[0]``.
    - Then the action is the usual base-stock rule: ``x = max(0, S - inv)``.
    - Finally, apply practical constraints (cap + batching).

    Engineering view (the *implementation*):
    - ``theta`` is a NumPy array of shape ``(1,)``.
    - We clamp ("project") the parameter into ``[theta_min, theta_max]``.
    - The helper methods can tune theta by minimizing mean total cost under
      strict CRN (common random numbers) for fair comparisons.
    """

    system: DynamicSystemMVP
    theta: np.ndarray
    x_max: float = 480.0
    dx: int = 1
    theta_min: float = 0.0
    theta_max: float = 1000.0

    def __post_init__(self) -> None:
        self.theta = np.asarray(self.theta, dtype=float).reshape(-1)
        if self.theta.shape != (1,):
            raise ValueError("OrderUpToBlackboxPFA expects theta shape (1,).")
        self.x_max = float(self.x_max)
        self.dx = int(self.dx)
        self.theta_min = float(self.theta_min)
        self.theta_max = float(self.theta_max)
        self._project_theta()

    def __repr__(self) -> str:
        return (
            "OrderUpToBlackboxPFA("
            f"theta={self.theta.tolist()}, x_max={self.x_max}, dx={self.dx}, "
            f"theta_min={self.theta_min}, theta_max={self.theta_max})"
        )

    def _project_theta(self) -> None:
        """Clamp theta into the allowed parameter box [theta_min, theta_max]."""
        self.theta[0] = float(np.clip(self.theta[0], self.theta_min, self.theta_max))

    def theta_value(self) -> float:
        """Convenience accessor for the scalar parameter theta[0]."""
        return float(self.theta[0])

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        """Return the order quantity as a length-1 action vector.

        Conventions:
        - inventory is read from ``state[0]``
        - action is returned as ``np.array([x], dtype=float)``
        """

        inv = float(state[0])
        target = float(self.theta[0])
        # Base-stock (order-up-to) rule
        x = max(0.0, target - inv)
        # Practical constraints
        x = min(x, self.x_max)
        x = _round_to_grid(x, self.dx)
        return np.array([float(x)], dtype=float)

    # ---- evaluation helpers
    def _objective(self, theta: np.ndarray, S0: State, step_seed_paths: List[np.ndarray], *, T: int, info: Optional[dict]) -> float:
        """Mean total cost across episodes under strict CRN.

        Important: every candidate theta is evaluated on the *same* set of
        per-step seeds (step_seed_paths). This reduces Monte Carlo noise when
        comparing policies/parameters.
        """

        saved = self.theta.copy()
        try:
            self.theta = np.asarray(theta, dtype=float).reshape(1)
            self._project_theta()
            totals = []
            for step_seeds in step_seed_paths:
                _, costs, _, _ = self.system.simulate_crn(self, S0, step_seeds, info=info)
                totals.append(float(costs.sum()))
            return float(np.mean(totals))
        finally:
            self.theta = saved

    @staticmethod
    def alpha_schedule(k: int, *, schedule: str = "power", k0: int = 10, beta: float = 0.7, gamma: float = 0.98) -> float:
        k = int(k)
        if schedule == "power":
            return float(1.0 / ((k0 + k) ** beta))
        if schedule == "exp":
            return float(gamma**k)
        raise ValueError("schedule must be 'power' or 'exp'")

    def estimate_gradient_fd(
        self,
        *,
        S0: State,
        T: int,
        step_seed_paths: List[np.ndarray],
        eps: float = 1.0,
        info: Optional[dict] = None,
    ) -> np.ndarray:
        """Finite-difference gradient estimate of the strict-CRN objective."""
        theta0 = self.theta.copy()
        eps = float(eps)
        grad = np.zeros_like(theta0)

        for j in range(theta0.shape[0]):
            e = np.zeros_like(theta0)
            e[j] = 1.0
            f_plus = self._objective(theta0 + eps * e, S0, step_seed_paths, T=T, info=info)
            f_minus = self._objective(theta0 - eps * e, S0, step_seed_paths, T=T, info=info)
            grad[j] = (f_plus - f_minus) / (2.0 * eps)
        return grad

    def _optimize_fd(
        self,
        *,
        S0: State,
        T: int,
        n_episodes: int,
        seed0: int,
        n_iter: int,
        eps: float = 1.0,
        alpha0: float = 50.0,
        schedule: str = "power",
        info: Optional[dict] = None,
    ) -> "OrderUpToBlackboxPFA":
        """Optimize theta using finite-difference gradients (strict CRN)."""
        rng = np.random.default_rng(int(seed0))

        best_theta = self.theta.copy()
        best_obj = float("inf")

        for k in range(int(n_iter)):
            step_seed_paths = [self.system.sample_crn_step_seeds(int(s), int(T)) for s in rng.integers(1, 2**31 - 1, size=int(n_episodes))]

            grad = self.estimate_gradient_fd(S0=S0, T=T, step_seed_paths=step_seed_paths, eps=eps, info=info)
            alpha = float(alpha0) * self.alpha_schedule(k, schedule=schedule)

            self.theta = self.theta - alpha * grad
            self._project_theta()

            obj = self._objective(self.theta, S0, step_seed_paths, T=T, info=info)
            if obj < best_obj:
                best_obj = obj
                best_theta = self.theta.copy()

        self.theta = best_theta
        self._project_theta()
        return self

    def _optimize_spsa(
        self,
        *,
        S0: State,
        T: int,
        n_episodes: int,
        seed0: int,
        n_iter: int,
        c0: float = 5.0,
        alpha0: float = 50.0,
        schedule: str = "power",
        info: Optional[dict] = None,
    ) -> "OrderUpToBlackboxPFA":
        """Optimize theta using SPSA gradient estimates (strict CRN)."""
        rng = np.random.default_rng(int(seed0))

        best_theta = self.theta.copy()
        best_obj = float("inf")

        for k in range(int(n_iter)):
            step_seed_paths = [self.system.sample_crn_step_seeds(int(s), int(T)) for s in rng.integers(1, 2**31 - 1, size=int(n_episodes))]

            delta = rng.choice([-1.0, 1.0], size=self.theta.shape)
            ck = float(c0) / ((k + 1) ** 0.101)
            theta_plus = self.theta + ck * delta
            theta_minus = self.theta - ck * delta

            f_plus = self._objective(theta_plus, S0, step_seed_paths, T=T, info=info)
            f_minus = self._objective(theta_minus, S0, step_seed_paths, T=T, info=info)

            ghat = (f_plus - f_minus) / (2.0 * ck) * delta
            alpha = float(alpha0) * self.alpha_schedule(k, schedule=schedule)

            self.theta = self.theta - alpha * ghat
            self._project_theta()

            obj = self._objective(self.theta, S0, step_seed_paths, T=T, info=info)
            if obj < best_obj:
                best_obj = obj
                best_theta = self.theta.copy()

        self.theta = best_theta
        self._project_theta()
        return self

    def optimize(
        self,
        *,
        S0: State,
        T: int = 50,
        n_episodes: int = 200,
        seed0: int = 1234,
        n_iter: int = 30,
        method: str = "spsa",
        info: Optional[dict] = None,
        **kwargs,
    ) -> "OrderUpToBlackboxPFA":
        """Optimize theta using strict-CRN Monte Carlo objective.

        `method`:
          - 'fd'   : finite-difference gradient
          - 'spsa' : SPSA gradient estimate
        """

        method = str(method).lower().strip()
        if method == "fd":
            return self._optimize_fd(S0=S0, T=T, n_episodes=n_episodes, seed0=seed0, n_iter=n_iter, info=info, **kwargs)
        if method == "spsa":
            return self._optimize_spsa(S0=S0, T=T, n_episodes=n_episodes, seed0=seed0, n_iter=n_iter, info=info, **kwargs)
        raise ValueError("method must be 'fd' or 'spsa'")


# ------------------------
# Regime-table (K-parameter) PFA
# ------------------------
@dataclass
class OrderUpToRegimeTablePFA(Policy):
    """Regime-indexed order-up-to targets (a small but useful PFA).

        Student view:
        - Instead of one global target, keep one target per regime.
        - ``theta[r]`` is the target level used when the system is in regime ``r``.

        Implementation details:
        - ``theta`` is a 1-D vector of length K (K = number of regimes).
        - The regime is read from ``state[regime_index]``, rounded to an integer, then
            clamped into ``{0,1,...,K-1}``.
        - Action uses the same base-stock rule + ``x_max`` + ``dx`` rounding.
        """

    system: DynamicSystemMVP
    theta: np.ndarray
    regime_index: int = 1
    x_max: float = 480.0
    dx: int = 1
    theta_min: float = 0.0
    theta_max: float = 1000.0

    def __post_init__(self) -> None:
        self.theta = np.asarray(self.theta, dtype=float).reshape(-1)
        if self.theta.ndim != 1 or self.theta.shape[0] < 1:
            raise ValueError("theta must be a 1-D vector of length K>=1")
        self.regime_index = int(self.regime_index)
        self.x_max = float(self.x_max)
        self.dx = int(self.dx)
        self.theta_min = float(self.theta_min)
        self.theta_max = float(self.theta_max)
        self._project_theta()

    def _project_theta(self) -> None:
        """Clamp every regime-target into [theta_min, theta_max]."""
        self.theta[:] = np.clip(self.theta, self.theta_min, self.theta_max)

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        """Order up to a regime-dependent target.

        Conventions:
        - inventory in ``state[0]``
        - regime in ``state[regime_index]``
        """

        inv = float(state[0])
        # Regime is stored as a float in the state; round to nearest integer index.
        r = int(np.round(float(state[self.regime_index])))
        # Clamp to valid regime set
        r = max(0, min(r, int(self.theta.shape[0]) - 1))
        target = float(self.theta[r])
        x = max(0.0, target - inv)
        x = min(x, self.x_max)
        x = _round_to_grid(x, self.dx)
        return np.array([float(x)], dtype=float)


# ------------------------
# State-dependent PFA via rollout improvement + regressor
# ------------------------
@dataclass
class OrderUpToStateDependentPFA(Policy):
    """State-dependent PFA: learn a target level from features, then order up to it.

    Student view:
    - We still use an order-up-to rule, but the *target level* is a learned function
      of the state (and time), rather than a constant or a regime table.
    - We learn a mapping: ``target_level ≈ f(feature_fn(state, t))``.

    How it is trained (rollout-based policy improvement):
    1) Simulate trajectories with a behavior policy.
    2) At visited states, try several candidate targets.
    3) For each candidate, run a short rollout of length H using the same CRN
       suffix seeds (fair comparison).
    4) Pick the target that minimizes rollout cost; use it as the label y.
    5) Fit a regressor to map features -> best target.

    Implementation notes:
    - ``regressor`` can be any sklearn-like object with ``fit`` and ``predict``.
    - ``feature_fn`` defaults to a small feature vector [inv, reg, t].
    - Feasibility is handled by `_project_target()` and `_action_from_target()`.
    """

    system: DynamicSystemMVP
    x_max: float = 480.0
    dx: int = 1
    s_max: Optional[float] = None
    target_min: float = 0.0
    target_max: Optional[float] = 1000.0
    rounding: str = "round"  # 'round' | 'floor' | 'ceil'
    H: int = 5

    regressor: Optional[Regressor] = None
    feature_fn: Optional[Callable[[State, int], np.ndarray]] = None

    is_fitted: bool = False

    def __post_init__(self) -> None:
        self.x_max = float(self.x_max)
        self.dx = int(self.dx)
        self.s_max = None if self.s_max is None else float(self.s_max)
        self.target_min = float(self.target_min)
        if self.target_max is None and self.s_max is not None:
            self.target_max = float(self.s_max)
        self.target_max = None if self.target_max is None else float(self.target_max)
        self.rounding = str(self.rounding)
        self.H = int(self.H)
        if self.H <= 0:
            raise ValueError("H must be >= 1")

        if self.regressor is None:
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore

                self.regressor = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.1)
            except Exception as e:
                raise ImportError(
                    "OrderUpToStateDependentPFA needs scikit-learn, or pass a custom regressor implementing fit/predict."
                ) from e

        if self.feature_fn is None:

            def _default_features(S: State, t: int) -> np.ndarray:
                S = np.asarray(S, dtype=float).reshape(-1)
                inv = float(S[0]) if S.size >= 1 else 0.0
                reg = float(S[1]) if S.size >= 2 else 0.0
                return np.array([inv, reg, float(t)], dtype=float)

            self.feature_fn = _default_features

    def __repr__(self) -> str:
        regressor_name = type(self.regressor).__name__ if self.regressor is not None else None
        feature_fn_name = None
        if self.feature_fn is not None:
            feature_fn_name = getattr(self.feature_fn, "__name__", type(self.feature_fn).__name__)
        return (
            "OrderUpToStateDependentPFA("
            f"x_max={self.x_max}, dx={self.dx}, s_max={self.s_max}, target_min={self.target_min}, "
            f"target_max={self.target_max}, rounding={self.rounding!r}, H={self.H}, "
            f"regressor={regressor_name!r}, feature_fn={feature_fn_name!r}, is_fitted={self.is_fitted})"
        )

    def _round(self, x: float) -> float:
        if self.rounding == "round":
            return float(np.round(x))
        if self.rounding == "floor":
            return float(np.floor(x))
        if self.rounding == "ceil":
            return float(np.ceil(x))
        raise ValueError("rounding must be 'round', 'floor', or 'ceil'.")

    def _project_target(self, target: float) -> float:
        """Clamp and round the predicted target level into a safe, integer target."""
        tgt = float(target)
        tgt = max(tgt, self.target_min)
        if self.target_max is not None:
            tgt = min(tgt, self.target_max)
        tgt = self._round(tgt)
        return float(int(tgt))

    def _action_from_target(self, inv: float, target: float) -> float:
        """Convert a (possibly learned) target level into a feasible order quantity."""
        x = max(0.0, float(target) - float(inv))
        x = min(x, self.x_max)
        if self.s_max is not None:
            x = min(x, max(0.0, float(self.s_max) - float(inv)))

        if self.dx > 1:
            x = float(self.dx * np.round(x / float(self.dx)))
            x = max(0.0, x)
            x = min(x, self.x_max)
            if self.s_max is not None:
                x = min(x, max(0.0, float(self.s_max) - float(inv)))

        x = float(int(np.round(x)))
        return float(x)

    def predict_target(self, state: State, t: int) -> float:
        """Predict and project the target level for a specific (state, t)."""
        assert self.feature_fn is not None
        feats = np.asarray(self.feature_fn(state, int(t)), dtype=float).reshape(1, -1)
        pred = float(np.asarray(self.regressor.predict(feats)).reshape(-1)[0])  # type: ignore[union-attr]
        return self._project_target(pred)

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        """Choose an order quantity using the learned target-level model.

        If the model is not fitted yet, we use a simple fallback target.
        """

        inv = float(state[0])
        if not self.is_fitted:
            # reasonable fallback until fit() has been called
            if self.target_max is not None:
                target = 0.5 * (self.target_min + self.target_max)
            else:
                target = self.target_min
        else:
            target = self.predict_target(state, t)
        x = self._action_from_target(inv, target)
        return np.array([float(x)], dtype=float)

    def _evaluate_candidate_target_at_state(
        self,
        *,
        state: State,
        t: int,
        target: float,
        base_policy: Policy,
        step_seeds: np.ndarray,
        info: Optional[dict] = None,
    ) -> float:
        """Evaluate a candidate target using a short rollout from (state, t).

        Uses a policy that overrides the action at t (equivalently t=0 relative
        to the rollout start), then follows `base_policy`.
        """

        inv = float(state[0])
        x0 = self._action_from_target(inv, float(target))
        pol = _OneStepOverridePolicy(base_policy=base_policy, x0=x0)

        # rollout from current state for H steps using suffix of seeds
        step_seeds = np.asarray(step_seeds, dtype=np.int64)
        suffix = step_seeds[int(t) : int(t) + int(self.H)]
        if suffix.shape[0] <= 0:
            return 0.0

        _, costs, _, _ = self.system.simulate_crn(pol, state, suffix, info=info)
        return float(costs.sum())

    def collect_training_data(
        self,
        *,
        S0: State,
        T: int,
        n_episodes: int,
        seed0: int,
        behavior_policy: Policy,
        base_policy_for_rollout: Policy,
        candidate_targets: Sequence[float],
        info: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect (features, best_target) labels from visited states.

        For each visited state, we evaluate all `candidate_targets` using the same
        CRN suffix seeds, pick the target with minimum rollout cost, and store it
        as the supervised learning label.
        """

        candidate_targets = [self._project_target(float(z)) for z in candidate_targets]
        rng = np.random.default_rng(int(seed0))

        X_rows: List[np.ndarray] = []
        y_rows: List[float] = []

        for ep_seed in rng.integers(1, 2**31 - 1, size=int(n_episodes)):
            step_seeds = self.system.sample_crn_step_seeds(int(ep_seed), int(T))
            traj, _, _, _ = self.system.simulate_crn(behavior_policy, S0, step_seeds, info=info)

            for t in range(int(T)):
                state = traj[t]

                # Evaluate candidates using same suffix CRN
                vals = [
                    self._evaluate_candidate_target_at_state(
                        state=state,
                        t=t,
                        target=z,
                        base_policy=base_policy_for_rollout,
                        step_seeds=step_seeds,
                        info=info,
                    )
                    for z in candidate_targets
                ]
                best = float(candidate_targets[int(np.argmin(vals))])

                feats = np.asarray(self.feature_fn(state, int(t)), dtype=float)  # type: ignore[misc]
                X_rows.append(feats)
                y_rows.append(best)

        X = np.vstack([r.reshape(1, -1) for r in X_rows]) if X_rows else np.zeros((0, 0), dtype=float)
        y = np.asarray(y_rows, dtype=float)
        return X, y

    def fit_via_rollout_improvement(
        self,
        *,
        S0: State,
        T: int,
        n_episodes: int = 60,
        seed0: int = 1234,
        n_iter: int = 3,
        candidate_targets: Optional[Sequence[float]] = None,
        behavior_policy: Optional[Policy] = None,
        info: Optional[dict] = None,
        eval_every: int = 1,
        eval_n_episodes: int = 200,
        eval_seed0: int = 9999,
    ) -> "OrderUpToStateDependentPFA":
        """Iterative rollout-improvement loop.

        Default behavior policy starts as a mid-level order-up-to policy, then switches to the learned policy.
        """

        if candidate_targets is None:
            # coarse grid of targets for label generation
            candidate_targets = np.linspace(self.target_min, self.target_max, num=21)

        if behavior_policy is None:
            mid = 0.5 * (self.target_min + self.target_max)
            behavior_policy = OrderUpToPolicy(target_level=mid, x_max=self.x_max, dx=self.dx)

        # base policy used after t=0 in the short lookahead rollouts
        base_for_rollout: Policy = self if self.is_fitted else behavior_policy

        best_totals_mean = float("inf")
        best_model = None

        for it in range(int(n_iter)):
            X, y = self.collect_training_data(
                S0=S0,
                T=T,
                n_episodes=n_episodes,
                seed0=seed0 + 1000 * it,
                behavior_policy=behavior_policy,
                base_policy_for_rollout=base_for_rollout,
                candidate_targets=candidate_targets,
                info=info,
            )

            if X.shape[0] > 0:
                self.regressor.fit(X, y)  # type: ignore[union-attr]
                self.is_fitted = True

            if eval_every > 0 and (it % int(eval_every) == 0):
                totals = self.system.evaluate_policy_crn_mc(self, S0=S0, T=T, n_episodes=eval_n_episodes, seed0=eval_seed0 + it, info=info)
                m = float(np.mean(totals))
                if m < best_totals_mean:
                    best_totals_mean = m
                    best_model = self.regressor

            behavior_policy = self
            base_for_rollout = self

        if best_model is not None:
            self.regressor = best_model
            self.is_fitted = True

        return self


__all__ = [
    "OrderUpToBlackboxPFA",
    "OrderUpToRegimeTablePFA",
    "OrderUpToStateDependentPFA",
    # Notebook-compatible aliases (lecture_11_b naming)
    "OrderUpTo_blackbox_PFA",
    "OrderUpTo_regime_table_PFA",
    "OrderUpTo_state_dependent_PFA_2",
]


# Notebook-compatible aliases (lecture_11_b naming)
OrderUpTo_blackbox_PFA = OrderUpToBlackboxPFA
OrderUpTo_regime_table_PFA = OrderUpToRegimeTablePFA
OrderUpTo_state_dependent_PFA_2 = OrderUpToStateDependentPFA
