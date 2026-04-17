from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from inventory.core.dynamics import DynamicSystemMVP
from inventory.core.policy import Policy
from inventory.core.types import Action, State
from inventory.problems.demand_models import ExogenousPoissonRegime, ExogenousPoissonSeasonal


@dataclass
class PostDecisionGreedyVfaPolicy(Policy):
    """Greedy policy using a post-decision value function approximation.

    Extracted from `lecture_13_b_V0.1.ipynb`.

    Action choice (minimization):
      x_t = argmin_x  E[C(S_t, x, W_{t+1})] + gamma * Vbar_hat(S^x_t, t)

    The post-decision state is inventory-after-order, before-demand.

    Learning: TD(0) on the post-decision value.

    Notes:
      - Deterministic MC approximation of the expectation to avoid adding policy RNG.
      - Supports both 1D state [inventory] and 2D [inventory, regime].
    """

    model: DynamicSystemMVP
    gamma: float = 0.99
    x_max: float = 600.0
    dx: int = 1
    alpha: float = 0.05
    expectation_samples: int = 256
    expectation_seed: int = 12345
    feature: str = "poly2"
    inv_scale: float = 1.0

    def __post_init__(self) -> None:
        self.gamma = float(self.gamma)
        self.x_max = float(self.x_max)
        self.dx = int(self.dx)
        self.alpha = float(self.alpha)
        self.expectation_samples = int(self.expectation_samples)
        self.expectation_seed = int(self.expectation_seed)
        self.feature = str(self.feature)
        self.inv_scale = float(self.inv_scale)
        if self.inv_scale <= 0.0:
            raise ValueError("inv_scale must be positive")
        self.w: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return (
            "PostDecisionGreedyVfaPolicy("
            f"gamma={self.gamma}, x_max={self.x_max}, dx={self.dx}, "
            f"alpha={self.alpha}, expectation_samples={self.expectation_samples}, "
            f"expectation_seed={self.expectation_seed}, feature={self.feature!r}, "
            f"inv_scale={self.inv_scale})"
        )

    def _n_regimes(self) -> int:
        exo = self.model.exogenous_model
        if isinstance(exo, ExogenousPoissonRegime):
            return int(len(exo.lam_by_regime))
        return 0

    def _phi_post(self, S_post: np.ndarray, t: int) -> np.ndarray:
        inv = float(S_post[0]) / self.inv_scale
        tau = float(t) / 50.0

        if self.feature == "linear":
            base = [1.0, inv, tau]
        else:
            base = [1.0, inv, inv * inv, tau, tau * tau, inv * tau]

        if S_post.shape[0] >= 2:
            K = self._n_regimes()
            if K <= 0:
                r = float(S_post[1])
                base.extend([r, r * r])
            else:
                r = int(np.round(float(S_post[1])))
                r = max(0, min(r, K - 1))
                one_hot = [0.0] * K
                one_hot[r] = 1.0
                base.extend(one_hot)

        return np.asarray(base, dtype=float)

    def _ensure_w(self, phi: np.ndarray) -> None:
        if self.w is None:
            self.w = np.zeros(phi.shape[0], dtype=float)
        elif self.w.shape[0] != phi.shape[0]:
            raise ValueError("Feature dimension changed; cannot reuse existing weights.")

    def vbar_hat(self, S_post: np.ndarray, t: int) -> float:
        phi = self._phi_post(S_post, t)
        self._ensure_w(phi)
        return float(np.dot(self.w, phi))

    def _candidate_orders(self) -> np.ndarray:
        if self.dx <= 0:
            raise ValueError("dx must be positive")
        xs = np.arange(0, int(np.floor(self.x_max)) + 1, int(self.dx), dtype=int)
        return xs.astype(float)

    def _make_post_state(self, S: np.ndarray, x: float) -> np.ndarray:
        inv_post = max(0.0, float(S[0]) + float(x))
        if S.shape[0] == 1:
            return np.array([inv_post], dtype=float)
        return np.array([inv_post, float(S[1])], dtype=float)

    def _seed_for_expectation(self, t: int, inv: float) -> int:
        inv_key = int(np.round(inv))
        return int((self.expectation_seed + 1000003 * int(t) + 1009 * inv_key) % (2**31 - 1))

    def _expected_one_step_cost(self, S: np.ndarray, x: float, t: int) -> float:
        rng = np.random.default_rng(self._seed_for_expectation(t, float(S[0])))

        X = np.array([float(int(np.round(x)))], dtype=float)
        exo = self.model.exogenous_model
        n = int(self.expectation_samples)

        if isinstance(exo, ExogenousPoissonSeasonal):
            lam = exo.lambda_t(t)
            demands = rng.poisson(lam, size=n).astype(float)
            costs = np.empty(n, dtype=float)
            for i in range(n):
                W = np.array([demands[i]], dtype=float)
                costs[i] = float(self.model.cost_func(S, X, W, t))
            return float(costs.mean())

        if isinstance(exo, ExogenousPoissonRegime):
            r_t = int(np.round(float(S[exo.regime_index])))
            r_t = max(0, min(r_t, len(exo.lam_by_regime) - 1))
            probs = exo.P[r_t]
            r_next = rng.choice(len(probs), size=n, p=probs)
            lams = exo.lam_by_regime[r_next]
            demands = rng.poisson(lams).astype(float)
            costs = np.empty(n, dtype=float)
            for i in range(n):
                W = np.array([demands[i], float(r_next[i])], dtype=float)
                costs[i] = float(self.model.cost_func(S, X, W, t))
            return float(costs.mean())

        costs = np.empty(n, dtype=float)
        for i in range(n):
            W = exo.sample(S, X, t, rng)
            costs[i] = float(self.model.cost_func(S, X, W, t))
        return float(costs.mean())

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        xs = self._candidate_orders()
        best_x = float(xs[0])
        best_obj = float("inf")

        for x in xs:
            S_post = self._make_post_state(state, float(x))
            obj = self._expected_one_step_cost(state, float(x), t) + self.gamma * self.vbar_hat(S_post, t)
            if obj < best_obj:
                best_obj = obj
                best_x = float(x)

        return np.array([float(int(np.round(best_x)))], dtype=float)

    def train_td_value(
        self,
        *,
        S0: np.ndarray,
        T: int,
        n_episodes: int = 200,
        seed0: int = 2026,
        epsilon: float = 0.1,
    ) -> None:
        T = int(T)
        n_episodes = int(n_episodes)
        epsilon = float(epsilon)

        rng = np.random.default_rng(int(seed0))
        ep_seeds = [int(s) for s in rng.integers(1, 2**31 - 1, size=n_episodes)]

        for ep_seed in ep_seeds:
            step_seeds = self.model.sample_crn_step_seeds(ep_seed, T)
            S_t = np.asarray(S0, dtype=float).copy()

            for t in range(T):
                if rng.random() < epsilon:
                    x = float(rng.choice(self._candidate_orders()))
                else:
                    x = float(self.act(S_t, t)[0])

                X_t = np.array([float(int(np.round(x)))], dtype=float)
                S_post = self._make_post_state(S_t, x)
                phi = self._phi_post(S_post, t)
                self._ensure_w(phi)
                v_curr = float(np.dot(self.w, phi))

                rng_t = np.random.default_rng(int(step_seeds[t]))
                W_tp1 = self.model.exogenous_model.sample(S_t, X_t, t, rng_t)
                C_t = float(self.model.cost_func(S_t, X_t, W_tp1, t))
                S_tp1 = self.model.transition_func(S_t, X_t, W_tp1, t)

                if t == T - 1:
                    v_next = 0.0
                else:
                    x_next = float(self.act(S_tp1, t + 1)[0])
                    S_post_next = self._make_post_state(S_tp1, x_next)
                    v_next = self.vbar_hat(S_post_next, t + 1)

                delta = C_t + self.gamma * v_next - v_curr
                self.w += self.alpha * delta * phi

                S_t = np.asarray(S_tp1, dtype=float)


class _RidgeFallback:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.w: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        # Add intercept column
        X1 = np.hstack([np.ones((X.shape[0], 1)), X])
        A = X1.T @ X1 + self.alpha * np.eye(X1.shape[1])
        b = X1.T @ y
        self.w = np.linalg.solve(A, b)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fit")
        X = np.asarray(X, dtype=float)
        X1 = np.hstack([np.ones((X.shape[0], 1)), X])
        return (X1 @ self.w).astype(float)


@dataclass
class FqiGreedyVfaPolicy(Policy):
    """Fitted Q Iteration (batch/offline) for a discrete action set.

    Extracted from `lecture_13_b_V0.1.ipynb`.

    Train a Q surrogate on Bellman backup targets:
      y = C(S,a,W) + gamma * min_{a'} Q_hat(S', a')

    Then act greedily online:
      a = argmin_{a} Q_hat(S, a)

    Notes:
      - Supports both 1D inventory state and 2D [inventory, regime] state.
      - Uses sklearn models when available; ridge has a numpy fallback.
    """

    model: DynamicSystemMVP
    gamma: float = 0.99
    x_max: float = 600.0
    dx: int = 1

    mode: str = "ridge"
    ridge_alpha: float = 1.0
    elastic_alpha: float = 1.0
    elastic_l1_ratio: float = 0.5
    tree_max_depth: int = 4
    tree_learning_rate: float = 0.08
    tree_max_iter: int = 400
    tree_min_samples_leaf: int = 20
    mlp_hidden: Tuple[int, ...] = (64, 64)
    mlp_max_iter: int = 500
    random_state: int = 0
    feature: str = "poly2"
    inv_scale: float = 1.0
    a_scale: float = 1.0

    def __post_init__(self) -> None:
        self.gamma = float(self.gamma)
        self.x_max = float(self.x_max)
        self.dx = int(self.dx)

        self.mode = str(self.mode)
        self.ridge_alpha = float(self.ridge_alpha)
        self.elastic_alpha = float(self.elastic_alpha)
        self.elastic_l1_ratio = float(self.elastic_l1_ratio)
        self.tree_max_depth = int(self.tree_max_depth)
        self.tree_learning_rate = float(self.tree_learning_rate)
        self.tree_max_iter = int(self.tree_max_iter)
        self.tree_min_samples_leaf = int(self.tree_min_samples_leaf)
        self.mlp_hidden = tuple(int(h) for h in self.mlp_hidden)
        self.mlp_max_iter = int(self.mlp_max_iter)
        self.random_state = int(self.random_state)
        self.feature = str(self.feature)
        self.inv_scale = float(self.inv_scale)
        self.a_scale = float(self.a_scale)
        if self.inv_scale <= 0.0 or self.a_scale <= 0.0:
            raise ValueError("inv_scale and a_scale must be positive")

        self.q_model = None

    def __repr__(self) -> str:
        return (
            "FqiGreedyVfaPolicy("
            f"gamma={self.gamma}, x_max={self.x_max}, dx={self.dx}, "
            f"mode={self.mode!r}, ridge_alpha={self.ridge_alpha}, "
            f"feature={self.feature!r}, inv_scale={self.inv_scale}, "
            f"a_scale={self.a_scale})"
        )

    def _n_regimes(self) -> int:
        exo = self.model.exogenous_model
        if isinstance(exo, ExogenousPoissonRegime):
            return int(len(exo.lam_by_regime))
        return 0

    def _candidate_orders(self) -> np.ndarray:
        if self.dx <= 0:
            raise ValueError("dx must be positive")
        xs = np.arange(0, int(np.floor(self.x_max)) + 1, int(self.dx), dtype=int)
        return xs.astype(float)

    def _phi_sa(self, S: np.ndarray, t: int, x: float) -> np.ndarray:
        inv = float(S[0]) / self.inv_scale
        a = float(x) / self.a_scale
        tau = float(t) / 50.0

        if self.feature == "linear":
            base = [1.0, inv, a, tau]
        else:
            base = [1.0, inv, inv * inv, a, a * a, inv * a, tau, tau * tau, inv * tau, a * tau]

        if S.shape[0] >= 2:
            K = self._n_regimes()
            if K <= 0:
                r = float(S[1])
                base.extend([r, r * r, r * a, r * inv])
            else:
                r = int(np.round(float(S[1])))
                r = max(0, min(r, K - 1))
                one_hot = [0.0] * K
                one_hot[r] = 1.0
                base.extend(one_hot)

        return np.asarray(base, dtype=float)

    def _fit_model(self):
        if self.mode == "ridge":
            try:
                from sklearn.linear_model import Ridge

                return Ridge(alpha=self.ridge_alpha, fit_intercept=True)
            except Exception:
                return _RidgeFallback(alpha=self.ridge_alpha)

        if self.mode == "elastic":
            from sklearn.linear_model import ElasticNet
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            return Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "elastic",
                        ElasticNet(
                            alpha=self.elastic_alpha,
                            l1_ratio=self.elastic_l1_ratio,
                            fit_intercept=True,
                            random_state=self.random_state,
                            max_iter=10_000,
                        ),
                    ),
                ]
            )

        if self.mode == "tree":
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(
                learning_rate=self.tree_learning_rate,
                max_depth=self.tree_max_depth,
                max_iter=self.tree_max_iter,
                min_samples_leaf=self.tree_min_samples_leaf,
                random_state=self.random_state,
            )

        if self.mode == "gbrt":
            from sklearn.ensemble import GradientBoostingRegressor

            return GradientBoostingRegressor(
                n_estimators=self.tree_max_iter,
                learning_rate=self.tree_learning_rate,
                max_depth=self.tree_max_depth,
                min_samples_leaf=self.tree_min_samples_leaf,
                random_state=self.random_state,
            )

        if self.mode == "mlp":
            from sklearn.neural_network import MLPRegressor
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            return Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPRegressor(
                            hidden_layer_sizes=self.mlp_hidden,
                            activation="relu",
                            alpha=1e-4,
                            learning_rate_init=1e-3,
                            max_iter=self.mlp_max_iter,
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )

        raise ValueError("mode must be one of {'ridge','elastic','tree','gbrt','mlp'}")

    def _predict_q(self, S: np.ndarray, t: int, xs: np.ndarray) -> np.ndarray:
        if self.q_model is None:
            return np.zeros(xs.shape[0], dtype=float)
        Phi = np.vstack([self._phi_sa(S, t, float(x)) for x in xs])
        return np.asarray(self.q_model.predict(Phi), dtype=float)

    def _min_q_next(self, S_next: np.ndarray, t_next: int) -> float:
        xs = self._candidate_orders()
        q = self._predict_q(S_next, t_next, xs)
        return float(np.min(q))

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        xs = self._candidate_orders()
        q = self._predict_q(state, t, xs)
        best_x = float(xs[int(np.argmin(q))])
        return np.array([float(int(np.round(best_x)))], dtype=float)

    def train_fqi(
        self,
        *,
        S0: np.ndarray,
        T: int,
        n_episodes: int = 300,
        seed0: int = 2026,
        n_iterations: int = 12,
        behavior: str = "random",
        eps_behavior: float = 0.2,
    ) -> None:
        T = int(T)
        n_episodes = int(n_episodes)
        n_iterations = int(n_iterations)
        eps_behavior = float(eps_behavior)

        rng = np.random.default_rng(int(seed0))
        ep_seeds = [int(s) for s in rng.integers(1, 2**31 - 1, size=n_episodes)]

        S_list: List[np.ndarray] = []
        t_list: List[int] = []
        a_list: List[float] = []
        c_list: List[float] = []
        Snext_list: List[np.ndarray] = []
        tnext_list: List[int] = []

        xs = self._candidate_orders()

        for ep_seed in ep_seeds:
            step_seeds = self.model.sample_crn_step_seeds(ep_seed, T)
            S_t = np.asarray(S0, dtype=float).copy()

            for t in range(T):
                if behavior == "random":
                    x = float(rng.choice(xs))
                elif behavior == "egreedy":
                    if rng.random() < eps_behavior:
                        x = float(rng.choice(xs))
                    else:
                        x = float(self.act(S_t, t)[0])
                else:
                    raise ValueError("behavior must be one of {'random','egreedy'}")

                X_t = np.array([float(int(np.round(x)))], dtype=float)
                rng_t = np.random.default_rng(int(step_seeds[t]))
                W_tp1 = self.model.exogenous_model.sample(S_t, X_t, t, rng_t)
                C_t = float(self.model.cost_func(S_t, X_t, W_tp1, t))
                S_tp1 = self.model.transition_func(S_t, X_t, W_tp1, t)

                S_list.append(np.asarray(S_t, dtype=float).copy())
                t_list.append(int(t))
                a_list.append(float(X_t[0]))
                c_list.append(float(C_t))
                Snext_list.append(np.asarray(S_tp1, dtype=float).copy())
                tnext_list.append(int(t + 1))

                S_t = np.asarray(S_tp1, dtype=float)

        Phi = np.vstack([self._phi_sa(S_list[i], t_list[i], a_list[i]) for i in range(len(S_list))])
        c_arr = np.asarray(c_list, dtype=float)

        for _k in range(n_iterations):
            y = np.empty_like(c_arr)
            for i in range(len(S_list)):
                if t_list[i] >= T - 1:
                    y[i] = c_arr[i]
                else:
                    y[i] = c_arr[i] + self.gamma * self._min_q_next(Snext_list[i], tnext_list[i])

            model_k = self._fit_model()
            model_k.fit(Phi, y)
            self.q_model = model_k


# Backwards-compatible names from the notebook
PostDecision_Greedy_VFA = PostDecisionGreedyVfaPolicy
FQI_Greedy_VFA = FqiGreedyVfaPolicy


__all__ = [
    "PostDecisionGreedyVfaPolicy",
    "FqiGreedyVfaPolicy",
    "PostDecision_Greedy_VFA",
    "FQI_Greedy_VFA",
]
