from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from inventory.core.dynamics import DynamicSystemMVP
from inventory.core.policy import Policy
from inventory.core.types import Action, State


class HybridAlphaZeroPolicy(Policy):
    """Hybrid AlphaZero-style DLA+PFA/VFA policy for inventory problems.

    Decision time:
    - Runs PUCT-MCTS guided by a tiny policy+value network to produce an improved policy pi.
    - Acts greedily (argmax) by default for low-variance evaluation.

    Training time (optional):
    - Self-play data generation with MCTS targets (features, pi, return-to-go).
    - Supervised learning: CE(pi, p_theta) + value_weight * (z - v_theta)^2.
    - Optional strict-CRN evaluation gate using `DynamicSystemMVP.simulate_crn`.

        Strict CRN notes:
        - Environment randomness is controlled by `DynamicSystemMVP` via `info['crn_step_seed']`.
        - Inside this policy, MCTS is deterministic given (state, t) to keep CRN evaluation meaningful.
        - During TRAINING ONLY, we inject Dirichlet noise at the ROOT priors (classic AlphaZero trick)
            to avoid early plateaus; this is OFF in evaluation.

    Canonical notebook: `lectures/lecture_17_b_V0.1.ipynb`.
    """

    # -----------------------------
    # Tiny NumPy MLP: policy logits + value
    # -----------------------------
    class _TinyMLP_PV:
        def __init__(self, d_in: int, d_h: int, n_actions: int, seed: int = 0):
            rng = np.random.default_rng(int(seed))
            self.W1 = 0.05 * rng.standard_normal((d_h, d_in))
            self.b1 = np.zeros((d_h,), dtype=float)
            self.Wp = 0.05 * rng.standard_normal((n_actions, d_h))
            self.bp = np.zeros((n_actions,), dtype=float)
            self.Wv = 0.05 * rng.standard_normal((1, d_h))
            self.bv = np.zeros((1,), dtype=float)

        def clone(self) -> "HybridAlphaZeroPolicy._TinyMLP_PV":
            nn = HybridAlphaZeroPolicy._TinyMLP_PV(
                d_in=int(self.W1.shape[1]),
                d_h=int(self.W1.shape[0]),
                n_actions=int(self.Wp.shape[0]),
                seed=0,
            )
            nn.W1 = self.W1.copy()
            nn.b1 = self.b1.copy()
            nn.Wp = self.Wp.copy()
            nn.bp = self.bp.copy()
            nn.Wv = self.Wv.copy()
            nn.bv = self.bv.copy()
            return nn

        @staticmethod
        def _softmax(logits: np.ndarray) -> np.ndarray:
            z = logits - np.max(logits)
            p = np.exp(z)
            return p / (p.sum() + 1e-12)

        def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, np.ndarray]]:
            z1 = self.W1 @ x + self.b1
            h = np.maximum(0.0, z1)
            logits = self.Wp @ h + self.bp
            p = self._softmax(logits)
            v = float((self.Wv @ h + self.bv)[0])
            cache = {"x": x, "z1": z1, "h": h, "p": p}
            return p, v, cache

        def predict(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
            p, v, _ = self.forward(x)
            return p, v

        def train_step(
            self,
            x: np.ndarray,
            target_pi: np.ndarray,
            target_v: float,
            *,
            lr: float,
            value_weight: float = 1.0,
            l2: float = 1e-4,
        ) -> float:
            p, v, cache = self.forward(x)
            h = cache["h"]
            z1 = cache["z1"]

            ce = -float(np.sum(target_pi * np.log(p + 1e-12)))
            mse = float((target_v - v) ** 2)
            loss = ce + value_weight * mse

            # policy head
            dlogits = (p - target_pi)  # (A,)
            dWp = np.outer(dlogits, h)
            dbp = dlogits

            # value head
            dv = -2.0 * (target_v - v) * value_weight
            dWv = dv * h.reshape(1, -1)
            dbv = np.array([dv], dtype=float)

            # trunk
            dh = self.Wp.T @ dlogits + (self.Wv.T * dv).reshape(-1)
            dz1 = dh * (z1 > 0.0)

            dW1 = np.outer(dz1, cache["x"])
            db1 = dz1

            # L2
            dW1 += l2 * self.W1
            db1 += l2 * self.b1
            dWp += l2 * self.Wp
            dbp += l2 * self.bp
            dWv += l2 * self.Wv
            dbv += l2 * self.bv

            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.Wp -= lr * dWp
            self.bp -= lr * dbp
            self.Wv -= lr * dWv
            self.bv -= lr * dbv

            return float(loss)

    # -----------------------------
    # MCTS internals
    # -----------------------------
    @dataclass
    class _Edge:
        N: int = 0
        W: float = 0.0
        Q: float = 0.0

    @dataclass
    class _Node:
        N: int = 0
        priors: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
        edges: Dict[int, "HybridAlphaZeroPolicy._Edge"] = field(default_factory=dict)

    def __init__(
        self,
        model: DynamicSystemMVP,
        *,
        actions: Optional[np.ndarray] = None,
        x_max: int = 480,
        dx: int = 10,
        s_max: Optional[float] = None,
        hidden: int = 32,
        H: int = 10,
        n_sims: int = 250,
        c_puct: float = 1.5,
        gamma: float = 1.0,
        seed: int = 0,
        tau_eval: float = 0.0,
        dirichlet_alpha: float = 0.30,
        dirichlet_eps: float = 0.25,
        value_scale: float = 5000.0,
        d_state: Optional[int] = None,
    ):
        self.model = model
        self.T_default: Optional[int] = None

        self.x_max = int(x_max)
        self.dx = int(dx)
        self.s_max = None if s_max is None else float(s_max)

        self.actions = actions if actions is not None else np.arange(0, self.x_max + 1, self.dx, dtype=int)
        self.actions = np.asarray(self.actions, dtype=int)
        if self.actions.ndim != 1:
            raise ValueError("actions must be a 1-D array of integer order quantities")

        self.hidden = int(hidden)
        self.H = int(H)
        self.n_sims = int(n_sims)
        self.c_puct = float(c_puct)
        self.gamma = float(gamma)
        self.seed = int(seed)
        self.tau_eval = float(tau_eval)

        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_eps = float(dirichlet_eps)
        self.value_scale = float(value_scale)

        if self.dx <= 0:
            raise ValueError("dx must be positive")
        if self.H <= 0:
            raise ValueError("H must be >= 1")
        if self.n_sims <= 0:
            raise ValueError("n_sims must be >= 1")

        if d_state is None:
            d_state = self.model.d_s
        if d_state is None:
            raise ValueError("HybridAlphaZeroPolicy requires the system to have d_s set (or pass d_state=...).")

        self._use_regime = int(d_state) >= 2
        self.d_in = (1 + (1 if self._use_regime else 0) + 3)
        self._net = HybridAlphaZeroPolicy._TinyMLP_PV(d_in=self.d_in, d_h=self.hidden, n_actions=len(self.actions), seed=self.seed)

    # -----------------------------
    # Feature engineering (kept simple for lecture clarity)
    # -----------------------------
    @staticmethod
    def _time_features(t: int, T: int) -> np.ndarray:
        frac = 0.0 if T <= 1 else float(t) / float(T - 1)
        return np.array([frac, np.sin(2.0 * np.pi * frac), np.cos(2.0 * np.pi * frac)], dtype=float)

    def _featurize(self, state: State, t: int, T: int) -> np.ndarray:
        inv = float(state[0])
        if self._use_regime:
            reg = float(state[1])
            return np.concatenate(([inv, reg], self._time_features(t, T)), axis=0)
        return np.concatenate(([inv], self._time_features(t, T)), axis=0)

    # -----------------------------
    # Feasible actions
    # -----------------------------
    def _feasible_mask(self, state: State) -> np.ndarray:
        if self.s_max is None:
            return np.ones(len(self.actions), dtype=bool)
        inv = float(state[0])
        cap = max(0.0, float(self.s_max) - inv)
        cap_int = int(np.floor(cap + 1e-9))
        return self.actions <= cap_int

    # -----------------------------
    # Deterministic per-decision seed (matches canonical Lecture 17)
    # -----------------------------
    def _decision_seed(self, state: State, t: int) -> int:
        inv_key = int(np.round(float(state[0]) * 1000.0))
        reg_key = int(np.round(float(state[1]) * 1000.0)) if self._use_regime and state.shape[0] >= 2 else 0
        x = (inv_key * 1315423911) ^ (reg_key * 2654435761) ^ (int(t) * 97531) ^ (self.seed * 3643)
        return int(x & 0xFFFFFFFF)

    # -----------------------------
    # One MCTS improvement step at (state,t)
    # -----------------------------
    def _mcts_improved_pi(
        self,
        state: State,
        t: int,
        T: int,
        *,
        tau: float,
        info: Optional[dict],
        training: bool = False,
    ) -> np.ndarray:
        rng = np.random.default_rng(self._decision_seed(state, t))
        tree: Dict[Tuple[int, int, int], HybridAlphaZeroPolicy._Node] = {}

        def bucket_state(Sx: np.ndarray) -> Tuple[int, int]:
            inv_b = int(np.round(float(Sx[0])))
            if self._use_regime and Sx.shape[0] >= 2:
                reg_b = int(np.round(float(Sx[1])))
                return inv_b, reg_b
            return inv_b, 0

        def masked_priors(Sx: np.ndarray, tx: int) -> np.ndarray:
            x = self._featurize(Sx, tx, T)
            p, _ = self._net.predict(x)
            mask = self._feasible_mask(Sx)
            p = p * mask.astype(float)
            s = float(p.sum())
            if s <= 1e-12:
                # Fallback: uniform over feasible actions
                if mask.any():
                    p = mask.astype(float) / float(mask.sum())
                else:
                    p = np.ones_like(p) / float(len(p))
            else:
                p = p / s
            return p

        def get_node(Sx: np.ndarray, tx: int) -> HybridAlphaZeroPolicy._Node:
            inv_b, reg_b = bucket_state(Sx)
            key = (int(tx), int(inv_b), int(reg_b))
            if key in tree:
                return tree[key]
            p = masked_priors(Sx, tx)

            # AlphaZero exploration: Dirichlet noise at ROOT only (training)
            if training and (int(tx) == int(t)):
                feas = self._feasible_mask(Sx)
                noise = rng.dirichlet(self.dirichlet_alpha * np.ones(len(self.actions)))
                noise = noise * feas.astype(float)
                noise = noise / (float(noise.sum()) + 1e-12)

                p = (1.0 - self.dirichlet_eps) * p + self.dirichlet_eps * noise
                p = p / (float(p.sum()) + 1e-12)
            node = HybridAlphaZeroPolicy._Node(
                N=0,
                priors=p.copy(),
                edges={ai: HybridAlphaZeroPolicy._Edge() for ai in range(len(self.actions))},
            )
            tree[key] = node
            return node

        def select_action(node: HybridAlphaZeroPolicy._Node, Sx: np.ndarray) -> int:
            feas = self._feasible_mask(Sx)
            best_a, best_score = 0, -1e18
            sqrtN = np.sqrt(node.N + 1e-8)
            for ai, es in node.edges.items():
                if not bool(feas[ai]):
                    continue
                prior = float(node.priors[ai])
                U = self.c_puct * prior * (sqrtN / (1.0 + es.N))
                score = es.Q + U
                if score > best_score:
                    best_score = score
                    best_a = ai
            return int(best_a)

        def rollout(Sx: np.ndarray, tx: int, depth: int) -> float:
            if tx >= T:
                return 0.0
            if depth >= self.H:
                x = self._featurize(Sx, tx, T)
                _, v = self._net.predict(x)
                return float(v)

            node = get_node(Sx, tx)
            ai = select_action(node, Sx)
            x_qty = int(self.actions[ai])
            action = np.array([float(x_qty)], dtype=float)

            exog = self.model.exogenous_model.sample(Sx, action, tx, rng)
            exog = np.asarray(exog, dtype=float)
            cost = float(self.model.cost_func(Sx, action, exog, tx))
            S_next = np.asarray(self.model.transition_func(Sx, action, exog, tx), dtype=float)

            r = -cost
            G = r + self.gamma * rollout(S_next, tx + 1, depth + 1)

            node.N += 1
            es = node.edges[ai]
            es.N += 1
            es.W += G
            es.Q = es.W / es.N
            return float(G)

        state0 = np.asarray(state, dtype=float)
        for _ in range(self.n_sims):
            rollout(state0, int(t), 0)

        root = get_node(state0, int(t))
        visits = np.array([root.edges[ai].N for ai in range(len(self.actions))], dtype=float)

        feas_root = self._feasible_mask(state0)
        visits = visits * feas_root.astype(float)

        if tau <= 1e-8:
            pi = np.zeros_like(visits)
            if float(visits.sum()) <= 0.0:
                # fallback to feasible argmax under prior
                pri = root.priors * feas_root.astype(float)
                if float(pri.sum()) > 0.0:
                    ai = int(np.argmax(pri))
                else:
                    ai = int(np.argmax(feas_root.astype(int)))
            else:
                ai = int(np.argmax(visits))
            pi[ai] = 1.0
        else:
            pi = visits ** (1.0 / tau)
            s = float(pi.sum())
            if s <= 1e-12:
                pi = feas_root.astype(float)
                pi = pi / (float(pi.sum()) + 1e-12)
            else:
                pi = pi / s
        return pi

    # -----------------------------
    # Policy API
    # -----------------------------
    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        info = info or {}

        T = info.get("T", None)
        if T is None:
            T = self.T_default if self.T_default is not None else 10

        # Standard evaluation convention:
        # - deterministic=True: greedy argmax action (default)
        # - deterministic=False + det_mode="sample": sample from pi (optionally CRN-seeded)
        deterministic = bool(info.get("deterministic", True))
        det_mode = str(info.get("det_mode", "argmax"))

        tau = float(info.get("tau", self.tau_eval))
        pi = self._mcts_improved_pi(state, t, int(T), tau=float(tau), info=info, training=False)

        if (not deterministic) and det_mode == "sample":
            step_seed = info.get("crn_step_seed", None)
            if step_seed is not None:
                ss = np.random.SeedSequence([self.seed, int(step_seed), int(t)])
                rng = np.random.default_rng(int(ss.generate_state(1)[0]))
            else:
                rng = np.random.default_rng(self._decision_seed(np.asarray(state, dtype=float), int(t)))
            ai = int(rng.choice(len(self.actions), p=pi))
        else:
            ai = int(np.argmax(pi))

        x_qty = int(self.actions[ai])
        return np.array([float(x_qty)], dtype=float)

    # -----------------------------
    # Training: self-play + supervised fit (pi, return-to-go)
    # -----------------------------
    def fit_self_play(
        self,
        *,
        S0: State,
        T: int,
        n_iterations: int = 10,
        episodes_per_iter: int = 25,
        buffer_max: int = 8000,
        tau_train: float = 1.0,
        lr: float = 0.02,
        value_weight: float = 0.5,
        train_steps: int = 1200,
        batch_size: int = 64,
        gate_episodes: int = 30,
        gate_seed0: int = 2026,
        seed0: int = 0,
        verbose: bool = True,
        info: Optional[dict] = None,
    ) -> Dict[str, Any]:
        S0 = np.asarray(S0, dtype=float)
        self.T_default = int(T)

        rng = np.random.default_rng(int(seed0))
        buffer: List[Tuple[np.ndarray, np.ndarray, float]] = []
        history: Dict[str, List[Any]] = {
            "iter": [],
            "selfplay_return_mean": [],
            "gate_old_mean_cost": [],
            "gate_new_mean_cost": [],
            "accepted": [],
        }

        gate_rng = np.random.default_rng(int(gate_seed0))
        ep_seeds = [int(s) for s in gate_rng.integers(1, 2**31 - 1, size=int(gate_episodes))]
        gate_step_seed_paths = [self.model.sample_crn_step_seeds(ep_seed, int(T)) for ep_seed in ep_seeds]

        def evaluate_cost(net: HybridAlphaZeroPolicy._TinyMLP_PV) -> float:
            pol = HybridAlphaZeroPolicy(
                self.model,
                actions=self.actions,
                x_max=self.x_max,
                dx=self.dx,
                s_max=self.s_max,
                hidden=int(net.W1.shape[0]),
                H=self.H,
                n_sims=self.n_sims,
                c_puct=self.c_puct,
                gamma=self.gamma,
                seed=self.seed,
                tau_eval=0.0,
                dirichlet_alpha=self.dirichlet_alpha,
                dirichlet_eps=self.dirichlet_eps,
                value_scale=self.value_scale,
            )
            pol._net = net

            # Gate evaluation convention: deterministic.
            gate_info = {} if info is None else dict(info)
            gate_info.setdefault("deterministic", True)
            gate_info.setdefault("det_mode", "argmax")

            totals = np.empty(len(gate_step_seed_paths), dtype=float)
            for i, step_seeds in enumerate(gate_step_seed_paths):
                _, costs, _, _ = self.model.simulate_crn(pol, S0, step_seeds, info=gate_info)
                totals[i] = float(costs.sum())
            return float(totals.mean())

        for it in range(1, int(n_iterations) + 1):
            returns: List[float] = []

            for _ep in range(int(episodes_per_iter)):
                ep_seed = int(rng.integers(1, 2**31 - 1))
                ep_rng = np.random.default_rng(ep_seed)

                S = S0.copy()
                traj: List[Tuple[np.ndarray, np.ndarray, float]] = []
                total_return = 0.0

                for t in range(int(T)):
                    pi = self._mcts_improved_pi(S, t, int(T), tau=float(tau_train), info=info, training=True)
                    ai = int(ep_rng.choice(len(self.actions), p=pi))
                    x_qty = int(self.actions[ai])
                    action = np.array([float(x_qty)], dtype=float)

                    exog = self.model.exogenous_model.sample(S, action, t, ep_rng)
                    exog = np.asarray(exog, dtype=float)
                    cost = float(self.model.cost_func(S, action, exog, t))
                    S_next = np.asarray(self.model.transition_func(S, action, exog, t), dtype=float)

                    r = -cost
                    total_return += r

                    x = self._featurize(S, t, int(T))
                    traj.append((x, pi.copy(), float(r)))
                    S = S_next

                returns.append(float(total_return))

                # return-to-go targets
                G = 0.0
                for x, pi, r in reversed(traj):
                    G = float(r) + G
                    buffer.append((x, pi, float(G) / float(self.value_scale)))
                if len(buffer) > int(buffer_max):
                    buffer = buffer[-int(buffer_max) :]

            cand = self._net.clone()
            n_buf = len(buffer)
            if n_buf <= 0:
                raise RuntimeError("Empty training buffer — self-play produced no data")

            train_rng = np.random.default_rng(self.seed + 17 * it)
            for _k in range(int(train_steps)):
                idx = train_rng.integers(0, n_buf, size=int(batch_size))
                for j in idx:
                    x, pi, z = buffer[int(j)]
                    cand.train_step(x, pi, z, lr=float(lr), value_weight=float(value_weight), l2=1e-4)

            old_mean_cost = evaluate_cost(self._net)
            new_mean_cost = evaluate_cost(cand)
            accepted = bool(new_mean_cost < old_mean_cost - 1e-9)
            if accepted:
                self._net = cand

            history["iter"].append(it)
            history["selfplay_return_mean"].append(float(np.mean(returns)))
            history["gate_old_mean_cost"].append(float(old_mean_cost))
            history["gate_new_mean_cost"].append(float(new_mean_cost))
            history["accepted"].append(accepted)

            if verbose:
                print(
                    f"Iter {it:02d} | self-play mean return {np.mean(returns):.3f} (higher better) | "
                    f"gate old mean cost {old_mean_cost:.3f} vs new {new_mean_cost:.3f} | "
                    f"{'ACCEPT' if accepted else 'REJECT'}"
                )

        return history


# Notebook-compatible alias
Hybrid_AlphaZero = HybridAlphaZeroPolicy


__all__ = ["HybridAlphaZeroPolicy", "Hybrid_AlphaZero"]
