from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from inventory.core.policy import Policy
from inventory.core.types import Action, State

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from inventory.core.dynamics import DynamicSystemMVP
    from torch import Tensor
else:  # pragma: no cover
    DynamicSystemMVP = Any  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment]

_TORCH_AVAILABLE = torch is not None and nn is not None and optim is not None


@dataclass(frozen=True)
class PPOHyperParams:
    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_eps: float = 0.20
    vf_coef: float = 0.50
    ent_coef: float = 0.01
    max_grad_norm: float = 0.50

    lr: float = 3e-4
    n_epochs: int = 10
    minibatch_size: int = 256


class _RunningNorm:
    """Running mean/std normalizer for state vectors (Welford)."""

    def __init__(self, d: int, eps: float = 1e-8):
        self.d = int(d)
        self.eps = float(eps)
        self.n = 0
        self.mean = np.zeros(self.d, dtype=np.float64)
        self.M2 = np.zeros(self.d, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        for row in x:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            delta2 = row - self.mean
            self.M2 += delta * delta2

    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones(self.d, dtype=np.float64)
        var = self.M2 / (self.n - 1)
        return np.sqrt(np.maximum(var, self.eps))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean.astype(np.float32)) / self.std().astype(np.float32)


if _TORCH_AVAILABLE:
    class _ActorCriticNet(nn.Module):
        def __init__(self, d_s: int, n_a: int, hidden: int = 128):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(d_s, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )
            self.policy_head = nn.Linear(hidden, n_a)
            self.value_head = nn.Linear(hidden, 1)

        def forward(self, s: Tensor):
            h = self.shared(s)
            logits = self.policy_head(h)
            value = self.value_head(h).squeeze(-1)
            return logits, value
else:  # pragma: no cover
    _ActorCriticNet = object  # type: ignore[assignment]


class HybridPpoPolicy(Policy):
    """Lecture 16 hybrid: PPO actor-critic (PFA trained using a VFA critic).

    - Discrete action grid {0, dx, 2dx, ..., x_max}
    - Feasibility enforced by masking: inv + x <= S_max
    - Reward = -cost (min-cost objective)

    Strict-CRN: if `info['crn_step_seed']` is present and the policy is sampling stochastically,
    it will sample using that seed so decisions are reproducible under CRN.
    """

    def __init__(
        self,
        *,
        d_s: int,
        s_max: int = 480,
        x_max: int = 480,
        dx: int = 10,
        hidden: int = 128,
        device: str = "cpu",
        hparams: PPOHyperParams = PPOHyperParams(),
        seed: int = 0,
        deterministic_eval: bool = False,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("HybridPpoPolicy requires PyTorch (`pip install torch`).")

        self.d_s = int(d_s)
        self.s_max = int(s_max)
        self.x_max = int(x_max)
        self.dx = int(dx)

        if self.dx <= 0:
            raise ValueError("dx must be >= 1")
        if self.x_max < 0 or self.s_max < 0:
            raise ValueError("x_max and s_max must be >= 0")
        if self.x_max % self.dx != 0:
            raise ValueError("x_max should be divisible by dx for a clean discrete action grid")

        self.action_grid = np.arange(0, self.x_max + 1, self.dx, dtype=np.int64)
        self.n_a = int(len(self.action_grid))

        self.device = torch.device(device)
        self.hparams = hparams
        self.seed = int(seed)
        self.deterministic_eval = bool(deterministic_eval)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.net = _ActorCriticNet(self.d_s, self.n_a, hidden=int(hidden)).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=float(self.hparams.lr))
        self.norm = _RunningNorm(self.d_s)

    # -----------------------------
    # Gate helpers (save/restore)
    # -----------------------------
    def get_params(self) -> Dict[str, Any]:
        return {
            "net": {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()},
            "norm_n": int(self.norm.n),
            "norm_mean": self.norm.mean.copy(),
            "norm_M2": self.norm.M2.copy(),
        }

    def set_params(self, params: Dict[str, Any]) -> None:
        self.net.load_state_dict(params["net"])
        self.norm.n = int(params["norm_n"])
        self.norm.mean = np.asarray(params["norm_mean"], dtype=np.float64).copy()
        self.norm.M2 = np.asarray(params["norm_M2"], dtype=np.float64).copy()

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _state_tensor(self, state: State) -> Tensor:
        s = self.norm.normalize(np.asarray(state, dtype=np.float32))
        return torch.tensor(s, dtype=torch.float32, device=self.device)

    def _masked_logits(self, logits_1d: Tensor, inv: float) -> Tensor:
        feasible = (inv + self.action_grid.astype(np.float64)) <= (float(self.s_max) + 1e-9)
        if not np.any(feasible):
            feasible = self.action_grid == 0

        mask = torch.tensor(feasible, dtype=torch.bool, device=logits_1d.device)
        out = logits_1d.clone()
        out[~mask] = -1e9
        return out

    @staticmethod
    def _numpy_choice_from_logits(logits_1d: Tensor, *, seed: int) -> int:
        probs = torch.softmax(logits_1d.detach().cpu(), dim=-1).numpy().astype(np.float64)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
        rng = np.random.default_rng(int(seed))
        return int(rng.choice(len(probs), p=probs))

    # -----------------------------
    # Policy API
    # -----------------------------
    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        """Choose an action.

        Matches Lecture 16 "LASTEST" behavior:
        - Stochastic sampling by default (needed for training)
        - If deterministic, supports:
          - det_mode="mean" (default): risk-reduced mean-action on the dx-grid
          - det_mode="argmax": greedy mode, optionally penalized by risk_alpha

        Strict-CRN extension (package): if `info['crn_step_seed']` is provided and the policy
        is sampling stochastically, the decision sampling is made reproducible under CRN.
        """

        info = info or {}
        deterministic = self.deterministic_eval or bool(info.get("deterministic", False))
        det_mode = str(info.get("det_mode", "mean"))
        risk_alpha = float(info.get("risk_alpha", 0.0))

        inv = float(np.asarray(state, dtype=float).reshape(-1)[0])

        with torch.no_grad():
            s_t = self._state_tensor(state)
            logits, _ = self.net(s_t)
            logits = self._masked_logits(logits, inv)

            dist = torch.distributions.Categorical(logits=logits)

            if not deterministic:
                crn_step_seed = info.get("crn_step_seed", None)
                if crn_step_seed is not None:
                    ss = np.random.SeedSequence([self.seed, int(crn_step_seed), int(t)])
                    a_idx = self._numpy_choice_from_logits(logits, seed=int(ss.generate_state(1)[0]))
                else:
                    a_idx = int(dist.sample().item())

                x = int(self.action_grid[a_idx])
                return np.array([float(x)], dtype=float)

            # ------------------------------------------------------
            # Deterministic evaluation modes (Lecture 16)
            # ------------------------------------------------------
            probs = dist.probs.detach().cpu().numpy().astype(np.float64)

            if det_mode == "argmax":
                if risk_alpha > 0.0:
                    scores = np.log(probs + 1e-12) - risk_alpha * (self.action_grid.astype(np.float64) / float(self.s_max))
                    a_idx = int(np.argmax(scores))
                else:
                    a_idx = int(np.argmax(probs))

                x = int(self.action_grid[a_idx])
                return np.array([float(x)], dtype=float)

            # det_mode == "mean" (default)
            x_mean = float(np.dot(probs, self.action_grid.astype(np.float64)))

            x_rounded = int(self.dx * np.round(x_mean / float(self.dx)))
            x_rounded = int(np.clip(x_rounded, 0, self.x_max))

            # Enforce feasibility inv + x <= s_max
            x_cap = int(max(0, int(self.s_max - inv)))
            x_rounded = int(min(x_rounded, x_cap))

            # Snap again to grid after clipping (use floor to stay feasible)
            x_rounded = int(self.dx * np.floor(x_rounded / float(self.dx)))

            return np.array([float(x_rounded)], dtype=float)

    # -----------------------------
    # PPO training
    # -----------------------------
    def train_ppo(
        self,
        *,
        system: DynamicSystemMVP,
        S0: np.ndarray,
        T: int,
        n_episodes: int,
        seed0: int = 1234,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        hp = self.hparams
        T = int(T)
        n_episodes = int(n_episodes)

        states: List[np.ndarray] = []
        actions: List[int] = []
        old_logps: List[float] = []
        values: List[float] = []
        rewards: List[float] = []

        rng = np.random.default_rng(int(seed0))
        episode_seeds = [int(s) for s in rng.integers(1, 2**31 - 1, size=n_episodes)]

        self.norm.update(np.asarray(S0, dtype=np.float64))

        self.net.train()
        for ep_seed in episode_seeds:
            traj, costs, acts, _ = system.simulate(self, S0, T=T, seed=ep_seed, info=None)

            r = (-costs).astype(np.float32)
            self.norm.update(traj[:-1])

            for tt in range(T):
                s_t_np = traj[tt].astype(np.float32)
                inv = float(s_t_np[0])

                a_val = int(np.round(float(acts[tt, 0])))
                idxs = np.where(self.action_grid == a_val)[0]
                if len(idxs) == 0:
                    raise ValueError(f"Encountered action {a_val} not on action_grid.")
                a_idx = int(idxs[0])

                s_t = self._state_tensor(s_t_np)
                logits, v = self.net(s_t)
                logits = self._masked_logits(logits, inv)
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(torch.tensor(a_idx, device=self.device))

                states.append(s_t_np)
                actions.append(a_idx)
                old_logps.append(float(lp.detach().cpu().item()))
                values.append(float(v.detach().cpu().item()))
                rewards.append(float(r[tt]))

        states_np = np.asarray(states, dtype=np.float32)
        actions_np = np.asarray(actions, dtype=np.int64)
        old_logps_np = np.asarray(old_logps, dtype=np.float32)
        values_np = np.asarray(values, dtype=np.float32)
        rewards_np = np.asarray(rewards, dtype=np.float32)

        N = len(rewards_np)

        adv = np.zeros(N, dtype=np.float32)
        ret = np.zeros(N, dtype=np.float32)

        for ep in range(n_episodes):
            start = ep * T
            end = start + T

            v_ep = values_np[start:end]
            r_ep = rewards_np[start:end]

            v_next = np.append(v_ep[1:], 0.0).astype(np.float32)

            gae = 0.0
            for tt in reversed(range(T)):
                delta = r_ep[tt] + hp.gamma * v_next[tt] - v_ep[tt]
                gae = delta + hp.gamma * hp.gae_lambda * gae
                adv[start + tt] = gae

            ret[start:end] = adv[start:end] + v_ep

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states_t = torch.tensor(self.norm.normalize(states_np), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions_np, dtype=torch.int64, device=self.device)
        old_logps_t = torch.tensor(old_logps_np, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=self.device)

        losses = {"policy_loss": [], "value_loss": [], "entropy": [], "total_loss": []}
        idxs = np.arange(N)

        action_grid_f64 = self.action_grid.astype(np.float64)

        for epoch in range(int(hp.n_epochs)):
            np.random.shuffle(idxs)
            for mb_start in range(0, N, int(hp.minibatch_size)):
                mb = idxs[mb_start : mb_start + int(hp.minibatch_size)]

                s_mb = states_t[mb]
                a_mb = actions_t[mb]
                oldlp_mb = old_logps_t[mb]
                adv_mb = adv_t[mb]
                ret_mb = ret_t[mb]

                logits, v = self.net(s_mb)

                # Apply the SAME feasibility mask as in act()
                inv_mb = states_np[mb, 0].astype(np.float64)
                feasible = (inv_mb[:, None] + action_grid_f64[None, :]) <= (float(self.s_max) + 1e-9)
                if not np.all(feasible.any(axis=1)):
                    feasible = feasible.copy()
                    bad = ~feasible.any(axis=1)
                    feasible[bad, :] = action_grid_f64[None, :] == 0.0

                mask_t = torch.tensor(feasible, dtype=torch.bool, device=self.device)
                logits = logits.masked_fill(~mask_t, -1e9)

                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(a_mb)
                entropy = dist.entropy().mean()

                ratio = torch.exp(lp - oldlp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - hp.clip_eps, 1.0 + hp.clip_eps) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (ret_mb - v).pow(2).mean()
                total_loss = policy_loss + hp.vf_coef * value_loss - hp.ent_coef * entropy

                self.opt.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), hp.max_grad_norm)
                self.opt.step()

                losses["policy_loss"].append(float(policy_loss.detach().cpu().item()))
                losses["value_loss"].append(float(value_loss.detach().cpu().item()))
                losses["entropy"].append(float(entropy.detach().cpu().item()))
                losses["total_loss"].append(float(total_loss.detach().cpu().item()))

            if verbose:
                k = max(1, N // int(hp.minibatch_size))
                pl = float(np.mean(losses["policy_loss"][-k:]))
                vl = float(np.mean(losses["value_loss"][-k:]))
                ent = float(np.mean(losses["entropy"][-k:]))
                print(f"[PPO] epoch {epoch+1:02d}/{hp.n_epochs} | pi={pl:.4f} v={vl:.4f} ent={ent:.4f}")

        self.net.eval()
        return losses


def train_ppo_with_eval_gate(
    *,
    system: DynamicSystemMVP,
    ppo: HybridPpoPolicy,
    baseline_policy: Policy,
    S0: np.ndarray,
    T: int,
    total_train_episodes: int = 1200,
    chunk_episodes: int = 200,
    eval_episodes: int = 300,
    eval_seed0: int = 999,
    train_seed0: int = 1234,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """PPO training with an evaluation gate (AlphaZero-style accept/reject)."""

    if total_train_episodes % chunk_episodes != 0:
        raise ValueError("total_train_episodes must be divisible by chunk_episodes")

    history: List[Dict[str, Any]] = []
    n_gates = total_train_episodes // chunk_episodes

    init_results, _ = system.evaluate_policies_crn_mc(
        {"Baseline": baseline_policy, "PPO": ppo},
        S0=S0,
        T=int(T),
        n_episodes=int(eval_episodes),
        seed0=int(eval_seed0),
    )

    best_params = ppo.get_params()
    best_ppo_mean = float(init_results["PPO"].mean)
    base_mean = float(init_results["Baseline"].mean)

    if verbose:
        print(f"[Gate 0] Baseline mean={base_mean:.3f} | PPO mean={best_ppo_mean:.3f} (lower is better)")

    for g in range(1, n_gates + 1):
        ppo.train_ppo(
            system=system,
            S0=S0,
            T=int(T),
            n_episodes=int(chunk_episodes),
            seed0=int(train_seed0) + 10_000 * int(g),
            verbose=False,
        )

        results, _ = system.evaluate_policies_crn_mc(
            {"Baseline": baseline_policy, "PPO": ppo},
            S0=S0,
            T=int(T),
            n_episodes=int(eval_episodes),
            seed0=int(eval_seed0),
        )

        base_mean = float(results["Baseline"].mean)
        cand_mean = float(results["PPO"].mean)

        accept = cand_mean < best_ppo_mean
        if accept:
            best_ppo_mean = cand_mean
            best_params = ppo.get_params()
            decision = "ACCEPT"
        else:
            ppo.set_params(best_params)
            decision = "REJECT"

        history.append(
            {
                "gate": int(g),
                "baseline_mean": base_mean,
                "candidate_ppo_mean": cand_mean,
                "best_ppo_mean": best_ppo_mean,
                "accepted": bool(accept),
            }
        )

        if verbose:
            print(
                f"[Gate {g}] Baseline mean={base_mean:.3f} | Candidate PPO mean={cand_mean:.3f} | "
                f"Best PPO mean={best_ppo_mean:.3f} -> {decision}"
            )

    return history


# Notebook-compatibility aliases (Lecture 16 naming)
Hybrid_PPO = HybridPpoPolicy


__all__ = [
    "PPOHyperParams",
    "HybridPpoPolicy",
    "Hybrid_PPO",
    "train_ppo_with_eval_gate",
]
