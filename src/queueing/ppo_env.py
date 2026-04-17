# queueing/ppo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RoutingPPOAgent:
    """
    Minimal placeholder agent with a tiny linear policy.
    Replace with a real PPO later (torch, etc.).
    """
    obs_dim: int
    max_actions: int
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(scale=0.05, size=(self.max_actions, self.obs_dim)).astype(np.float32)

    def act(self, dp_id: str, obs: np.ndarray, mask: np.ndarray, deterministic: bool = True) -> Tuple[int, float, float]:
        obs = np.asarray(obs, dtype=np.float32)
        logits = self.W @ obs  # [A]
        logits = np.where(mask, logits, -1e9)

        if deterministic:
            a = int(np.argmax(logits))
        else:
            # softmax sampling
            z = logits - np.max(logits)
            p = np.exp(z) * mask
            s = float(np.sum(p))
            if s <= 0:
                a = int(np.flatnonzero(mask)[0]) if np.any(mask) else 0
            else:
                p = p / s
                a = int(np.random.choice(len(p), p=p))
        # dummy logp/value
        return a, 0.0, 0.0


def train_ppo_stub(*args, **kwargs) -> RoutingPPOAgent:
    raise NotImplementedError("Replace with your real PPO trainer (torch or numpy) later.")