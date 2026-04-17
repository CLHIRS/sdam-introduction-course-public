# queueing/routing_rules.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


class RoutingRule:
    def route(self, rng: np.random.Generator, from_node: int) -> Optional[int]:
        raise NotImplementedError


@dataclass(frozen=True)
class MarkovRouting(RoutingRule):
    """
    P[i,j] = prob route i->j; exit prob = 1 - sum_j P[i,j]
    """
    P: np.ndarray

    def __post_init__(self) -> None:
        if self.P.ndim != 2 or self.P.shape[0] != self.P.shape[1]:
            raise ValueError("P must be square")
        if np.any(self.P < -1e-12):
            raise ValueError("P must be nonnegative")
        if np.any(self.P.sum(axis=1) > 1.0 + 1e-9):
            raise ValueError("each row sum must be <= 1")

    def route(self, rng: np.random.Generator, from_node: int) -> Optional[int]:
        probs = self.P[int(from_node)]
        s = float(probs.sum())
        u = float(rng.random())
        if u >= s:
            return None
        cs = np.cumsum(probs)
        j = int(np.searchsorted(cs, u, side="right"))
        return j