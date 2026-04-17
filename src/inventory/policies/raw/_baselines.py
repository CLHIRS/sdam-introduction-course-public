from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from inventory.core.policy import Policy
from inventory.core.types import Action, State


@dataclass(frozen=True)
class OrderUpToPolicy(Policy):
    """Order-up-to policy (vector format).

    Conventions:
    - Inventory is stored in `state[0]`.
    - Action is length-1 vector: `action[0] = order_qty`.

    Parameters:
    - target_level: desired inventory position
    - x_max: optional cap on order quantity
    - dx: optional batch size (multiples of dx)
    """

    target_level: float
    x_max: Optional[float] = None
    dx: Optional[int] = None

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        on_hand = float(state[0])
        x = max(0.0, float(self.target_level) - on_hand)

        if self.x_max is not None:
            x = min(x, float(self.x_max))

        if self.dx is not None and int(self.dx) > 1:
            dx = int(self.dx)
            x = dx * np.round(x / dx)
            x = max(0.0, x)
            if self.x_max is not None:
                x = min(x, float(self.x_max))

        x_int = int(np.round(float(x)))
        return np.array([x_int], dtype=float)


@dataclass(frozen=True)
class OrderUpToCapacityPolicy(Policy):
    """Order-up-to policy with capacity feasibility.

    Constraints enforced:
    - x in multiples of dx
    - x <= x_max
    - inv + x <= s_max
    """

    target_level: float
    x_max: float
    dx: int
    s_max: float

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        inv = float(state[0])
        x = max(0.0, float(self.target_level) - inv)
        x = min(x, float(self.x_max))
        x = min(x, max(0.0, float(self.s_max) - inv))

        if int(self.dx) > 1:
            dx = int(self.dx)
            x = dx * np.round(x / dx)
            x = max(0.0, x)
            x = min(x, float(self.x_max))
            x = min(x, max(0.0, float(self.s_max) - inv))

        x = float(int(np.round(float(x))))
        return np.array([x], dtype=float)


# Backwards-compatible alias (older notebooks/scripts)
PolicyOrderUpToCapacity = OrderUpToCapacityPolicy


__all__ = ["OrderUpToPolicy", "OrderUpToCapacityPolicy", "PolicyOrderUpToCapacity"]
