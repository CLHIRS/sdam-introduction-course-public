from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from inventory.core.dynamics import DynamicSystemMVP
from inventory.core.exogenous import ExogenousModel
from inventory.core.types import Action, Exog, State


@dataclass(frozen=True)
class Inventory1DParams:
    """Parameters for a basic 1-item inventory model."""

    price: float = 2.0
    unit_cost: float = 0.5
    holding_cost: float = 0.03
    backorder_cost: float = 1.0


def transition_inventory_1d_backorders(state: State, action: Action, exog: Exog, t: int) -> State:
    """1D inventory transition with backorders allowed.

    Conventions:
    - state[0] = inventory level (can be negative for backorders)
    - action[0] = order qty (arrives immediately in this MVP)
    - exog[0] = demand

    S_{t+1} = S_t + X_t - D_{t+1}
    """

    inv = float(state[0])
    x = float(action[0])
    d = float(exog[0])
    return np.array([inv + x - d], dtype=float)


# Module-name-friendly alias
transition_inventory_backorders = transition_inventory_1d_backorders


def cost_inventory_1d_backorders(
    state: State,
    action: Action,
    exog: Exog,
    t: int,
    *,
    params: Inventory1DParams,
) -> float:
    """Simple profit-style cost (negative reward) with holding + backorder penalty.

    This is intentionally minimal; you can evolve it in lectures.

    - Sales are limited by available inventory after ordering.
    - Backorders are penalized.

    Returns a scalar cost (lower is better).
    """

    inv0 = float(state[0])
    x = float(action[0])
    d = float(exog[0])

    available = inv0 + x
    sales = min(available, d)
    end_inv = available - d

    revenue = params.price * sales
    purchasing = params.unit_cost * x
    holding = params.holding_cost * max(end_inv, 0.0)
    backorder = params.backorder_cost * max(-end_inv, 0.0)

    profit = revenue - purchasing - holding - backorder
    return float(-profit)


# Module-name-friendly alias
cost_inventory_backorders = cost_inventory_1d_backorders


class ExogenousPoissonDemand(ExogenousModel):
    """W_{t+1} = [D] where D ~ Poisson(lam)."""

    def __init__(self, lam: float = 300.0):
        self.lam = float(lam)

    def sample(self, state: State, action: Action, t: int, rng: np.random.Generator) -> Exog:
        d = float(rng.poisson(lam=max(self.lam, 0.0)))
        return np.array([d], dtype=float)


def make_inventory_1d_system(
    *,
    exogenous_model: ExogenousModel,
    cost_params: Optional[Inventory1DParams] = None,
    sim_seed: int = 42,
) -> DynamicSystemMVP:
    """Convenience factory returning a configured DynamicSystemMVP for 1D inventory."""

    params = Inventory1DParams() if cost_params is None else cost_params

    def _cost(s: State, a: Action, w: Exog, t: int) -> float:
        return cost_inventory_1d_backorders(s, a, w, t, params=params)

    return DynamicSystemMVP(
        transition_func=transition_inventory_1d_backorders,
        cost_func=_cost,
        exogenous_model=exogenous_model,
        sim_seed=int(sim_seed),
        d_s=1,
        d_x=1,
        d_w=1,
    )


# Module-name-friendly alias
make_inventory_backorders_system = make_inventory_1d_system


__all__ = [
    "Inventory1DParams",
    "transition_inventory_1d_backorders",
    "transition_inventory_backorders",
    "cost_inventory_1d_backorders",
    "cost_inventory_backorders",
    "ExogenousPoissonDemand",
    "make_inventory_1d_system",
    "make_inventory_backorders_system",
]
