from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from inventory.core.dynamics import DynamicSystemMVP
from inventory.core.exogenous import ExogenousModel
from inventory.core.types import Action, Exog, State


@dataclass(frozen=True)
class InventoryCostParams:
    price: float = 2.0
    unit_cost: float = 0.5
    holding_cost: float = 0.03
    stockout_penalty: float = 1.0


def inventory_cost(
    state: State,
    action: Action,
    exog: Exog,
    t: int,
    *,
    p: float = 2.0,
    c: float = 0.5,
    h: float = 0.03,
    b: float = 1.0,
) -> float:
    """Baseline one-step cost (lower is better) with revenue, purchase, holding, stockout.

    Conventions:
    - state[0] start-of-period on-hand inventory
    - action[0] order quantity arriving immediately
    - exog[0] demand

    cost = purchase_cost + holding_cost + stockout_penalty - revenue

    This matches the `minimal_baseline.py` semantics (lost-sales, nonnegative end inventory).
    """

    inv = float(state[0])
    order = float(action[0])
    demand = float(exog[0])

    on_hand = inv + order
    sales = min(on_hand, demand)
    lost = max(0.0, demand - on_hand)
    inv_end = max(0.0, on_hand - demand)

    revenue = float(p) * sales
    purchase_cost = float(c) * order
    holding_cost = float(h) * inv_end
    stockout_penalty = float(b) * lost

    return float(purchase_cost + holding_cost + stockout_penalty - revenue)


def inventory_cost_extended(
    state: State,
    action: Action,
    exog: Exog,
    t: int,
    *,
    p: float = 2.0,
    c: float = 0.5,
    h: float = 0.03,
    b: float = 6.0,
    K: float = 50.0,
) -> float:
    """Extended one-step cost with an additional fixed setup cost when ordering.

    Conventions (same as `inventory_cost`):
    - state[0] start-of-period on-hand inventory
    - action[0] order quantity arriving immediately
    - exog[0] demand

    cost = purchase_cost + setup_cost + holding_cost + stockout_penalty - revenue
    where setup_cost = K if order > 0 else 0.
    """

    inv = float(state[0])
    order = float(action[0])
    demand = float(exog[0])

    on_hand = inv + order
    sales = min(on_hand, demand)
    lost = max(0.0, demand - on_hand)
    inv_end = max(0.0, on_hand - demand)

    revenue = float(p) * sales
    purchase_cost = float(c) * order
    setup_cost = float(K) if order > 0.0 else 0.0
    holding_cost = float(h) * inv_end
    stockout_penalty = float(b) * lost

    return float(purchase_cost + setup_cost + holding_cost + stockout_penalty - revenue)


def inventory_transition(state: State, action: Action, exog: Exog, t: int) -> State:
    """Baseline inventory transition.

    Case A (no regime):
      S = [inventory]
      X = [order_qty]
      W = [demand]
      -> S_next = [max(0, inventory + order - demand)]

    Case B (observable regime):
      S = [inventory, regime]
      X = [order_qty]
      W = [demand, regime_next]
      -> S_next = [max(0, inv + order - demand), regime_next]

        Case C (multiple observable regimes):
            S = [inventory, r1, r2, ...]
            X = [order_qty]
            W = [demand, r1_next, r2_next, ...]
            -> S_next = [max(0, inv + order - demand), r1_next, r2_next, ...]

        Notes:
        - If regime components exist in state but are missing from exog, the values are
            carried forward from state.
    """

    inv = float(state[0])
    order = float(action[0])
    demand = float(exog[0])

    inv_next = max(0.0, inv + order - demand)

    d_s = int(state.shape[0])
    if d_s == 1:
        return np.array([inv_next], dtype=float)

    next_components = [float(inv_next)]
    for i in range(1, d_s):
        next_components.append(float(exog[i]) if int(exog.shape[0]) > i else float(state[i]))
    return np.asarray(next_components, dtype=float)


def inventory_transition_capped_1d(state: State, action: Action, exog: Exog, t: int, *, s_max: int) -> State:
    inv = float(state[0])
    order = float(action[0])
    demand = float(exog[0])

    inv_next = inv + order - demand
    inv_next = max(0.0, inv_next)
    inv_next = min(float(s_max), inv_next)
    return np.array([inv_next], dtype=float)


def inventory_transition_regime_capped(state: State, action: Action, exog: Exog, t: int, *, s_max: int) -> State:
    """Regime-observable transition with nonnegativity + capacity cap.

    Supports the common 2D case (inventory + 1 regime) and also the multi-regime
    case where additional observable regime components are appended to the state.
    """

    inv = float(state[0])
    order = float(action[0])
    demand = float(exog[0])

    inv_next = inv + order - demand
    inv_next = max(0.0, inv_next)
    inv_next = min(float(s_max), inv_next)

    d_s = int(state.shape[0])
    if d_s == 1:
        return np.array([inv_next], dtype=float)

    next_components = [float(inv_next)]
    for i in range(1, d_s):
        next_components.append(float(exog[i]) if int(exog.shape[0]) > i else float(state[i]))
    return np.asarray(next_components, dtype=float)


def make_inventory_mvp_system(
    *,
    exogenous_model: Optional[ExogenousModel] = None,
    exogenous: Optional[ExogenousModel] = None,
    T: Optional[int] = None,
    cost_params: Optional[InventoryCostParams] = None,
    s_max: Optional[int] = None,
    sim_seed: int = 42,
    d_s: Optional[int] = None,
) -> DynamicSystemMVP:
    """Factory for the minimal-baseline inventory MVP system.

    - Uses `inventory_cost` and `inventory_transition`.
    - Supports state dim 1 (no regime) or dim 2 (inventory+regime).

    Provide `d_s` explicitly if you want strict dimension checking.

    Backwards-compatibility:
    - `exogenous` is accepted as an alias for `exogenous_model`.
    - `T` is accepted but unused (horizon is chosen at evaluation/rollout time).
    """

    if exogenous_model is None and exogenous is None:
        raise TypeError("make_inventory_mvp_system requires `exogenous_model` (or alias `exogenous`).")
    if exogenous_model is not None and exogenous is not None:
        raise TypeError("Specify only one of `exogenous_model` or `exogenous`.")
    exogenous_model = exogenous_model if exogenous_model is not None else exogenous
    _ = T  # intentionally unused

    if d_s is None:
        # Infer state dimension for common exogenous models.
        # - demand-only exogenous -> S=[inventory]
        # - single-regime exogenous -> S=[inventory, regime]
        # - multi-regime exogenous -> S=[inventory, season, day, weather]
        if getattr(exogenous_model, "regime_index", None) is not None and hasattr(exogenous_model, "P"):
            d_s = 2
        elif all(hasattr(exogenous_model, a) for a in ("P_season", "P_day", "P_weather")):
            d_s = 4
        else:
            d_s = 1

    params = InventoryCostParams() if cost_params is None else cost_params

    def _cost(s: State, a: Action, w: Exog, t: int) -> float:
        return inventory_cost(s, a, w, t, p=params.price, c=params.unit_cost, h=params.holding_cost, b=params.stockout_penalty)

    transition_func = inventory_transition
    if s_max is not None:
        if int(d_s) == 1:
            transition_func = lambda s, a, w, t: inventory_transition_capped_1d(s, a, w, t, s_max=int(s_max))
        else:
            transition_func = lambda s, a, w, t: inventory_transition_regime_capped(s, a, w, t, s_max=int(s_max))

    return DynamicSystemMVP(
        transition_func=transition_func,
        cost_func=_cost,
        exogenous_model=exogenous_model,
        sim_seed=int(sim_seed),
        d_s=d_s,
        d_x=1,
        d_w=int(d_s) if int(d_s) > 1 else 1,
    )


def make_inventory_multi_regime_system(
    *,
    exogenous_model: Optional[ExogenousModel] = None,
    exogenous: Optional[ExogenousModel] = None,
    s_max: Optional[int] = None,
    sim_seed: int = 42,
    p: float = 2.0,
    c: float = 0.5,
    h: float = 0.03,
    b: float = 6.0,
    K: float = 50.0,
) -> DynamicSystemMVP:
    """Factory for the multi-regime (4D) inventory system.

    Intended for `ExogenousPoissonMultiRegime`-style exogenous models where:
    - state is `S=[inventory, season, day, weather]` (d_s=4)
    - exog is `W=[demand, season_next, day_next, weather_next]` (d_w=4)

    Uses `inventory_transition` (which supports multi-regime propagation) and
    `inventory_cost_extended` by default.
    """

    if exogenous_model is None and exogenous is None:
        raise TypeError("make_inventory_multi_regime_system requires `exogenous_model` (or alias `exogenous`).")
    if exogenous_model is not None and exogenous is not None:
        raise TypeError("Specify only one of `exogenous_model` or `exogenous`.")
    exogenous_model = exogenous_model if exogenous_model is not None else exogenous

    def _cost(s: State, a: Action, w: Exog, t: int) -> float:
        return inventory_cost_extended(s, a, w, t, p=p, c=c, h=h, b=b, K=K)

    transition_func = inventory_transition
    if s_max is not None:
        transition_func = lambda s, a, w, t: inventory_transition_regime_capped(s, a, w, t, s_max=int(s_max))

    return DynamicSystemMVP(
        transition_func=transition_func,
        cost_func=_cost,
        exogenous_model=exogenous_model,
        sim_seed=int(sim_seed),
        d_s=4,
        d_x=1,
        d_w=4,
    )


__all__ = [
    "InventoryCostParams",
    "inventory_cost",
    "inventory_cost_extended",
    "inventory_transition",
    "inventory_transition_capped_1d",
    "inventory_transition_regime_capped",
    "make_inventory_mvp_system",
    "make_inventory_multi_regime_system",
]
