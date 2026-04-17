from __future__ import annotations

from typing import Optional

import numpy as np
from inventory.core.types import Action, Exog, State


def inventory_transition_leadtime(
    state: State,
    action: Action,
    exog: Exog,
    t: Optional[int] = None,
    *,
    L: int = 2,
) -> State:
    """Inventory transition with fixed lead time (lost-sales) using an explicit pipeline.

    Notebook conventions (Lecture 12_b):
    - State:  state = [I_t, p1_t, p2_t, ..., pL_t]  (length 1+L)
        - I_t  : on-hand inventory
        - pk_t : pipeline arriving in k steps
    - Action: action = [x_t] (order placed now, arrives in L steps)
    - Exog:   exog  = [D_{t+1}] demand

    Transition steps:
    1) Receive arriving pipeline p1
    2) Satisfy demand (lost sales)
    3) Shift pipeline left and insert new order at the end
    """

    _ = t  # intentionally unused (kept for DynamicSystemMVP signature compatibility)

    L = int(L)
    I = float(state[0])
    pipe = np.asarray(state[1 : 1 + L], dtype=float)  # shape (L,)
    x = float(action[0])
    d = float(exog[0])

    arriving = pipe[0] if L > 0 else 0.0
    on_hand = I + arriving

    # lost sales
    _sales = min(on_hand, d)
    I_next = max(0.0, on_hand - d)

    if L > 0:
        pipe_next = np.empty_like(pipe)
        pipe_next[:-1] = pipe[1:]
        pipe_next[-1] = x
        S_next = np.concatenate([[I_next], pipe_next])
    else:
        S_next = np.array([I_next], dtype=float)

    return S_next


def inventory_cost_leadtime(
    state: State,
    action: Action,
    exog: Exog,
    t: Optional[int] = None,
    *,
    p: float = 2.0,
    c: float = 0.5,
    h: float = 0.03,
    b: float = 1.0,
    L: int = 2,
) -> float:
    """One-step cost for the lead-time inventory model (lost-sales).

    Same economics as the baseline inventory model:
      cost = purchase + holding + stockout_penalty - revenue

    but sales happen after receiving the pipeline arrival for this step.
    """

    _ = t  # intentionally unused (kept for DynamicSystemMVP signature compatibility)

    L = int(L)
    I = float(state[0])
    pipe = np.asarray(state[1 : 1 + L], dtype=float) if L > 0 else np.zeros(0)
    x = float(action[0])
    d = float(exog[0])

    arriving = pipe[0] if L > 0 else 0.0
    on_hand = I + arriving

    sales = min(on_hand, d)
    lost = max(0.0, d - on_hand)
    I_next = max(0.0, on_hand - d)

    revenue = float(p) * sales
    purchase_cost = float(c) * x
    holding_cost = float(h) * I_next
    stockout_penalty = float(b) * lost

    return float(purchase_cost + holding_cost + stockout_penalty - revenue)


__all__ = ["inventory_transition_leadtime", "inventory_cost_leadtime"]
