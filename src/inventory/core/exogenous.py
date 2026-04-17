from __future__ import annotations

import numpy as np
from inventory.core.types import Action, Exog, State


class ExogenousModel:
    """Official exogenous interface.

    Required:
      sample(S, X, t, rng) -> W_{t+1} (exogenous vector)

    Notes:
    - May depend on (S, X, t) to support regimes/queues/etc.
    - For strict CRN, the caller controls `rng` via per-step seeds.
    """

    def sample(self, state: State, action: Action, t: int, rng: np.random.Generator) -> Exog:
        raise NotImplementedError


__all__ = ["ExogenousModel"]
