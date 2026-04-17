from __future__ import annotations

import numpy as np

# All SDAM inventory examples use vector conventions (even if dim=1)
State = np.ndarray   # shape (dS,)
Action = np.ndarray  # shape (dX,)
Exog = np.ndarray    # shape (dW,)

__all__ = ["State", "Action", "Exog"]
