from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from inventory.core.policy import Policy
from inventory.core.types import Action, State



class OrderRandomPolicy(Policy):
    """
    Random order on grid [x_min, x_max] with spacing dx.

    distr='equal'  -> uniform over all grid points
    distr='normal' -> normal around (x_mean, x_std), then clipped + snapped to grid
    """

    def __init__(
        self,
        *,
        x_min: float = 0.0,
        x_max: float = 480.0,
        dx: float = 10.0,
        seed: Optional[int] = None,
        distr: str = "equal",
        x_mean: Optional[float] = None,
        x_std: Optional[float] = None,
    ):
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.dx = float(dx)
        self.seed = None if seed is None else int(seed)

        self.distr = str(distr).lower().strip()
        self.x_mean = None if x_mean is None else float(x_mean)
        self.x_std = None if x_std is None else float(x_std)

        if self.dx <= 0:
            raise ValueError("dx must be > 0.")
        if self.x_max < self.x_min:
            raise ValueError("x_max must be >= x_min.")

        span = (self.x_max - self.x_min) / self.dx
        n_steps = int(round(span))
        if not np.isclose(span, n_steps):
            raise ValueError("x_max - x_min must be an integer multiple of dx.")

        self._n_actions = n_steps + 1
        self._grid = self.x_min + self.dx * np.arange(self._n_actions, dtype=float)
        self._rng = np.random.default_rng(self.seed)

        if self.distr not in {"equal", "normal"}:
            raise ValueError("distr must be either 'equal' or 'normal'.")

        if self.distr == "normal":
            if self.x_mean is None or self.x_std is None:
                raise ValueError("For distr='normal', provide x_mean and x_std.")
            if self.x_std <= 0:
                raise ValueError("x_std must be > 0 for distr='normal'.")

    def __repr__(self) -> str:
        return (
            "OrderRandomPolicy("
            f"x_min={self.x_min}, x_max={self.x_max}, dx={self.dx}, "
            f"seed={self.seed}, distr='{self.distr}', "
            f"x_mean={self.x_mean}, x_std={self.x_std})"
        )

    def _sample_action_from_rng(self, rng: np.random.Generator) -> np.ndarray:
        if self.distr == "equal":
            k = int(rng.integers(0, self._n_actions))
            return np.array([self._grid[k]], dtype=float)

        x = float(rng.normal(loc=self.x_mean, scale=self.x_std))
        x = float(np.clip(x, self.x_min, self.x_max))
        k = int(np.rint((x - self.x_min) / self.dx))
        k = max(0, min(k, self._n_actions - 1))
        return np.array([self._grid[k]], dtype=float)

    def _draw_from_seed(self, step_seed: int) -> np.ndarray:
        if self.seed is None:
            mix_seed = int(step_seed)
        else:
            mix_seed = int(np.random.SeedSequence([int(step_seed), self.seed]).generate_state(1)[0])
        rng = np.random.default_rng(mix_seed)
        return self._sample_action_from_rng(rng)

    def act(self, s, t: int, info: Optional[dict] = None) -> np.ndarray:
        info = {} if info is None else info
        deterministic = bool(info.get("deterministic", False))
        step_seed = info.get("crn_step_seed", None)

        if deterministic:
            if step_seed is None:
                raise ValueError("OrderRandomPolicy deterministic mode requires info['crn_step_seed'].")
            return self._draw_from_seed(int(step_seed))

        if step_seed is not None:
            return self._draw_from_seed(int(step_seed))

        return self._sample_action_from_rng(self._rng)
    
    __all__ = ["OrderRandomPolicy"]