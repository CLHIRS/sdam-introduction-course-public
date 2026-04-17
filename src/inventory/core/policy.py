from __future__ import annotations

from typing import Optional

from inventory.core.types import Action, State


class Policy:
    """Official policy interface.

    Required:
      act(S: np.ndarray, t: int, info: dict|None) -> np.ndarray (action vector)

    Notebook-friendly:
      policy(S, t) calls act(S, t).
    """

    def act(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        raise NotImplementedError

    def __call__(self, state: State, t: int, info: Optional[dict] = None) -> Action:
        return self.act(state, t, info=info)


__all__ = ["Policy"]
