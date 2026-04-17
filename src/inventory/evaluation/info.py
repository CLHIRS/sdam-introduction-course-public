from __future__ import annotations

from typing import Any, Dict, Optional


def make_eval_info(
    *,
    det_mode: str = "argmax",
    risk_alpha: float = 0.0,
    tau: Optional[float] = None,
    T: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Standard evaluation `info` dict passed to `Policy.act`.

    Conventions:
    - `deterministic=True` by default for evaluation.
    - `det_mode` selects a deterministic action rule (policy-specific interpretation).
      Examples:
        - PPO: "mean" | "argmax"
        - AlphaZero: "argmax" (and optionally "sample" when deterministic=False)
    - `risk_alpha` is used by PPO's risk-aware argmax variant.
    - `tau` can be used by MCTS-style policies (e.g., AlphaZero) when applicable.
    - `T` can be passed for policies that need the horizon to featurize time.

    Notes:
    - `crn_step_seed` is injected by `DynamicSystemMVP` per step; do not set it here.
        - `last_demand` is injected by `DynamicSystemMVP` from the previous realized demand;
            it is absent at the first decision step.
        - `demand_history` may also be injected when the system is configured with
            `demand_history_window > 0`; it contains a bounded recent history of realized
            demands and is absent until at least one demand has been observed.
    """

    out: Dict[str, Any] = {"deterministic": True, "det_mode": str(det_mode)}

    if risk_alpha:
        out["risk_alpha"] = float(risk_alpha)
    if tau is not None:
        out["tau"] = float(tau)
    if T is not None:
        out["T"] = int(T)

    if extra:
        out.update(dict(extra))

    return out


def make_train_info(
    *,
    det_mode: str = "sample",
    tau: Optional[float] = None,
    T: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Standard training `info` dict.

    Conventions:
    - `deterministic=False` by default for training.
    - Policies may use `crn_step_seed` (if present) for reproducible *decision sampling*.
        - Forecasting policies may use `last_demand` (if present) as the most recently
            observed realized demand.
        - Forecasting policies that explicitly opt into richer history may also use
            `demand_history` when present.
    """

    out: Dict[str, Any] = {"deterministic": False, "det_mode": str(det_mode)}

    if tau is not None:
        out["tau"] = float(tau)
    if T is not None:
        out["T"] = int(T)

    if extra:
        out.update(dict(extra))

    return out


__all__ = ["make_eval_info", "make_train_info"]
