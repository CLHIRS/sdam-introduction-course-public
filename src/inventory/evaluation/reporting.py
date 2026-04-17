from __future__ import annotations

from typing import Dict

import numpy as np


def summarize_totals(totals_by_policy: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Simple summary dict (mean/std/n)."""

    out: Dict[str, Dict[str, float]] = {}
    for name, totals in totals_by_policy.items():
        arr = np.asarray(totals, dtype=float)

        mean = float(np.mean(arr)) if len(arr) else float("nan")
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        out[name] = {
            "mean": round(mean, 3),
            "std": round(std, 3),
            "n": float(len(arr)),
        }
    return out


__all__ = ["summarize_totals"]
