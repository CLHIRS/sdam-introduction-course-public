# scenarios/scenario_01_mm1.py
from __future__ import annotations

import os
from typing import Any, Dict

from queueing.eval_crn import evaluate_policies_crn_mc_queue, print_strict_crn_report_queue


def build_config(*, lam=0.85, mu=1.0) -> Dict[str, Any]:
    return {
        "job_classes": {"A": {"value": 1.0}},
        "arrival_process": {"type": "poisson", "rate": float(lam), "class_probs": {"A": 1.0}, "node": 0},
        "nodes": [
            {"id": 0, "name": "Server", "kind": "server", "servers": 1, "buffer_capacity": 200},
            {"id": 1, "name": "Sink", "kind": "sink"},
        ],
        "edges": [{"from": 0, "to": 1}],
        "service": {"0": {"A": {"dist": "exp", "rate": float(mu)}}},
        "decision_points": [],  # none
        "objective": {"holding_cost_per_job_per_time": 1.0},
    }


def _is_fast() -> bool:
    return os.environ.get("SDAM_FAST", "").strip().lower() in {"1", "true", "yes"}


def run_demo(*, T: float | None = None, n_rep: int | None = None, seed0: int = 0) -> None:
    cfg = build_config()
    # No decision points => routing policy is never queried.
    # Policies are still provided for the evaluation harness, but they're empty.
    policies = {
        "Baseline": {},
        "Baseline2": {},
    }

    fast = _is_fast()
    T = float(T if T is not None else (200.0 if fast else 2000.0))
    n_rep = int(n_rep if n_rep is not None else (3 if fast else 20))

    report = evaluate_policies_crn_mc_queue(
        cfg,
        policies=policies,
        baseline_name="Baseline",
        T=T,
        n_rep=n_rep,
        seed0=int(seed0),
        overflow_mode="block",
        include_env_reward=False,
        verbose=(not fast),
    )
    print_strict_crn_report_queue(report)


if __name__ == "__main__":
    run_demo()