# scenarios/scenario_01_mm1.py
from __future__ import annotations

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


def run_demo():
    cfg = build_config()
    # No decision points => routing policy is never queried.
    # Policies are still provided for the evaluation harness, but they're empty.
    policies = {
        "Baseline": {},
        "Baseline2": {},
    }

    report = evaluate_policies_crn_mc_queue(
        cfg,
        policies=policies,
        baseline_name="Baseline",
        T=2000.0,
        n_rep=20,
        seed0=0,
        overflow_mode="block",
        include_env_reward=False,
        verbose=True,
    )
    print_strict_crn_report_queue(report)


if __name__ == "__main__":
    run_demo()