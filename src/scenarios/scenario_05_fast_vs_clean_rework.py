# scenarios/scenario_05_fast_vs_clean_rework.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from queueing.eval_crn import evaluate_policies_crn_mc_queue, print_strict_crn_report_queue
from queueing.rl_env import RoutingObsSpec
from queueing.routing_policies import JoinShortestQueue, RandomRoute, RouteDecisionPolicy

from scenarios._fast_mode import fast_T, fast_n_rep


@dataclass(frozen=True)
class AlwaysChoose(RouteDecisionPolicy):
    """Always take the same action index within a decision point."""

    a: int

    def select(self, rng, *, dp_id, job, from_node, actions, state) -> int:
        return int(self.a)


def build_config(*, lam=0.90) -> Dict[str, Any]:
    return {
        "job_classes": {
            "A": {"value": 500.0},
            "B": {"value": 350.0},
        },
        "arrival_process": {"type": "poisson", "rate": float(lam), "class_probs": {"A": 0.6, "B": 0.4}, "node": 0},
        "nodes": [
            {"id": 0, "name": "Dispatcher", "kind": "pool"},
            {"id": 1, "name": "Machining", "kind": "server", "servers": 1, "buffer_capacity": 60},
            {"id": 2, "name": "Paint", "kind": "server", "servers": 1, "buffer_capacity": 80},
            {"id": 3, "name": "AssemblyFast", "kind": "server", "servers": 1, "buffer_capacity": 60},
            {"id": 4, "name": "AssemblyClean", "kind": "server", "servers": 1, "buffer_capacity": 60},
            {"id": 5, "name": "Rework", "kind": "server", "servers": 1, "buffer_capacity": 120},
            {"id": 6, "name": "FinishedGoods", "kind": "sink"},
            {"id": 7, "name": "Scrap", "kind": "sink"},
        ],
        "edges": [
            {"from": 0, "to": 1},
            {"from": 1, "to": 2},
            {"from": 2, "to": 3},
            {"from": 2, "to": 4},
            {"from": 3, "to": 6},
            {"from": 4, "to": 6},
            {"from": 5, "to": 6},
            {"from": 5, "to": 7},
        ],
        "service": {
            "1": {"A": {"dist": "exp", "rate": 1.05}, "B": {"dist": "exp", "rate": 1.05}},
            "2": {"A": {"dist": "exp", "rate": 2.50}, "B": {"dist": "exp", "rate": 2.50}},
            "3": {"A": {"dist": "exp", "rate": 1.20}, "B": {"dist": "exp", "rate": 1.20}},
            "4": {"A": {"dist": "exp", "rate": 0.90}, "B": {"dist": "exp", "rate": 0.90}},
            "5": {"A": {"dist": "exp", "rate": 0.65}, "B": {"dist": "exp", "rate": 0.65}},
        },
        "decision_points": [
            {
                "id": "assembly_routing",
                "when": {"event": "service_complete", "node": 2},
                "action_type": "route",
                "feasible_actions": [{"to": 3, "allowed_classes": ["A", "B"]}, {"to": 4, "allowed_classes": ["A", "B"]}],
            },
            {
                "id": "defect_decision",
                "when": {"event": "quality_realized", "nodes": [3, 4]},
                "action_type": "route",
                "feasible_actions": [{"to": 5}, {"to": 7}],
            },
        ],
        "quality_model": {
            "assembly_nodes": [3, 4],
            "rework_node": 5,
            "fg_node": 6,
            "scrap_node": 7,
            "defect_prob": {"3": {"A": 0.18, "B": 0.10}, "4": {"A": 0.06, "B": 0.03}},
            "rework_success_prob": 0.95,
        },
        "objective": {
            "holding_cost_per_job_per_time": 1.0,
            "tardiness_penalty_per_time": 10.0,
            "scrap_penalty": {"A": 500.0, "B": 350.0},
            "rework_penalty": 30.0,
            "throughput_reward": {"A": 500.0, "B": 350.0},
        },
    }


def run_demo():
    cfg = build_config()
    obs_spec = RoutingObsSpec(node_ids=list(range(len(cfg["nodes"]))), class_labels=["A", "B"])

    # CRN note:
    # - If a policy is randomized, it should draw from the simulator/env-provided RNG.
    #   So we prefer RouteDecisionPolicy implementations (RandomRoute, JoinShortestQueue, ...)
    #   over agents with their own RNG.
    policies = {
        "Random": {
            "assembly_routing": RandomRoute(),
            "defect_decision": RandomRoute(),
        },
        "JSQ": {
            "assembly_routing": JoinShortestQueue(),
            "defect_decision": AlwaysChoose(0),  # prefer rework over scrap
        },
    }

    report = evaluate_policies_crn_mc_queue(
        cfg,
        policies=policies,
        baseline_name="Random",
        T=fast_T(2000.0),
        n_rep=fast_n_rep(30),
        seed0=0,
        overflow_mode="block",
        include_env_reward=True,
        env_obs_spec=obs_spec,
        verbose=True,
    )
    print_strict_crn_report_queue(report)


if __name__ == "__main__":
    run_demo()