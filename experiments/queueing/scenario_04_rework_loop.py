# scenarios/scenario_04_rework_loop.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from queueing.config_compile import compile_queue_network_config
from queueing.eval_crn import evaluate_policies_crn_mc_queue, print_strict_crn_report_queue
from queueing.routing_policies import RouteDecisionPolicy
from queueing.sim import QueueingNetworkSim

# -----------------------------
# Simple defect-decision policies
# -----------------------------

@dataclass(frozen=True)
class AlwaysChoose(RouteDecisionPolicy):
    to_node: int  # destination node index

    def select(self, rng, *, dp_id, job, from_node, actions, state) -> int:
        for a, act in enumerate(actions):
            if act.to == self.to_node:
                return int(a)
        return 0


@dataclass(frozen=True)
class ReworkIfQueueShort(RouteDecisionPolicy):
    """
    Rework if the rework queue is not too long; else scrap.
    Assumes feasible actions include the rework node and a scrap node.
    """
    rework_node: int
    max_q: int = 5

    def select(self, rng, *, dp_id, job, from_node, actions, state) -> int:
        # find index of rework action
        idx_rework = None
        for a, act in enumerate(actions):
            if act.to == self.rework_node:
                idx_rework = int(a)
                break
        if idx_rework is None:
            idx_rework = 0

        qlen = int(state.queue_lens[int(self.rework_node)])
        if qlen <= int(self.max_q):
            return idx_rework

        # choose a non-rework action (scrap)
        for a, act in enumerate(actions):
            if int(a) != idx_rework:
                return int(a)
        return idx_rework


# -----------------------------
# Scenario 04: build config
# -----------------------------

def build_cfg() -> dict:
    """
    Scenario 04 — Rework loop (quality-driven dynamics)

    Network: Source → Process → Quality check → {Rework, Scrap, FG}
    Decision: when defective, rework vs scrap
    KPIs: WIP, throughput, cycle time (+ env_total_reward captures scrap/rework penalties)
    """
    return {
        "job_classes": {
            "A": {"value": 500, "due_date_dist": {"type": "shifted_exp", "mean": 20.0, "shift": 5.0}},
        },
        "arrival_process": {"type": "poisson", "rate": 0.90, "class_probs": {"A": 1.0}},
        "nodes": [
            {"id": 0, "name": "Source", "kind": "pool"},
            {"id": 1, "name": "Process", "kind": "server", "servers": 1, "buffer_capacity": 200},
            {"id": 2, "name": "Rework", "kind": "server", "servers": 1, "buffer_capacity": 200},
            {"id": 3, "name": "FinishedGoods", "kind": "sink"},
            {"id": 4, "name": "Scrap", "kind": "sink"},
        ],
        "edges": [
            {"from": 0, "to": 1},
            {"from": 1, "to": 2},
            {"from": 1, "to": 4},
            {"from": 2, "to": 3},
            {"from": 2, "to": 4},
        ],
        "service": {
            1: {"A": {"dist": "exp", "rate": 1.20}},
            2: {"A": {"dist": "exp", "rate": 0.60}},  # rework bottleneck
        },
        "decision_points": [
            {
                "id": "defect_decision",
                "when": {"event": "quality_realized", "node": 1},
                "action_type": "route",
                "feasible_actions": [
                    {"to": 2},  # rework
                    {"to": 4},  # scrap
                ],
            }
        ],
        "quality_model": {
            "defect_prob": {1: {"A": 0.20}},
            "rework_success_prob": 0.95,
        },
        "objective": {
            "holding_cost_per_job_per_time": 1.0,
            "tardiness_penalty_per_time": 5.0,
            "scrap_penalty": {"A": 500},
            "rework_penalty": 50.0,
            "throughput_reward": {"A": 500},
        },
    }


# -----------------------------
# Scenario 04: baselines
# -----------------------------

# --- old style ---
#def make_baselines() -> Dict[str, Any]:
#    """
#    Node indices are by node order in cfg:
#      0 Source, 1 Process, 2 Rework, 3 FG, 4 Scrap
#    """
#    REWORK = 2
#    SCRAP = 4
#
#    policies = {
#        "always_rework": {"defect_decision": AlwaysChoose(to_node=REWORK)},
#        "always_scrap": {"defect_decision": AlwaysChoose(to_node=SCRAP)},
#        "threshold_rework": {"defect_decision": ReworkIfQueueShort(rework_node=REWORK, max_q=5)},
#    }
#
#    return {
#        "policies": policies,
#        "dispatch": None,               # routing-only scenario
#        "baseline_name": "always_rework",
#        "T_des": 2000.0,
#        "T_env": 800.0,
#        "n_rep": 50,
#   }


# --- new style ---
from queueing.eval_crn import PolicySpec


def make_baselines() -> Dict[str, Any]:
    """
    Node indices are by node order in cfg:
      0 Source, 1 Process, 2 Rework, 3 FG, 4 Scrap
    """
    REWORK = 2
    SCRAP = 4

    policies = {
        "always_rework": PolicySpec(
            routing_dp_policies={"defect_decision": AlwaysChoose(to_node=REWORK)},
            dispatch_by_node_name=None,
        ),
        "always_scrap": PolicySpec(
            routing_dp_policies={"defect_decision": AlwaysChoose(to_node=SCRAP)},
            dispatch_by_node_name=None,
        ),
        "threshold_rework": PolicySpec(
            routing_dp_policies={"defect_decision": ReworkIfQueueShort(rework_node=REWORK, max_q=5)},
            dispatch_by_node_name=None,
        ),
    }

    return {
        "policies": policies,
        "baseline_name": "always_rework",
        "T_des": 2000.0,
        "T_env": 800.0,
        "n_rep": 50,
    }

# -----------------------------
# Scenario 04: demos
# -----------------------------

def run_des_demo(seed: int = 0) -> None:
    cfg = build_cfg()
    base = make_baselines()

    print("\n=== Scenario 04: DES demo (single run) ===")
    for name, pol in base["policies"].items():
        conf, quality_cfg, _ = compile_queue_network_config(
            cfg,
            routing_dp_policies=pol.routing_dp_policies,
            dispatch_by_node_name=pol.dispatch_by_node_name,
            overflow_mode="block",
        )
        kpis = QueueingNetworkSim(conf, quality_cfg=quality_cfg).run(T=base["T_des"], seed=seed)
        print(f"\nPolicy: {name}")
        print(f"  throughput        = {kpis.throughput:.4f}")
        print(f"  mean_cycle_time   = {kpis.mean_cycle_time:.4f}")
        print(f"  mean_waiting_time = {kpis.mean_waiting_time:.4f}")
        print(f"  wip_time_avg      = {kpis.wip_time_avg:.4f}")

# --- old style ---
#def run_crn_demo(seed0: int = 0) -> None:
#    cfg = build_cfg()
#    base = make_baselines()
#
#    print("\n=== Scenario 04: Strict-CRN demo (compare defect decisions) ===")#
#
#    report = evaluate_policies_crn_mc_queue(
#        cfg,
#        policies=base["policies"],
#        baseline_name=base["baseline_name"],
#        T=base["T_des"],
#        env_T=base["T_env"],
#        n_rep=base["n_rep"],
#        seed0=seed0,
#        overflow_mode="block",
#        include_env_reward=True,
#        verbose=False,
#    )
#
#    print_strict_crn_report_queue(
#        report,
#        metrics=["env_total_reward", "throughput", "wip_time_avg", "mean_cycle_time", "mean_waiting_time"],
#    )
#
#    print("\nStudent-facing takeaway:")
#    print("  - 'always_rework' reduces scrap but can overload rework → WIP and cycle time rise.")
#    print("  - 'always_scrap' protects flow but throws away value.")
#    print("  - threshold policy is the first ‘congestion-aware’ control and foreshadows PPO.")
    

# --- new style ---
def run_crn_demo(seed0: int = 0) -> None:
    cfg = build_cfg()
    base = make_baselines()

    print("\n=== Scenario 04: Strict-CRN demo (compare defect decisions) ===")

    report = evaluate_policies_crn_mc_queue(
        cfg,
        policies=base["policies"],
        baseline_name=base["baseline_name"],
        T=base["T_des"],
        env_T=base["T_env"],
        n_rep=base["n_rep"],
        seed0=seed0,
        overflow_mode="block",
        include_env_reward=True,
        verbose=False,
    )

    print_strict_crn_report_queue(
        report,
        metrics=["env_total_reward", "throughput", "wip_time_avg", "mean_cycle_time", "mean_waiting_time"],
    )

    print("\nStudent-facing takeaway:")
    print("  - 'always_rework' reduces scrap but can overload rework → WIP and cycle time rise.")
    print("  - 'always_scrap' protects flow but throws away value.")
    print("  - threshold policy is the first ‘congestion-aware’ control and foreshadows PPO.")

if __name__ == "__main__":
    run_des_demo(seed=0)
    run_crn_demo(seed0=0)