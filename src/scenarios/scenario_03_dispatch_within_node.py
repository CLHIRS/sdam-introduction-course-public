# scenarios/scenario_03_dispatching_within_node_FIFO_EDD.py
from __future__ import annotations

from queueing.config_compile import compile_queue_network_config
from queueing.dispatch import FIFO, DisciplineDispatchPolicy, EDDDispatchPolicy
from queueing.eval_crn import evaluate_policies_crn_mc_queue, print_strict_crn_report_queue
from queueing.sim import QueueingNetworkSim

from scenarios._fast_mode import fast_T, fast_n_rep

# -----------------------------
# Scenario 03: build config
# -----------------------------

def build_cfg() -> dict:
    """
    Scenario 03 — Dispatching within a node (EDD vs FIFO)

    Network: Source (jobs w/ due dates) → One server → Sink
    Decision point: dispatch rule at server
    Policies: FIFO vs EDD
    KPIs: mean_tardiness, on_time_delivery, mean_cycle_time
    """
    return {
        "job_classes": {
            "A": {"value": 1.0, "due_date_dist": {"type": "shifted_exp", "mean": 10.0, "shift": 4.0}},
        },
        "arrival_process": {"type": "poisson", "rate": 0.85, "class_probs": {"A": 1.0}},
        "nodes": [
            {"id": 0, "name": "Source", "kind": "pool"},
            {"id": 1, "name": "Machine", "kind": "server", "servers": 1, "buffer_capacity": 200},
            {"id": 2, "name": "Sink", "kind": "sink"},
        ],
        "edges": [{"from": 0, "to": 1}, {"from": 1, "to": 2}],
        "service": {1: {"A": {"dist": "exp", "rate": 1.00}}},
        "decision_points": [],  # no routing decisions
        "quality_model": {},
        # objective influences env_total_reward (optional but useful)
        "objective": {
            "holding_cost_per_job_per_time": 0.2,
            "tardiness_penalty_per_time": 5.0,
            "scrap_penalty": {"A": 0.0},
            "rework_penalty": 0.0,
            "throughput_reward": {"A": 0.0},
        },
    }


# -----------------------------
# Scenario 03: baselines
# -----------------------------

# --- old style: {"name": {"dp_id": RouteDecisionPolicy, ...}} (routing-only) ---
#def make_baselines() -> Dict[str, Any]:
#    """
#    Returns a standard structure used by all scenarios.
#    """
#    policies = {
#        "FIFO": {},  # routing dp policies (empty here)
#        "EDD": {},
#    }
#
#    dispatch: Dict[str, Dict[str, DispatchPolicy]] = {
#        "FIFO": {"Machine": DisciplineDispatchPolicy(FIFO())},
#        "EDD": {"Machine": EDDDispatchPolicy()},
#    }
#
#    return {
#        "policies": policies,
#        "dispatch": dispatch,
#        "baseline_name": "FIFO",
#        "T_des": 2000.0,
#        "T_env": 500.0,
#        "n_rep": 50,
#    }

#--- new style: {"name": PolicySpec(...)} --
from queueing.eval_crn import PolicySpec


def make_baselines():
    policies = {
        "FIFO": PolicySpec(
            routing_dp_policies={},
            dispatch_by_node_name={"Machine": DisciplineDispatchPolicy(FIFO())},
        ),
        "EDD": PolicySpec(
            routing_dp_policies={},
            dispatch_by_node_name={"Machine": EDDDispatchPolicy()},
        ),
    }
    return {
        "policies": policies,
        "baseline_name": "FIFO",
        "T_des": 2000.0,
        "T_env": 500.0,
        "n_rep": 50,
    }


# -----------------------------
# Scenario 03: demos
# -----------------------------

def run_des_demo(seed: int = 0) -> None:
    cfg = build_cfg()
    base = make_baselines()

    print("\n=== Scenario 03: DES demo (single run) ===")
    for name in ["FIFO", "EDD"]:
        pol = base["policies"][name]
        conf, quality_cfg, _ = compile_queue_network_config(
            cfg,
            routing_dp_policies=pol.routing_dp_policies,
            dispatch_by_node_name=pol.dispatch_by_node_name,
            overflow_mode="block",
        )
        kpis = QueueingNetworkSim(conf, quality_cfg=quality_cfg).run(T=fast_T(base["T_des"]), seed=seed)
        print(f"\nPolicy: {name}")
        print(f"  throughput        = {kpis.throughput:.4f}")
        print(f"  mean_cycle_time   = {kpis.mean_cycle_time:.4f}")
        print(f"  mean_tardiness    = {kpis.mean_tardiness:.4f}")
        print(f"  on_time_delivery  = {kpis.on_time_delivery:.4f}")

# --- old style ---
#def run_crn_demo(seed0: int = 0) -> None:
#    """
#    Strict-CRN MC compare using the DES+Env evaluator.
#    Note: evaluator varies routing policies; here we also need to vary dispatch.
#    We do this by running the evaluator once per policy with its dispatch map,
#    and printing summary blocks. (Keeps evaluator stable and scenario code simple.)
#    """
#    cfg = build_cfg()
#    base = make_baselines()
#
#    print("\n=== Scenario 03: Strict-CRN demo (compare dispatch rules) ===")
#
#    reports = {}
#    for name in ["FIFO", "EDD"]:
#        rep = evaluate_policies_crn_mc_queue(
#            cfg,
#            policies={name: base["policies"][name]},
#            baseline_name=name,
#            T=base["T_des"],
#            env_T=base["T_env"],
#            n_rep=base["n_rep"],
#            seed0=seed0,
#            overflow_mode="block",
#            include_env_reward=True,
#            dispatch_by_node_name=base["dispatch"][name],
#            verbose=False,
#        )
#        reports[name] = rep
#
#    metrics = ["env_total_reward", "mean_tardiness", "on_time_delivery", "mean_cycle_time", "throughput"]
#    print("\n--- FIFO summary ---")
#    print_strict_crn_report_queue(reports["FIFO"], metrics=metrics)
#    print("\n--- EDD summary ---")
#    print_strict_crn_report_queue(reports["EDD"], metrics=metrics)
#
#    print("\nStudent-facing takeaway:")
#    print("  - EDD typically improves mean_tardiness / on_time_delivery by prioritizing urgent jobs.")
#    print("  - The impact on mean_cycle_time depends on due-date tightness and variability.")


#--- new style ---
def run_crn_demo(seed0: int = 0) -> None:
    cfg = build_cfg()
    base = make_baselines()
    print("\n=== Scenario 03: Strict-CRN demo (compare dispatch rules) ===")
    report = evaluate_policies_crn_mc_queue(
        cfg,
        policies=base["policies"],
        baseline_name=base["baseline_name"],
        T=fast_T(base["T_des"]),
        env_T=fast_T(base["T_env"]),
        n_rep=fast_n_rep(base["n_rep"]),
        seed0=seed0,
        overflow_mode="block",
        include_env_reward=True,
        verbose=False,
    )

    print_strict_crn_report_queue(report, metrics=[
        "env_total_reward",
        "mean_tardiness",
        "on_time_delivery",
        "mean_cycle_time",
        "throughput",
    ])


if __name__ == "__main__":
    run_des_demo(seed=0)
    run_crn_demo(seed0=0)