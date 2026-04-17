from __future__ import annotations

import runpy
from pathlib import Path

from queueing.dispatch import FIFO, DisciplineDispatchPolicy, EDDDispatchPolicy
from queueing.eval_crn import (
    PolicySpec,
    evaluate_policies_crn_mc_queue,
    print_strict_crn_report_queue,
)
from queueing.routing_policies import JoinShortestQueue, RandomRoute, SlackAwareAssembly


def main() -> None:
    scenario_path = Path(__file__).with_name("scenario_06_final_ppo_ready_manufacturing.py")
    ns = runpy.run_path(str(scenario_path))
    cfg = ns["build_cfg"]()

    routing_baseline = {
        "release_routing": JoinShortestQueue(),
        "assembly_routing": SlackAwareAssembly(fast_to=5, clean_to=6, slack_threshold=3.0),
        "defect_decision": RandomRoute(),
    }

    dispatch_fifo = {
        "M1A": DisciplineDispatchPolicy(FIFO()),
        "M1B": DisciplineDispatchPolicy(FIFO()),
        "Paint": DisciplineDispatchPolicy(FIFO()),
        "AssemblyFast": DisciplineDispatchPolicy(FIFO()),
        "AssemblyClean": DisciplineDispatchPolicy(FIFO()),
        "Rework": DisciplineDispatchPolicy(FIFO()),
    }

    dispatch_edd_assembly = {
        **dispatch_fifo,
        "AssemblyFast": EDDDispatchPolicy(),
        "AssemblyClean": EDDDispatchPolicy(),
    }

    policies = {
        "baseline_fifo": PolicySpec(
            routing_dp_policies=routing_baseline,
            dispatch_by_node_name=dispatch_fifo,
        ),
        "edd_assembly": PolicySpec(
            routing_dp_policies=routing_baseline,
            dispatch_by_node_name=dispatch_edd_assembly,
        ),
    }

    report = evaluate_policies_crn_mc_queue(
        cfg,
        policies=policies,
        baseline_name="baseline_fifo",
        T=1000.0,
        n_rep=20,
        seed0=0,
        overflow_mode="block",
        include_env_reward=True,
        verbose=True,
    )

    print_strict_crn_report_queue(
        report,
        metrics=["throughput", "mean_cycle_time", "mean_tardiness", "env_total_reward"],
    )


if __name__ == "__main__":
    main()
