# scenarios/scenario_06_final_ppo_ready_manufacturing.py
from __future__ import annotations

import numpy as np
from queueing.config_compile import compile_queue_network_config
from queueing.dispatch import FIFO, DisciplineDispatchPolicy, EDDDispatchPolicy
from queueing.rl_env import RoutingObsSpec, RoutingOnlyEnv
from queueing.routing_policies import JoinShortestQueue, RandomRoute, SlackAwareAssembly
from queueing.sim import QueueingNetworkSim


def build_cfg() -> dict:
    # This mirrors your YAML-like draft, as a Python dict.
    return {
        "job_classes": {
            "A": {"value": 500, "due_date_dist": {"type": "shifted_exp", "mean": 20, "shift": 5}},
            "B": {"value": 350, "due_date_dist": {"type": "shifted_exp", "mean": 22, "shift": 5}},
        },
        "arrival_process": {
            "type": "poisson",
            "rate": 0.95,
            "class_probs": {"A": 0.6, "B": 0.4},
        },
        "nodes": [
            {"id": 0, "name": "Dispatcher", "kind": "pool"},
            {"id": 1, "name": "M1A", "kind": "server", "servers": 1, "buffer_capacity": 20},
            {"id": 2, "name": "M1B", "kind": "server", "servers": 1, "buffer_capacity": 30},
            {"id": 3, "name": "B1", "kind": "buffer", "capacity": 40},
            {"id": 4, "name": "Paint", "kind": "server", "servers": 1, "buffer_capacity": 40},
            {"id": 5, "name": "AssemblyFast", "kind": "server", "servers": 1, "buffer_capacity": 30},
            {"id": 6, "name": "AssemblyClean", "kind": "server", "servers": 1, "buffer_capacity": 30},
            {"id": 7, "name": "Rework", "kind": "server", "servers": 1, "buffer_capacity": 50},
            {"id": 8, "name": "FinishedGoods", "kind": "sink"},
            {"id": 9, "name": "Scrap", "kind": "sink"},
        ],
        "edges": [
            {"from": 0, "to": 1},
            {"from": 0, "to": 2},
            {"from": 1, "to": 3},
            {"from": 2, "to": 3},
            {"from": 3, "to": 4},
            {"from": 4, "to": 5},
            {"from": 4, "to": 6},
            {"from": 5, "to": 8},
            {"from": 5, "to": 7},
            {"from": 6, "to": 8},
            {"from": 6, "to": 7},
            {"from": 7, "to": 8},
            {"from": 7, "to": 9},
        ],
        "service": {
            # node_id -> {class -> {dist, rate}}
            1: {"A": {"dist": "exp", "rate": 1.05}},  # M1A only A
            2: {"A": {"dist": "exp", "rate": 0.90}, "B": {"dist": "exp", "rate": 0.90}},
            4: {"A": {"dist": "exp", "rate": 2.50}, "B": {"dist": "exp", "rate": 2.50}},
            5: {"A": {"dist": "exp", "rate": 1.20}, "B": {"dist": "exp", "rate": 1.20}},
            6: {"A": {"dist": "exp", "rate": 0.90}, "B": {"dist": "exp", "rate": 0.90}},
            7: {"A": {"dist": "exp", "rate": 0.65}, "B": {"dist": "exp", "rate": 0.65}},
        },
        "decision_points": [
            {
                "id": "release_routing",
                "when": {"event": "arrival_or_release", "node": 0},
                "action_type": "route",
                "feasible_actions": [
                    {"to": 1, "allowed_classes": ["A"]},
                    {"to": 2, "allowed_classes": ["A", "B"]},
                ],
            },
            {
                "id": "assembly_routing",
                "when": {"event": "service_complete", "node": 4},
                "action_type": "route",
                "feasible_actions": [
                    {"to": 5, "allowed_classes": ["A", "B"]},
                    {"to": 6, "allowed_classes": ["A", "B"]},
                ],
            },
            {
                "id": "defect_decision",
                "when": {"event": "quality_realized", "node": 5},  # compiler uses node id; env triggers via quality override
                "action_type": "route",
                "feasible_actions": [
                    {"to": 7},  # rework
                    {"to": 9},  # scrap
                ],
            },
        ],
        "quality_model": {
            "defect_prob": {
                # assembly node ids -> {class -> p_def}
                5: {"A": 0.18, "B": 0.10},
                6: {"A": 0.06, "B": 0.03},
            },
            "rework_success_prob": 0.95,
        },
        "objective": {
            "holding_cost_per_job_per_time": 1.0,
            "tardiness_penalty_per_time": 10.0,
            "scrap_penalty": {"A": 500, "B": 350},
            "rework_penalty": 50.0,
            "throughput_reward": {"A": 500, "B": 350},
        },
    }


def run_des_kpis(cfg: dict) -> None:
    routing_dp_policies = {
        "release_routing": JoinShortestQueue(),
        "assembly_routing": SlackAwareAssembly(fast_to=5, clean_to=6, slack_threshold=3.0),
        "defect_decision": RandomRoute(),
    }

    dispatch_by_node_name = {
        # keep simple; show EDD only on assembly nodes as a “dispatch” add-on
        "M1A": DisciplineDispatchPolicy(FIFO()),
        "M1B": DisciplineDispatchPolicy(FIFO()),
        "Paint": DisciplineDispatchPolicy(FIFO()),
        "AssemblyFast": EDDDispatchPolicy(),
        "AssemblyClean": EDDDispatchPolicy(),
        "Rework": DisciplineDispatchPolicy(FIFO()),
    }

    conf, quality_cfg, aux = compile_queue_network_config(
        cfg,
        routing_dp_policies=routing_dp_policies,
        dispatch_by_node_name=dispatch_by_node_name,
        overflow_mode="block",
    )

    sim = QueueingNetworkSim(conf, quality_cfg=quality_cfg)
    kpis = sim.run(T=2000.0, seed=0)

    print("\n=== Manufacturing Queueing Network KPIs (DES) ===")
    print(f"sim_time           = {kpis.sim_time:.1f}")
    print(f"n_completed        = {kpis.n_completed}")
    print(f"throughput         = {kpis.throughput:.4f}")
    print(f"mean_cycle_time    = {kpis.mean_cycle_time:.4f}")
    print(f"mean_waiting_time  = {kpis.mean_waiting_time:.4f}")
    print(f"mean_tardiness     = {kpis.mean_tardiness:.4f}")
    print(f"on_time_delivery   = {kpis.on_time_delivery:.4f}")
    print(f"WIP time-average   = {kpis.wip_time_avg:.4f}")
    print(f"fill_rate          = {kpis.fill_rate:.4f}")
    print(f"lost_external      = {kpis.lost_external}")
    print(f"dropped_internal   = {kpis.dropped_internal}")
    print(f"blocked_internal   = {kpis.blocked_internal}")
    print(f"unblocked_internal = {kpis.unblocked_internal}")

    print("\nutilization_processing per node:")
    print(np.round(kpis.utilization_processing, 3))
    print("blocked_time_avg per node:")
    print(np.round(kpis.blocked_time_avg, 3))
    print("starved_time_avg per node:")
    print(np.round(kpis.starved_time_avg, 3))


def run_env_one_episode(cfg: dict, seed: int = 0) -> None:
    routing_dp_policies = {
        "release_routing": JoinShortestQueue(),
        "assembly_routing": SlackAwareAssembly(fast_to=5, clean_to=6, slack_threshold=3.0),
        "defect_decision": RandomRoute(),
    }

    conf, quality_cfg, aux = compile_queue_network_config(
        cfg,
        routing_dp_policies=routing_dp_policies,
        dispatch_by_node_name=None,
        overflow_mode="block",
    )

    obs_spec = RoutingObsSpec(node_ids=list(range(conf.n_nodes)), class_labels=["A", "B"])

    env = RoutingOnlyEnv(
        conf,
        cfg=cfg,
        quality_cfg=quality_cfg,
        obs_spec=obs_spec,
        seed=seed,
        T=500.0,
    )

    ctx, r, done, info = env.reset_and_get_first()
    total = r
    steps = 0

    while not done:
        # baseline: act using the same routing_policy as the DES (policy-in-sim style)
        # we pick action by calling the configured dp policy (heuristic), not PPO yet.
        dp = conf.routing_policy.dp_policy[ctx.dp_id]
        state = env._snapshot()
        a = int(dp.select(env.rng, dp_id=ctx.dp_id, job=env.jobs[ctx.job_id], from_node=ctx.from_node, actions=ctx.actions, state=state))

        env.apply_action(ctx, a)
        ctx, r, done, info = env.step_until_decision()
        total += r
        steps += 1

    print("\n=== RoutingOnlyEnv (one episode) ===")
    print(f"seed={seed} T={env.T:.1f} decision_steps={steps}")
    print(f"total_reward={total:.3f}  (higher is better)")
    print(f"completed_jobs={env.n_completed} lost_external={env.lost_external} dropped_internal={env.dropped_internal}")


if __name__ == "__main__":
    cfg = build_cfg()
    run_des_kpis(cfg)
    run_env_one_episode(cfg, seed=0)