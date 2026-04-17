# scenarios/scenario_02_parallel.py
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from queueing.eval_crn import evaluate_policies_crn_mc_queue, print_strict_crn_report_queue
from queueing.rl_env import RoutingObsSpec
from queueing.routing_policies import JoinShortestQueue, RandomRoute, RoutingDecisionAgent

from scenarios._fast_mode import fast_T, fast_n_rep


class RandomMaskedAgent(RoutingDecisionAgent):
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, dp_id: str, obs: np.ndarray, mask: np.ndarray, *, deterministic: bool = True) -> int:
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            return 0
        if deterministic:
            return int(valid[0])
        return int(self.rng.choice(valid))


class JSQAgent(RoutingDecisionAgent):
    def __init__(self, obs_spec: RoutingObsSpec, dp_action_to_nodes: Dict[str, List[int]]):
        self.obs_spec = obs_spec
        self.dp_action_to_nodes = dp_action_to_nodes

    def _q_from_obs(self, obs: np.ndarray) -> np.ndarray:
        """Parse queue-length block from RoutingObsSpec.encode().

        Encoding is: [class_onehot | from_onehot | slack | q | in_service | blocked]
        """
        n_class = len(self.obs_spec.class_labels)
        n_node = len(self.obs_spec.node_ids)
        q_start = n_class + n_node + 1
        q_end = q_start + n_node
        return obs[q_start:q_end]

    def act(self, dp_id: str, obs: np.ndarray, mask: np.ndarray, *, deterministic: bool = True) -> int:
        q = self._q_from_obs(obs)
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            return 0
        nodes = self.dp_action_to_nodes[dp_id]
        best_a = int(valid[0])
        best_q = float("inf")
        for a in valid:
            to = nodes[int(a)]
            v = float(q[to])
            if v < best_q:
                best_q = v
                best_a = int(a)
        return best_a


def build_config(*, lam=0.90, mu1=1.0, mu2=1.0) -> Dict[str, Any]:
    return {
        "job_classes": {"A": {"value": 1.0}},
        "arrival_process": {"type": "poisson", "rate": float(lam), "class_probs": {"A": 1.0}, "node": 0},
        "nodes": [
            {"id": 0, "name": "Dispatcher", "kind": "pool"},
            {"id": 1, "name": "M1", "kind": "server", "servers": 1, "buffer_capacity": 100},
            {"id": 2, "name": "M2", "kind": "server", "servers": 1, "buffer_capacity": 100},
            {"id": 3, "name": "Sink", "kind": "sink"},
        ],
        "edges": [
            {"from": 0, "to": 1}, {"from": 0, "to": 2},
            {"from": 1, "to": 3}, {"from": 2, "to": 3},
        ],
        "service": {
            "1": {"A": {"dist": "exp", "rate": float(mu1)}},
            "2": {"A": {"dist": "exp", "rate": float(mu2)}},
        },
        "decision_points": [
            {
                "id": "route_parallel",
                "when": {"event": "arrival_or_release", "node": 0},
                "action_type": "route",
                "feasible_actions": [{"to": 1, "allowed_classes": ["A"]}, {"to": 2, "allowed_classes": ["A"]}],
            }
        ],
        "objective": {"holding_cost_per_job_per_time": 1.0},
    }


def run_demo():
    cfg = build_config()
    obs_spec = RoutingObsSpec(node_ids=list(range(len(cfg["nodes"]))), class_labels=["A"])

    # CRN note:
    # - Exogenous randomness is paired by seed.
    # - If a policy is randomized, it should draw from the simulator/env-provided RNG.
    #   So for strict-CRN comparisons we prefer RouteDecisionPolicy implementations
    #   (RandomRoute, JoinShortestQueue, ...) over agents with their own RNG.
    policies = {
        "Random": {"route_parallel": RandomRoute()},
        "JSQ": {"route_parallel": JoinShortestQueue()},
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