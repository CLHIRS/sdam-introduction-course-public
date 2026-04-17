# queueing/routing_policies.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from queueing.core_types import Job, QueueStateSnapshot


@dataclass(frozen=True)
class RouteAction:
    to: Optional[int]                 # destination node index; None = exit
    allowed_classes: Optional[List[Any]] = None


def build_action_mask(*, dp_actions: List[RouteAction], job: Job) -> np.ndarray:
    mask = np.ones(len(dp_actions), dtype=bool)
    for a, act in enumerate(dp_actions):
        if act.allowed_classes is not None:
            mask[a] = job.job_class in act.allowed_classes
    return mask


class RouteDecisionPolicy:
    def select(
        self,
        rng: np.random.Generator,
        *,
        dp_id: str,
        job: Job,
        from_node: int,
        actions: List[RouteAction],
        state: QueueStateSnapshot,
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class RandomRoute(RouteDecisionPolicy):
    def select(self, rng, *, dp_id, job, from_node, actions, state) -> int:
        mask = build_action_mask(dp_actions=actions, job=job)
        valid = np.flatnonzero(mask)
        return int(rng.choice(valid)) if len(valid) else 0


@dataclass(frozen=True)
class JoinShortestQueue(RouteDecisionPolicy):
    """
    Choose destination with smallest current queue length among valid actions.
    """
    def select(self, rng, *, dp_id, job, from_node, actions, state) -> int:
        mask = build_action_mask(dp_actions=actions, job=job)
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            return 0
        best_a = int(valid[0])
        best_q = float("inf")
        for a in valid:
            to = actions[int(a)].to
            if to is None:
                continue
            q = float(state.queue_lens[int(to)])
            if q < best_q:
                best_q = q
                best_a = int(a)
        return best_a


@dataclass(frozen=True)
class SlackAwareAssembly(RouteDecisionPolicy):
    """
    If slack is tight, choose fast_to; else choose clean_to.
    Only meaningful for assembly routing decision points.
    """
    fast_to: int
    clean_to: int
    slack_threshold: float = 3.0  # in time units (not normalized)

    def select(self, rng, *, dp_id, job, from_node, actions, state) -> int:
        mask = build_action_mask(dp_actions=actions, job=job)
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            return 0
        if job.due_time is None:
            # fallback JSQ
            return JoinShortestQueue().select(rng, dp_id=dp_id, job=job, from_node=from_node, actions=actions, state=state)
        slack = float(job.due_time - state.t)
        want = self.fast_to if slack < self.slack_threshold else self.clean_to
        for a in valid:
            if actions[int(a)].to == want:
                return int(a)
        return int(valid[0])


@dataclass(frozen=True)
class DecisionPointRoutingPolicy:
    """
    Routing policy that switches by decision point id.

    dp_at_node: node_index -> dp_id for route decisions triggered at service completion at that node.
    dp_actions: dp_id -> list(RouteAction)
    dp_policy:  dp_id -> RouteDecisionPolicy
    default_next: if no dp at node, deterministic next if out-degree=1, else None (exit or handled by quality plugin)
    """
    dp_at_node: Dict[int, str]
    dp_actions: Dict[str, List[RouteAction]]
    dp_policy: Dict[str, RouteDecisionPolicy]
    default_next: Dict[int, Optional[int]]

    def next_node(
        self,
        rng: np.random.Generator,
        job: Job,
        from_node: int,
        state: QueueStateSnapshot,
        *,
        override_dp_id: Optional[str] = None,
    ) -> Optional[int]:
        dp_id = override_dp_id or self.dp_at_node.get(int(from_node), None)
        if dp_id is None:
            return self.default_next.get(int(from_node), None)

        actions = self.dp_actions[dp_id]
        pol = self.dp_policy[dp_id]
        a = int(pol.select(rng, dp_id=dp_id, job=job, from_node=int(from_node), actions=actions, state=state))
        mask = build_action_mask(dp_actions=actions, job=job)
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            return actions[0].to
        if a not in valid:
            a = int(valid[0])
        return actions[a].to


# --- Agent interface + adapter (for PPO / CRN evaluation) ---

class RoutingDecisionAgent:
    def act(self, dp_id: str, obs: np.ndarray, mask: np.ndarray, *, deterministic: bool = True) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class AgentRouteDecisionPolicy(RouteDecisionPolicy):
    agent: RoutingDecisionAgent
    obs_spec: Any  # RoutingObsSpec
    deterministic: bool = True

    def select(self, rng, *, dp_id, job, from_node, actions, state) -> int:
        obs = self.obs_spec.encode(job=job, from_node=int(from_node), state=state)
        mask = build_action_mask(dp_actions=actions, job=job)
        a = int(self.agent.act(dp_id, obs, mask, deterministic=self.deterministic))
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            return 0
        if a not in valid:
            a = int(valid[0])
        return a


@dataclass(frozen=True)
class PPORoutingAgentAdapter(RoutingDecisionAgent):
    ppo_agent: Any

    def act(self, dp_id: str, obs: np.ndarray, mask: np.ndarray, *, deterministic: bool = True) -> int:
        a, _, _ = self.ppo_agent.act(dp_id, obs, mask, deterministic=deterministic)
        return int(a)