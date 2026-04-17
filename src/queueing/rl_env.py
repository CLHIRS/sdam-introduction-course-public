"""queueing/rl_env.py

Routing-only RL environment.

This module is intentionally kept *independent* of evaluation code (eval_crn.py)
to avoid circular imports.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from queueing.core_types import Event, Job, QueueingNetworkConf, QueueStateSnapshot
from queueing.crn import bernoulli, rng_for
from queueing.dispatch import FIFO, DisciplineDispatchPolicy
from queueing.objectives import ObjectiveSpec, QualitySpec, terminal_reward

if TYPE_CHECKING:
    from queueing.routing_policies import RouteAction


@dataclass(frozen=True)
class RoutingObsSpec:
    node_ids: List[int]
    class_labels: List[Any]

    def encode(self, *, job: Job, from_node: int, state: QueueStateSnapshot) -> np.ndarray:
        """Simple, stable observation encoding for routing decisions."""
        cls_onehot = np.zeros(len(self.class_labels), dtype=np.float32)
        if job.job_class in self.class_labels:
            cls_onehot[self.class_labels.index(job.job_class)] = 1.0

        slack = 0.0
        if job.due_time is not None:
            slack = float(job.due_time - state.t)

        q = np.asarray(state.queue_lens, dtype=np.float32)
        proc = np.asarray(state.in_service, dtype=np.float32)
        blk = np.asarray(state.blocked, dtype=np.float32)

        from_onehot = np.zeros(len(self.node_ids), dtype=np.float32)
        if int(from_node) in self.node_ids:
            from_onehot[self.node_ids.index(int(from_node))] = 1.0

        # [class onehot | from onehot | slack | queue | in_service | blocked]
        return np.concatenate([cls_onehot, from_onehot, np.array([slack], dtype=np.float32), q, proc, blk], axis=0)


@dataclass(frozen=True)
class DecisionContext:
    dp_id: str
    job_id: int
    from_node: int
    actions: List["RouteAction"]


class RoutingOnlyEnv:
    """A step-wise environment that pauses at routing decisions.

    Reward convention: higher is better.
    - Holding cost: negative reward rate proportional to WIP.
    - Terminal reward/penalties: from ObjectiveSpec.
    """

    def __init__(
        self,
        conf: QueueingNetworkConf,
        *,
        cfg: Dict[str, Any],
        quality_cfg: Optional[Dict[str, Any]] = None,
        obs_spec: Optional[RoutingObsSpec] = None,
        seed: int = 0,
        T: float = 500.0,
        max_events: int = 2_000_000,
    ):
        if conf.routing_policy is None:
            raise ValueError("QueueingNetworkConf.routing_policy must be set")
        self.conf = conf
        self.cfg = cfg
        self.base_seed = int(seed)
        self.T = float(T)
        self.max_events = int(max_events)
        self.obs_spec = obs_spec

        self.qspec = QualitySpec.from_quality_cfg(quality_cfg or {})
        self.obj = ObjectiveSpec.from_cfg(cfg)

        self.dispatch = conf.dispatch_policy or DisciplineDispatchPolicy(FIFO())

        self._reset_state()

    def _reset_state(self) -> None:
        n = int(self.conf.n_nodes)
        self.servers = np.asarray(self.conf.servers, dtype=int)
        self.cap = np.asarray(self.conf.capacity, dtype=float)

        self.waiting: List[Deque[int]] = [deque() for _ in range(n)]
        self.in_service: List[List[int]] = [[] for _ in range(n)]
        self.blocked: List[List[int]] = [[] for _ in range(n)]
        self.blocked_to_dest: List[Deque[Tuple[int, int]]] = [deque() for _ in range(n)]

        self.jobs: Dict[int, Job] = {}
        self.job_id_seq = 0

        self.event_q: List[Event] = []
        self.seq_box = [0]

        # CRN counters
        self.arrival_idx = np.zeros(n, dtype=int)
        self.dispatch_cnt = np.zeros(n, dtype=int)
        self.service_visit: Dict[Tuple[int, int], int] = {}
        self.quality_visit: Dict[Tuple[int, int], int] = {}
        self.rework_visit: Dict[int, int] = {}
        self.route_decisions: Dict[int, int] = {}

        # metrics / reward accounting
        self.t = 0.0
        self.last_t = 0.0
        self.total_reward = 0.0
        self.n_completed = 0
        self.lost_external = 0
        self.dropped_internal = 0

        # pending decision
        self._pending: Optional[Tuple[str, int, int]] = None  # (dp_id, jid, from_node)

        # set per-decision RNG (so notebooks can call dp.select(env.rng, ...))
        self.rng: np.random.Generator = np.random.default_rng(self.base_seed)

    def _node_in_system(self, i: int) -> int:
        return len(self.waiting[i]) + len(self.in_service[i]) + len(self.blocked[i])

    def _can_accept(self, i: int) -> bool:
        if np.isinf(self.cap[i]):
            return True
        return self._node_in_system(i) < int(self.cap[i])

    def _snapshot(self) -> QueueStateSnapshot:
        n = int(self.conf.n_nodes)
        qlens = np.array([len(self.waiting[i]) for i in range(n)], dtype=float)
        proc = np.array([len(self.in_service[i]) for i in range(n)], dtype=float)
        blk = np.array([len(self.blocked[i]) for i in range(n)], dtype=float)
        return QueueStateSnapshot(
            t=float(self.t),
            queue_lens=qlens,
            in_service=proc,
            blocked=blk,
            servers=self.servers.astype(float),
            capacity=self.cap.copy(),
        )

    def _update_reward_to(self, t_new: float) -> None:
        t_new = float(t_new)
        dt = t_new - float(self.last_t)
        if dt <= 0:
            return
        wip = float(sum(self._node_in_system(i) for i in range(int(self.conf.n_nodes))))
        self.total_reward += -float(self.obj.holding_cost_per_job_per_time) * wip * dt
        self.last_t = t_new

    def _dispatch_rng(self, node: int) -> np.random.Generator:
        self.dispatch_cnt[node] += 1
        return rng_for(self.base_seed, "dispatch", int(node), int(self.dispatch_cnt[node]))

    def _routing_rng(self, jid: int, from_node: int, dp_id: str) -> np.random.Generator:
        k = self.route_decisions.get(jid, 0) + 1
        self.route_decisions[jid] = k
        return rng_for(self.base_seed, "route", int(jid), int(from_node), int(k), dp_id)

    def _service_time(self, jid: int, node: int, job: Job) -> float:
        key = (jid, node)
        k = self.service_visit.get(key, 0) + 1
        self.service_visit[key] = k
        r = rng_for(self.base_seed, "service", int(jid), int(node), int(k))
        return float(self.conf.services[node].sample_service_time(r, job, node, self.t))

    def _sample_defect(self, jid: int, node: int, job: Job) -> bool:
        p = float(self.qspec.defect_prob.get(int(node), {}).get(job.job_class, 0.0))
        key = (jid, node)
        k = self.quality_visit.get(key, 0) + 1
        self.quality_visit[key] = k
        return bool(bernoulli(self.base_seed, p, "defect", int(jid), int(node), int(k)))

    def _rework_outcome(self, jid: int) -> Optional[int]:
        if self.qspec.fg_node < 0 or self.qspec.scrap_node < 0:
            return None
        k = self.rework_visit.get(jid, 0) + 1
        self.rework_visit[jid] = k
        ok = bernoulli(self.base_seed, float(self.qspec.rework_success_prob), "rework", int(jid), int(k))
        return int(self.qspec.fg_node if ok else self.qspec.scrap_node)

    def _start_service_if_possible(self, i: int, t: float) -> None:
        while (len(self.in_service[i]) + len(self.blocked[i])) < int(self.servers[i]) and self.waiting[i]:
            drng = self._dispatch_rng(i)
            jid = int(self.dispatch.select_job(drng, i, self.waiting[i], self.jobs, self._snapshot()))
            job = self.jobs[jid]
            job.waiting_time_total += float(t - job.t_last_node_arrival)
            st = self._service_time(jid, i, job)
            job.service_time_total += float(st)
            self.in_service[i].append(jid)
            self.seq_box[0] += 1
            heapq.heappush(self.event_q, Event(t=float(t + st), seq=self.seq_box[0], kind="service_done", node=i, job_id=jid))

    def _try_unblock_to(self, dest: int, t: float) -> None:
        while self.blocked_to_dest[dest] and self._can_accept(dest):
            from_node, jid = self.blocked_to_dest[dest][0]
            if jid not in self.jobs:
                self.blocked_to_dest[dest].popleft()
                continue
            if jid in self.blocked[from_node]:
                self.blocked[from_node].remove(jid)
            self.blocked_to_dest[dest].popleft()

            job = self.jobs[jid]
            job.t_last_node_arrival = float(t)
            self.waiting[dest].append(jid)
            self._start_service_if_possible(dest, t)
            self._start_service_if_possible(from_node, t)

    def reset_and_get_first(self) -> Tuple[DecisionContext, float, bool, Dict[str, Any]]:
        self._reset_state()

        # schedule initial arrivals for nodes with an arrival process
        for i in range(int(self.conf.n_nodes)):
            self.arrival_idx[i] += 1
            r = rng_for(self.base_seed, "arrival", int(i), int(self.arrival_idx[i]))
            ia = float(self.conf.arrivals[i].sample_interarrival(r, 0.0))
            if np.isfinite(ia):
                self.seq_box[0] += 1
                heapq.heappush(self.event_q, Event(t=float(ia), seq=self.seq_box[0], kind="ext_arrival", node=i, job_id=None))

        ctx, r, done, info = self.step_until_decision()
        return ctx, r, done, info

    def step_until_decision(self) -> Tuple[DecisionContext, float, bool, Dict[str, Any]]:
        if self._pending is not None:
            raise RuntimeError("step_until_decision called while a decision is pending; call apply_action first")

        reward_before = float(self.total_reward)
        n_events = 0

        while self.event_q and n_events < self.max_events:
            ev = heapq.heappop(self.event_q)
            if float(ev.t) > self.T:
                self._update_reward_to(self.T)
                self.t = float(self.T)
                break

            self._update_reward_to(float(ev.t))
            self.t = float(ev.t)
            n_events += 1

            if ev.kind == "ext_arrival":
                # schedule next arrival at this node
                self.arrival_idx[ev.node] += 1
                r = rng_for(self.base_seed, "arrival", int(ev.node), int(self.arrival_idx[ev.node]))
                ia = float(self.conf.arrivals[ev.node].sample_interarrival(r, self.t))
                if np.isfinite(ia):
                    self.seq_box[0] += 1
                    heapq.heappush(self.event_q, Event(t=float(self.t + ia), seq=self.seq_box[0], kind="ext_arrival", node=ev.node, job_id=None))

                if not self._can_accept(ev.node):
                    self.lost_external += 1
                    continue

                self.job_id_seq += 1
                jid = int(self.job_id_seq)
                r_cls = rng_for(self.base_seed, "jobclass", int(jid))
                cls = self.conf.job_class(r_cls, self.t, ev.node) if self.conf.job_class else None

                due = None
                if self.conf.due_date is not None:
                    r_due = rng_for(self.base_seed, "duedate", int(jid))
                    due = float(self.conf.due_date(r_due, self.t))

                self.jobs[jid] = Job(
                    job_id=jid,
                    t_enter_system=float(self.t),
                    job_class=cls,
                    due_time=due,
                    t_last_node_arrival=float(self.t),
                )
                self.waiting[ev.node].append(jid)
                self._start_service_if_possible(ev.node, self.t)

            elif ev.kind == "service_done":
                i = int(ev.node)
                jid = int(ev.job_id) if ev.job_id is not None else -1
                if jid <= 0:
                    continue

                try:
                    self.in_service[i].remove(jid)
                except ValueError:
                    continue

                job = self.jobs.get(jid)
                if job is None:
                    continue

                override_dp: Optional[str] = None
                if i in self.qspec.assembly_nodes:
                    job.is_defective = self._sample_defect(jid, i, job)
                    if job.is_defective:
                        override_dp = "defect_decision"

                # leaving rework can override next-node to fg/scrap sinks
                if i == self.qspec.rework_node:
                    nxt_override = self._rework_outcome(jid)
                    if nxt_override is not None:
                        # treat as immediate transfer without a decision
                        self._route_job(jid, from_node=i, to_node=int(nxt_override), t_now=self.t)
                        continue

                dp_id = override_dp or self.conf.routing_policy.dp_at_node.get(i, None)
                if dp_id is not None:
                    self._pending = (str(dp_id), int(jid), int(i))
                    self.rng = self._routing_rng(int(jid), int(i), str(dp_id))
                    actions = self.conf.routing_policy.dp_actions[str(dp_id)]
                    ctx = DecisionContext(dp_id=str(dp_id), job_id=int(jid), from_node=int(i), actions=actions)
                    r_step = float(self.total_reward - reward_before)
                    return ctx, r_step, False, {}

                # no decision point => use configured default routing
                rrng = self._routing_rng(int(jid), int(i), "")
                nxt = self.conf.routing_policy.next_node(rrng, job, int(i), self._snapshot(), override_dp_id=None)
                if nxt is None:
                    self._terminate_job(jid, terminal_kind="exit", t_done=self.t)
                else:
                    self._route_job(jid, from_node=i, to_node=int(nxt), t_now=self.t)

            else:
                raise RuntimeError(f"Unknown event kind: {ev.kind}")

        # end loop
        done = (self.t >= self.T) or (not self.event_q)
        r_step = float(self.total_reward - reward_before)
        info = {"t": float(self.t), "events": int(n_events)}
        # if done and no further decisions, return a dummy context is awkward; caller expects a ctx.
        # We return the last available ctx by forcing an exception-free sentinel.
        if done:
            # Create a trivial ctx (no-op) to keep the interface stable.
            ctx = DecisionContext(dp_id="__done__", job_id=-1, from_node=-1, actions=[])
            return ctx, r_step, True, info

        # If we exited due to max_events, mark done.
        ctx = DecisionContext(dp_id="__done__", job_id=-1, from_node=-1, actions=[])
        return ctx, r_step, True, {**info, "truncated": True}

    def _terminate_job(self, jid: int, *, terminal_kind: str, t_done: float) -> None:
        job = self.jobs.pop(int(jid), None)
        if job is None:
            return
        if terminal_kind in {"fg", "exit"}:
            self.n_completed += 1
        self.total_reward += float(terminal_reward(self.obj, job=job, t_done=float(t_done), terminal_kind=str(terminal_kind)))

    def _route_job(self, jid: int, *, from_node: int, to_node: int, t_now: float) -> None:
        job = self.jobs.get(int(jid))
        if job is None:
            return

        # sinks: if node has no outgoing edge, treat as terminal
        if to_node == self.qspec.scrap_node:
            self._terminate_job(jid, terminal_kind="scrap", t_done=t_now)
            return
        if to_node == self.qspec.fg_node:
            self._terminate_job(jid, terminal_kind="fg", t_done=t_now)
            return

        if int(to_node) == int(self.qspec.rework_node) and int(self.qspec.rework_node) >= 0:
            job.n_rework += 1

        if self._can_accept(int(to_node)):
            job.t_last_node_arrival = float(t_now)
            self.waiting[int(to_node)].append(int(jid))
            self._start_service_if_possible(int(to_node), float(t_now))
        else:
            if self.conf.overflow_mode == "drop":
                self.dropped_internal += 1
                self._terminate_job(jid, terminal_kind="drop", t_done=t_now)
            else:
                self.blocked[int(from_node)].append(int(jid))
                self.blocked_to_dest[int(to_node)].append((int(from_node), int(jid)))

        self._start_service_if_possible(int(from_node), float(t_now))
        self._try_unblock_to(int(from_node), float(t_now))

    def apply_action(self, ctx: DecisionContext, action_index: int) -> None:
        if self._pending is None:
            raise RuntimeError("No pending decision")
        dp_id, jid, from_node = self._pending
        if ctx.dp_id != dp_id or ctx.job_id != jid or ctx.from_node != from_node:
            raise ValueError("ctx does not match pending decision")
        if not (0 <= int(action_index) < len(ctx.actions)):
            raise ValueError("action_index out of range")

        act = ctx.actions[int(action_index)]
        to = act.to

        self._pending = None

        if to is None:
            self._terminate_job(int(jid), terminal_kind="exit", t_done=float(self.t))
            self._start_service_if_possible(int(from_node), float(self.t))
            self._try_unblock_to(int(from_node), float(self.t))
            return

        self._route_job(int(jid), from_node=int(from_node), to_node=int(to), t_now=float(self.t))