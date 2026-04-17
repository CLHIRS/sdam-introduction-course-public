# queueing/sim.py
from __future__ import annotations

import heapq
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from queueing.core_types import Event, Job, QueueingNetworkConf, QueueingNetworkKPIs, QueueStateSnapshot
from queueing.crn import bernoulli, rng_for
from queueing.dispatch import FIFO, DisciplineDispatchPolicy
from queueing.objectives import QualitySpec


class QueueingNetworkSim:
    def __init__(self, conf: QueueingNetworkConf, *, quality_cfg: Optional[Dict[str, Any]] = None):
        self.conf = conf
        self.quality_cfg = quality_cfg or {}

        self.dispatch = conf.dispatch_policy or DisciplineDispatchPolicy(FIFO())

        if conf.routing_policy is None:
            raise ValueError("QueueingNetworkConf.routing_policy must be set (DecisionPointRoutingPolicy).")

        self.qspec = QualitySpec.from_quality_cfg(self.quality_cfg)

    def run(self, T: float, seed: int = 0, max_events: int = 2_000_000) -> QueueingNetworkKPIs:
        if T <= 0:
            raise ValueError("T must be > 0")

        base_seed = int(seed)
        n = self.conf.n_nodes
        servers = np.asarray(self.conf.servers, dtype=int)
        cap = np.asarray(self.conf.capacity, dtype=float)

        waiting: List[Deque[int]] = [deque() for _ in range(n)]
        in_service: List[List[int]] = [[] for _ in range(n)]
        blocked: List[List[int]] = [[] for _ in range(n)]
        blocked_to_dest: List[Deque[Tuple[int, int]]] = [deque() for _ in range(n)]  # (from_node, jid)

        jobs: Dict[int, Job] = {}
        job_id_seq = 0

        # CRN counters (make randomness order-invariant)
        arrival_idx = np.zeros(n, dtype=int)
        dispatch_cnt = np.zeros(n, dtype=int)
        service_visit: Dict[Tuple[int, int], int] = {}    # (jid,node)->k
        quality_visit: Dict[Tuple[int, int], int] = {}    # (jid,node)->k
        rework_visit: Dict[int, int] = {}                 # jid->k
        route_decisions: Dict[int, int] = {}              # jid->k

        lost_external = 0
        dropped_internal = 0
        blocked_internal = 0
        unblocked_internal = 0
        n_external_arrivals = 0

        last_t = 0.0
        area_queue = np.zeros(n, dtype=float)
        area_in_system = np.zeros(n, dtype=float)
        area_processing = np.zeros(n, dtype=float)
        area_blocked = np.zeros(n, dtype=float)
        area_starved = np.zeros(n, dtype=float)

        completed_cycle_times: List[float] = []
        completed_wait_times: List[float] = []
        completed_tardiness: List[float] = []
        completed_on_time = 0

        by_class: Dict[Any, Dict[str, float]] = {}

        def cls_bucket(cls: Any) -> Dict[str, float]:
            if cls not in by_class:
                by_class[cls] = {"n": 0.0, "sum_cycle": 0.0, "sum_wait": 0.0, "sum_tard": 0.0, "on_time": 0.0}
            return by_class[cls]

        def node_in_system(i: int) -> int:
            return len(waiting[i]) + len(in_service[i]) + len(blocked[i])

        def snapshot(t: float) -> QueueStateSnapshot:
            qlens = np.array([len(waiting[i]) for i in range(n)], dtype=float)
            proc = np.array([len(in_service[i]) for i in range(n)], dtype=float)
            blk = np.array([len(blocked[i]) for i in range(n)], dtype=float)
            return QueueStateSnapshot(
                t=float(t),
                queue_lens=qlens,
                in_service=proc,
                blocked=blk,
                servers=servers.astype(float),
                capacity=cap.copy(),
            )

        def update_areas(t: float) -> None:
            nonlocal last_t
            dt = float(t - last_t)
            if dt <= 0:
                return

            qlens = np.array([len(waiting[i]) for i in range(n)], dtype=float)
            in_sys = np.array([node_in_system(i) for i in range(n)], dtype=float)
            proc = np.array([len(in_service[i]) for i in range(n)], dtype=float)
            blk = np.array([len(blocked[i]) for i in range(n)], dtype=float)

            starv = np.zeros(n, dtype=float)
            for i in range(n):
                busy_total = len(in_service[i]) + len(blocked[i])
                if not waiting[i] and busy_total < servers[i]:
                    starv[i] = float(servers[i] - busy_total)

            area_queue[:] += qlens * dt
            area_in_system[:] += in_sys * dt
            area_processing[:] += proc * dt
            area_blocked[:] += blk * dt
            area_starved[:] += starv * dt
            last_t = float(t)

        def can_accept(i: int) -> bool:
            if np.isinf(cap[i]):
                return True
            return node_in_system(i) < int(cap[i])

        def _dispatch_rng(node: int) -> np.random.Generator:
            dispatch_cnt[node] += 1
            return rng_for(base_seed, "dispatch", int(node), int(dispatch_cnt[node]))

        def _routing_rng(jid: int, from_node: int, dp_id: str) -> np.random.Generator:
            k = route_decisions.get(jid, 0) + 1
            route_decisions[jid] = k
            return rng_for(base_seed, "route", int(jid), int(from_node), int(k), dp_id)

        def _service_time(jid: int, node: int, job: Job) -> float:
            key = (jid, node)
            k = service_visit.get(key, 0) + 1
            service_visit[key] = k
            # create a per-call rng so service times are tied to (job,node,visit)
            # and NOT to global event ordering.
            r = rng_for(base_seed, "service", int(jid), int(node), int(k))
            return float(self.conf.services[node].sample_service_time(r, job, node, last_t))

        def _sample_defect(jid: int, node: int, job: Job) -> bool:
            p = float(self.qspec.defect_prob.get(int(node), {}).get(job.job_class, 0.0))
            key = (jid, node)
            k = quality_visit.get(key, 0) + 1
            quality_visit[key] = k
            return bernoulli(base_seed, p, "defect", int(jid), int(node), int(k))

        def _rework_outcome(jid: int) -> Optional[int]:
            if self.qspec.fg_node < 0 or self.qspec.scrap_node < 0:
                return None
            k = rework_visit.get(jid, 0) + 1
            rework_visit[jid] = k
            ok = bernoulli(base_seed, float(self.qspec.rework_success_prob), "rework", int(jid), int(k))
            return int(self.qspec.fg_node if ok else self.qspec.scrap_node)

        def start_service_if_possible(i: int, t: float, event_q: List[Event], seq_box: List[int]) -> None:
            while (len(in_service[i]) + len(blocked[i])) < servers[i] and waiting[i]:
                drng = _dispatch_rng(i)
                jid = self.dispatch.select_job(drng, i, waiting[i], jobs, snapshot(t))
                job = jobs[jid]
                job.waiting_time_total += float(t - job.t_last_node_arrival)

                st = _service_time(jid, i, job)
                job.service_time_total += float(st)

                in_service[i].append(jid)
                seq_box[0] += 1
                heapq.heappush(event_q, Event(t=float(t + st), seq=seq_box[0], kind="service_done", node=i, job_id=jid))

        def try_unblock_to(dest: int, t: float, event_q: List[Event], seq_box: List[int]) -> None:
            nonlocal unblocked_internal
            while blocked_to_dest[dest] and can_accept(dest):
                from_node, jid = blocked_to_dest[dest][0]
                if jid not in jobs:
                    blocked_to_dest[dest].popleft()
                    continue
                if jid in blocked[from_node]:
                    blocked[from_node].remove(jid)
                blocked_to_dest[dest].popleft()
                unblocked_internal += 1

                job = jobs[jid]
                job.t_last_node_arrival = float(t)
                waiting[dest].append(jid)
                start_service_if_possible(dest, t, event_q, seq_box)
                start_service_if_possible(from_node, t, event_q, seq_box)

        event_q: List[Event] = []
        seq_box = [0]

        # initial arrivals: schedule one per node with external arrival process
        for i in range(n):
            # use arrival index 1 for the first scheduled arrival at node i
            arrival_idx[i] += 1
            r = rng_for(base_seed, "arrival", int(i), int(arrival_idx[i]))
            ia = float(self.conf.arrivals[i].sample_interarrival(r, 0.0))
            if np.isfinite(ia):
                seq_box[0] += 1
                heapq.heappush(event_q, Event(t=float(ia), seq=seq_box[0], kind="ext_arrival", node=i, job_id=None))

        n_events = 0
        while event_q and n_events < max_events:
            ev = heapq.heappop(event_q)
            if ev.t > T:
                update_areas(T)
                break
            update_areas(ev.t)
            n_events += 1

            if ev.kind == "ext_arrival":
                n_external_arrivals += 1

                # schedule next arrival at this node with arrival_idx++
                arrival_idx[ev.node] += 1
                r = rng_for(base_seed, "arrival", int(ev.node), int(arrival_idx[ev.node]))
                ia = float(self.conf.arrivals[ev.node].sample_interarrival(r, ev.t))
                if np.isfinite(ia):
                    seq_box[0] += 1
                    heapq.heappush(event_q, Event(t=float(ev.t + ia), seq=seq_box[0], kind="ext_arrival", node=ev.node, job_id=None))

                if not can_accept(ev.node):
                    lost_external += 1
                    continue

                job_id_seq += 1
                jid = job_id_seq

                # job class + due date: also should be CRN-stable by job id
                r_cls = rng_for(base_seed, "jobclass", int(jid))
                cls = self.conf.job_class(r_cls, ev.t, ev.node) if self.conf.job_class else None

                due = None
                if self.conf.due_date is not None:
                    r_due = rng_for(base_seed, "duedate", int(jid))
                    due = float(self.conf.due_date(r_due, ev.t))

                jobs[jid] = Job(job_id=jid, t_enter_system=float(ev.t), job_class=cls, due_time=due, t_last_node_arrival=float(ev.t))
                waiting[ev.node].append(jid)
                start_service_if_possible(ev.node, ev.t, event_q, seq_box)

            elif ev.kind == "service_done":
                i = int(ev.node)
                jid = int(ev.job_id) if ev.job_id is not None else -1
                if jid <= 0:
                    continue

                try:
                    in_service[i].remove(jid)
                except ValueError:
                    continue

                job = jobs.get(jid)
                if job is None:
                    continue

                override_dp = None

                # quality at assembly nodes: defect outcome is CRN-stable by (jid,node,visit)
                if i in self.qspec.assembly_nodes:
                    job.is_defective = _sample_defect(jid, i, job)
                    if job.is_defective:
                        override_dp = "defect_decision"

                # leaving rework: CRN-stable by (jid,rework_visit)
                if i == self.qspec.rework_node:
                    nxt_override = _rework_outcome(jid)
                    if nxt_override is not None:
                        nxt = int(nxt_override)
                    else:
                        rrng = _routing_rng(jid, i, override_dp or self.conf.routing_policy.dp_at_node.get(i, ""))
                        nxt = self.conf.routing_policy.next_node(rrng, job, i, snapshot(ev.t), override_dp_id=override_dp)
                else:
                    dp_guess = override_dp or self.conf.routing_policy.dp_at_node.get(i, "")
                    rrng = _routing_rng(jid, i, dp_guess)
                    nxt = self.conf.routing_policy.next_node(rrng, job, i, snapshot(ev.t), override_dp_id=override_dp)

                if nxt is not None and int(nxt) == self.qspec.rework_node:
                    job.n_rework += 1

                if nxt is None:
                    job = jobs.pop(jid, None)
                    if job is not None:
                        ct = float(ev.t - job.t_enter_system)
                        completed_cycle_times.append(ct)
                        completed_wait_times.append(float(job.waiting_time_total))
                        td = float(max(ev.t - job.due_time, 0.0)) if job.due_time is not None else 0.0
                        completed_tardiness.append(td)
                        if td <= 1e-12:
                            completed_on_time += 1

                        b = cls_bucket(job.job_class)
                        b["n"] += 1.0
                        b["sum_cycle"] += ct
                        b["sum_wait"] += float(job.waiting_time_total)
                        b["sum_tard"] += td
                        b["on_time"] += 1.0 if td <= 1e-12 else 0.0

                else:
                    j = int(nxt)
                    if can_accept(j):
                        job.t_last_node_arrival = float(ev.t)
                        waiting[j].append(jid)
                        start_service_if_possible(j, ev.t, event_q, seq_box)
                    else:
                        if self.conf.overflow_mode == "drop":
                            dropped_internal += 1
                            jobs.pop(jid, None)
                        else:
                            blocked_internal += 1
                            blocked[i].append(jid)
                            blocked_to_dest[j].append((i, jid))

                start_service_if_possible(i, ev.t, event_q, seq_box)
                try_unblock_to(i, ev.t, event_q, seq_box)

            else:
                raise RuntimeError(f"Unknown event kind: {ev.kind}")

        if last_t < T:
            update_areas(T)

        sim_time = float(T)
        n_completed = len(completed_cycle_times)
        throughput = n_completed / sim_time if sim_time > 0 else 0.0

        mean_cycle = float(np.mean(completed_cycle_times)) if n_completed else float("nan")
        mean_wait = float(np.mean(completed_wait_times)) if n_completed else float("nan")
        mean_tard = float(np.mean(completed_tardiness)) if n_completed else float("nan")
        on_time = (completed_on_time / n_completed) if n_completed else float("nan")

        in_sys_avg = area_in_system / sim_time
        q_avg = area_queue / sim_time
        wip_avg = float(in_sys_avg.sum())

        util_proc = area_processing / (servers * sim_time)
        blocked_avg = area_blocked / (servers * sim_time)
        starved_avg = area_starved / (servers * sim_time)

        busy_total = area_processing + area_blocked
        idle_time = servers * sim_time - busy_total

        fill_rate = (1.0 - (lost_external / n_external_arrivals)) if n_external_arrivals > 0 else float("nan")

        by_class_out: Dict[Any, Dict[str, float]] = {}
        for cls, b in by_class.items():
            n_c = b["n"]
            if n_c <= 0:
                continue
            by_class_out[cls] = {
                "n": n_c,
                "throughput": n_c / sim_time,
                "mean_cycle_time": b["sum_cycle"] / n_c,
                "mean_waiting_time": b["sum_wait"] / n_c,
                "mean_tardiness": b["sum_tard"] / n_c,
                "on_time_delivery": b["on_time"] / n_c,
            }

        return QueueingNetworkKPIs(
            sim_time=sim_time,
            n_completed=n_completed,
            throughput=throughput,
            mean_cycle_time=mean_cycle,
            mean_waiting_time=mean_wait,
            mean_tardiness=mean_tard,
            on_time_delivery=on_time,
            wip_time_avg=wip_avg,
            queue_len_time_avg=q_avg,
            in_system_time_avg=in_sys_avg,
            utilization_processing=util_proc,
            blocked_time_avg=blocked_avg,
            starved_time_avg=starved_avg,
            idle_time=idle_time,
            lost_external=lost_external,
            dropped_internal=dropped_internal,
            blocked_internal=blocked_internal,
            unblocked_internal=unblocked_internal,
            fill_rate=fill_rate,
            by_class=by_class_out,
        )