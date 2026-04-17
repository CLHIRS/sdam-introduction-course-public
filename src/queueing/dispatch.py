# queueing/dispatch.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict

import numpy as np
from queueing.core_types import Job, QueueStateSnapshot


class QueueDiscipline:
    def push(self, q: Deque[int], job_id: int) -> None:
        raise NotImplementedError

    def pop(self, q: Deque[int]) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class FIFO(QueueDiscipline):
    def push(self, q: Deque[int], job_id: int) -> None:
        q.append(job_id)

    def pop(self, q: Deque[int]) -> int:
        return q.popleft()


class DispatchPolicy:
    def select_job(
        self,
        rng: np.random.Generator,
        node: int,
        q: Deque[int],
        jobs: Dict[int, Job],
        state: QueueStateSnapshot,
    ) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class DisciplineDispatchPolicy(DispatchPolicy):
    discipline: QueueDiscipline

    def select_job(self, rng, node, q, jobs, state) -> int:
        return int(self.discipline.pop(q))


@dataclass(frozen=True)
class EDDDispatchPolicy(DispatchPolicy):
    """
    Earliest Due Date (EDD). If all due times missing, falls back to FIFO order.
    """
    def select_job(self, rng, node, q, jobs, state) -> int:
        if not q:
            raise RuntimeError("select_job called with empty queue")
        best_jid = None
        best_due = float("inf")
        for jid in list(q):
            due = jobs[jid].due_time
            d = float(due) if due is not None else float("inf")
            if d < best_due:
                best_due = d
                best_jid = jid
        if best_jid is None:
            best_jid = q[0]
        q.remove(best_jid)
        return int(best_jid)


@dataclass(frozen=True)
class PerNodeDispatchPolicy(DispatchPolicy):
    """
    Provide a dispatch policy per node index; fallback is FIFO if missing.
    """
    per_node: Dict[int, DispatchPolicy]
    fallback: DispatchPolicy = field(default_factory=lambda: DisciplineDispatchPolicy(FIFO()))

    def select_job(self, rng, node, q, jobs, state) -> int:
        pol = self.per_node.get(int(node), self.fallback)
        return int(pol.select_job(rng, node, q, jobs, state))