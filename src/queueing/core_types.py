# queueing/core_types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from queueing.processes import ArrivalProcess, ServiceProcess

if TYPE_CHECKING:
    from queueing.dispatch import DispatchPolicy, QueueDiscipline

if TYPE_CHECKING:
    from queueing.routing_policies import DecisionPointRoutingPolicy


# ---------------------------
# Core state / event / snapshot
# ---------------------------

@dataclass
class Job:
    job_id: int
    t_enter_system: float
    job_class: Any = None
    due_time: Optional[float] = None
    t_last_node_arrival: float = 0.0
    waiting_time_total: float = 0.0
    service_time_total: float = 0.0

    # used by quality/rework logic
    is_defective: bool = False
    n_rework: int = 0


@dataclass(order=True)
class Event:
    t: float
    seq: int
    kind: str = field(compare=False)  # "ext_arrival" | "service_done"
    node: int = field(compare=False)
    job_id: Optional[int] = field(compare=False, default=None)


@dataclass(frozen=True)
class QueueStateSnapshot:
    t: float
    queue_lens: np.ndarray
    in_service: np.ndarray
    blocked: np.ndarray
    servers: np.ndarray
    capacity: np.ndarray


# ---------------------------
# KPIs
# ---------------------------

@dataclass(frozen=True)
class QueueingNetworkKPIs:
    sim_time: float
    n_completed: int
    throughput: float
    mean_cycle_time: float
    mean_waiting_time: float
    mean_tardiness: float
    on_time_delivery: float

    wip_time_avg: float
    queue_len_time_avg: np.ndarray
    in_system_time_avg: np.ndarray

    utilization_processing: np.ndarray
    blocked_time_avg: np.ndarray
    starved_time_avg: np.ndarray
    idle_time: np.ndarray

    lost_external: int
    dropped_internal: int
    blocked_internal: int
    unblocked_internal: int
    fill_rate: float

    by_class: Dict[Any, Dict[str, float]]


# ---------------------------
# Network configuration
# ---------------------------

@dataclass(frozen=True)
class QueueingNetworkConf:
    n_nodes: int
    arrivals: List[ArrivalProcess]     # len n_nodes
    services: List[ServiceProcess]     # len n_nodes
    servers: np.ndarray                # shape (n_nodes,)
    capacity: np.ndarray               # shape (n_nodes,), np.inf ok
    discipline: List["QueueDiscipline"]  # len n_nodes (optional use)
    overflow_mode: str = "block"       # "drop" | "block"

    due_date: Optional[Any] = None     # due_date(rng, t)->float
    job_class: Optional[Any] = None    # job_class(rng, t, node)->label

    dispatch_policy: Optional["DispatchPolicy"] = None
    routing_policy: Optional["DecisionPointRoutingPolicy"] = None  # forward ref avoids circular import