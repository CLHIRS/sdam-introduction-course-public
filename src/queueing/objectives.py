# queueing/objectives.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from queueing.core_types import Job


@dataclass(frozen=True)
class ObjectiveSpec:
    holding_cost_per_job_per_time: float = 0.0
    tardiness_penalty_per_time: float = 0.0
    rework_penalty: float = 0.0
    throughput_reward: Dict[Any, float] = None
    scrap_penalty: Dict[Any, float] = None

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "ObjectiveSpec":
        obj = (cfg or {}).get("objective", {}) if cfg is not None else {}
        return ObjectiveSpec(
            holding_cost_per_job_per_time=float(obj.get("holding_cost_per_job_per_time", 0.0)),
            tardiness_penalty_per_time=float(obj.get("tardiness_penalty_per_time", 0.0)),
            rework_penalty=float(obj.get("rework_penalty", 0.0)),
            throughput_reward=dict(obj.get("throughput_reward", {})),
            scrap_penalty=dict(obj.get("scrap_penalty", {})),
        )


@dataclass(frozen=True)
class QualitySpec:
    """
    Compiled from cfg["quality_model"] AFTER node-id -> node-index mapping.
    If a node id is absent => store -1 (meaning "not configured").
    """
    assembly_nodes: set[int]
    rework_node: int
    fg_node: int
    scrap_node: int
    defect_prob: Dict[int, Dict[Any, float]]
    rework_success_prob: float

    @staticmethod
    def from_quality_cfg(quality_cfg: Dict[str, Any]) -> "QualitySpec":
        if not quality_cfg:
            return QualitySpec(
                assembly_nodes=set(),
                rework_node=-1,
                fg_node=-1,
                scrap_node=-1,
                defect_prob={},
                rework_success_prob=1.0,
            )
        return QualitySpec(
            assembly_nodes=set(int(x) for x in quality_cfg.get("assembly_nodes", [])),
            rework_node=int(quality_cfg.get("rework_node", -1)),
            fg_node=int(quality_cfg.get("fg_node", -1)),
            scrap_node=int(quality_cfg.get("scrap_node", -1)),
            defect_prob={int(k): dict(v) for k, v in quality_cfg.get("defect_prob", {}).items()},
            rework_success_prob=float(quality_cfg.get("rework_success_prob", 1.0)),
        )


def sample_defect(rng: np.random.Generator, q: QualitySpec, *, node: int, job: Job) -> bool:
    p = float(q.defect_prob.get(int(node), {}).get(job.job_class, 0.0))
    return bool(rng.random() < p)


def rework_outcome_next_node(rng: np.random.Generator, q: QualitySpec) -> Optional[int]:
    """
    Returns fg_node or scrap_node if BOTH configured. Otherwise returns None (no override).
    """
    if q.fg_node < 0 or q.scrap_node < 0:
        return None
    ok = bool(rng.random() < q.rework_success_prob)
    return int(q.fg_node if ok else q.scrap_node)


def terminal_reward(obj: ObjectiveSpec, *, job: Job, t_done: float, terminal_kind: str) -> float:
    """
    terminal_kind: "fg" | "scrap" | "exit" | "drop"
    Reward convention: higher is better. (So cost = -reward if you want.)
    """
    r = 0.0

    # throughput reward only for finished goods
    if terminal_kind == "fg":
        if obj.throughput_reward is not None:
            r += float(obj.throughput_reward.get(job.job_class, 0.0))

    # scrap penalty only when explicitly scrapped
    if terminal_kind == "scrap":
        if obj.scrap_penalty is not None:
            r -= float(obj.scrap_penalty.get(job.job_class, 0.0))

    # rework penalty per rework pass
    if obj.rework_penalty and job.n_rework > 0:
        r -= float(obj.rework_penalty) * float(job.n_rework)

    # tardiness penalty (apply on any terminal kind)
    if obj.tardiness_penalty_per_time and job.due_time is not None:
        tard = max(float(t_done - job.due_time), 0.0)
        r -= float(obj.tardiness_penalty_per_time) * tard

    return float(r)