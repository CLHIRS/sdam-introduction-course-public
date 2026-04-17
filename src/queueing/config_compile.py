# queueing/config_compile.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from queueing.core_types import QueueingNetworkConf
from queueing.dispatch import DispatchPolicy, PerNodeDispatchPolicy
from queueing.processes import ExponentialService, NoArrival, PoissonArrival, ServiceProcess
from queueing.routing_policies import (
    DecisionPointRoutingPolicy,
    RouteAction,
    RouteDecisionPolicy,
)

# ---------- helpers: distributions ----------

def make_due_date_sampler(due_cfg: Optional[Dict[str, Any]]):
    """
    due_date(rng, t_now) -> absolute due_time
    Supported:
      - None => no due dates
      - {"type":"shifted_exp","mean":20,"shift":5}
    """
    if not due_cfg:
        return None

    typ = str(due_cfg.get("type", "shifted_exp"))
    if typ != "shifted_exp":
        raise ValueError(f"Unknown due_date_dist type: {typ}")

    mean = float(due_cfg.get("mean", 20.0))
    shift = float(due_cfg.get("shift", 0.0))
    if mean <= 0:
        raise ValueError("due_date_dist.mean must be > 0")

    def _due_date(rng: np.random.Generator, t_now: float) -> float:
        # due time = now + shift + Exp(mean)
        return float(t_now + shift + rng.exponential(mean))

    return _due_date


def make_job_class_sampler(arrival_cfg: Dict[str, Any]):
    """
    job_class(rng, t, node) -> class label
    Uses arrival_process.class_probs.
    """
    class_probs = dict(arrival_cfg.get("class_probs", {}))
    if not class_probs:
        # default single class
        labels = ["A"]
        probs = np.array([1.0], dtype=float)
    else:
        labels = list(class_probs.keys())
        probs = np.array([float(class_probs[k]) for k in labels], dtype=float)
        s = probs.sum()
        if s <= 0:
            raise ValueError("arrival_process.class_probs must sum to > 0")
        probs = probs / s

    def _job_class(rng: np.random.Generator, t: float, node: int):
        return labels[int(rng.choice(len(labels), p=probs))]

    return _job_class


# ---------- per-class service wrapper ----------

@dataclass(frozen=True)
class PerClassExponentialService(ServiceProcess):
    """
    Selects rate by job_class.
    Example mapping: {"A": 1.2, "B": 0.9}
    """
    rates: Dict[Any, float]

    def sample_service_time(self, rng: np.random.Generator, job, node: int, t: float) -> float:
        cls = job.job_class
        if cls not in self.rates:
            raise ValueError(f"Missing service rate for class={cls} at node={node}")
        mu = float(self.rates[cls])
        if mu <= 0:
            raise ValueError("Service rate must be > 0")
        return float(rng.exponential(1.0 / mu))


# ---------- main compiler ----------

def compile_queue_network_config(
    cfg: Dict[str, Any],
    *,
    routing_dp_policies: Dict[str, RouteDecisionPolicy],
    dispatch_by_node_name: Optional[Dict[str, DispatchPolicy]] = None,
    overflow_mode: str = "block",
) -> Tuple[QueueingNetworkConf, Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      - QueueingNetworkConf (DES-consumable)
      - quality_cfg (compiled, node-index based)
      - aux (node maps etc.)

    Expected cfg keys (as per your YAML-like draft):
      job_classes, arrival_process, nodes, edges, service, decision_points, quality_model, objective
    """
    nodes = list(cfg["nodes"])
    edges = list(cfg.get("edges", []))

    name_to_idx: Dict[str, int] = {}
    id_to_idx: Dict[int, int] = {}
    idx_to_name: List[str] = []

    for idx, nd in enumerate(nodes):
        name = str(nd["name"])
        nid = int(nd["id"])
        name_to_idx[name] = idx
        id_to_idx[nid] = idx
        idx_to_name.append(name)

    n = len(nodes)

    # arrivals: only nodes referenced by arrival_process (we assume external arrivals go to dispatcher/pool node)
    arrival_cfg = dict(cfg.get("arrival_process", {}))
    arr_type = str(arrival_cfg.get("type", "poisson"))
    if arr_type != "poisson":
        raise ValueError(f"Only poisson arrivals supported in this compiler, got: {arr_type}")
    lam = float(arrival_cfg.get("rate", 0.0))

    arrivals = [NoArrival() for _ in range(n)]
    # pick first node with kind "pool" as arrival node; else node 0
    arrival_node_idx = 0
    for i, nd in enumerate(nodes):
        if str(nd.get("kind", "")).lower() == "pool":
            arrival_node_idx = i
            break
    arrivals[arrival_node_idx] = PoissonArrival(rate=lam)

    # servers + capacity
    servers = np.ones(n, dtype=int)
    capacity = np.full(n, np.inf, dtype=float)

    for i, nd in enumerate(nodes):
        kind = str(nd.get("kind", "server")).lower()
        if kind in {"server"}:
            servers[i] = int(nd.get("servers", 1))
            # interpret buffer_capacity as total node capacity (waiting+service+blocked) for simplicity
            cap = nd.get("buffer_capacity", nd.get("capacity", np.inf))
            capacity[i] = float(cap) if cap is not None else np.inf
        elif kind in {"buffer", "pool"}:
            servers[i] = int(nd.get("servers", 1)) if "servers" in nd else 1
            cap = nd.get("capacity", nd.get("buffer_capacity", np.inf))
            capacity[i] = float(cap) if cap is not None else np.inf
        elif kind in {"sink"}:
            servers[i] = 1
            capacity[i] = np.inf
        else:
            raise ValueError(f"Unknown node kind: {kind}")

    # services: default "instant" servers for non-service nodes
    services: List[ServiceProcess] = [ExponentialService(rate=1e9) for _ in range(n)]

    service_cfg = dict(cfg.get("service", {}))  # keys are node ids in your draft (ints)
    for node_id_str, per_class in service_cfg.items():
        node_id = int(node_id_str)
        if node_id not in id_to_idx:
            raise ValueError(f"service references unknown node id {node_id}")
        i = id_to_idx[node_id]
        # per_class like {"A": {"dist":"exp","rate":1.2}, "B": {...}}
        rates: Dict[Any, float] = {}
        for cls, cinfo in dict(per_class).items():
            dist = str(cinfo.get("dist", "exp"))
            if dist != "exp":
                raise ValueError("Only exp service supported in this compiler")
            rates[cls] = float(cinfo["rate"])
        services[i] = PerClassExponentialService(rates=rates)

    # decision points -> dp_actions
    dp_actions: Dict[str, List[RouteAction]] = {}
    dp_at_node: Dict[int, str] = {}

    decision_points = list(cfg.get("decision_points", []))
    for dp in decision_points:
        dp_id = str(dp["id"])
        when = dict(dp["when"])
        event = str(when.get("event", "service_complete"))
        if event not in {"service_complete", "arrival_or_release", "quality_realized"}:
            raise ValueError(f"Unsupported decision point event: {event}")

        # Routing decisions at service completion are triggered by node index (dp_at_node).
        # Quality decisions are triggered by the simulator/env via override_dp_id (see sim.py / rl_env.py),
        # so they should NOT be mapped into dp_at_node.
        from_idx: Optional[int] = None
        if event in {"service_complete", "arrival_or_release"}:
            node = when.get("node", None)
            if node is None:
                raise ValueError(f"decision point {dp_id}: missing when.node")
            node_id = int(node)
            if node_id not in id_to_idx:
                raise ValueError(f"decision point {dp_id}: unknown node id {node_id}")
            from_idx = id_to_idx[node_id]
        else:
            # quality_realized: allow either a single node (when.node) or a list (when.nodes)
            nodes = when.get("nodes", None)
            node = when.get("node", None)
            if nodes is None and node is None:
                raise ValueError(f"decision point {dp_id}: missing when.node or when.nodes")
            if nodes is None:
                nodes = [node]
            for nid in list(nodes):
                node_id = int(nid)
                if node_id not in id_to_idx:
                    raise ValueError(f"decision point {dp_id}: unknown node id {node_id}")

        acts: List[RouteAction] = []
        for fa in dp.get("feasible_actions", []):
            to_id = fa.get("to", None)
            if to_id is None:
                to_idx = None
            else:
                to_idx = id_to_idx[int(to_id)]
            allowed = fa.get("allowed_classes", None)
            allowed_classes = list(allowed) if allowed is not None else None
            acts.append(RouteAction(to=to_idx, allowed_classes=allowed_classes))

        dp_actions[dp_id] = acts
        if from_idx is not None:
            dp_at_node[from_idx] = dp_id

    # default_next from edges: if out-degree==1 and not a dp node => deterministic next; else None
    out = {i: [] for i in range(n)}
    for e in edges:
        fr = id_to_idx[int(e["from"])]
        to = id_to_idx[int(e["to"])]
        out[fr].append(to)

    default_next: Dict[int, Optional[int]] = {}
    for i in range(n):
        if i in dp_at_node:
            default_next[i] = None
        else:
            if len(out[i]) == 1:
                default_next[i] = int(out[i][0])
            else:
                default_next[i] = None

    # hook up dp_policy dict using provided policies
    dp_policy: Dict[str, RouteDecisionPolicy] = {}
    for dp_id in dp_actions.keys():
        if dp_id not in routing_dp_policies:
            raise ValueError(f"Missing routing_dp_policies entry for dp_id='{dp_id}'")
        dp_policy[dp_id] = routing_dp_policies[dp_id]

    routing_policy = DecisionPointRoutingPolicy(
        dp_at_node=dp_at_node,
        dp_actions=dp_actions,
        dp_policy=dp_policy,
        default_next=default_next,
    )

    # dispatch policy: map node names to indices
    dispatch_pol: Optional[DispatchPolicy] = None
    if dispatch_by_node_name:
        per: Dict[int, DispatchPolicy] = {}
        for nm, pol in dispatch_by_node_name.items():
            if nm not in name_to_idx:
                raise ValueError(f"dispatch_by_node_name references unknown node name '{nm}'")
            per[name_to_idx[nm]] = pol
        dispatch_pol = PerNodeDispatchPolicy(per_node=per)

    # due date + job class samplers
    job_classes_cfg = dict(cfg.get("job_classes", {}))
    # choose first class' due dist if present; or take a global due_date_dist in cfg
    # (You can refine later per class if you want.)
    due_dist = None
    if job_classes_cfg:
        first_cls = next(iter(job_classes_cfg.keys()))
        due_dist = job_classes_cfg[first_cls].get("due_date_dist", None)
    due_date = make_due_date_sampler(due_dist)
    job_class = make_job_class_sampler(arrival_cfg)

    # quality compilation: node-index based
    qcfg = dict(cfg.get("quality_model", {}))
    defect_prob_raw = dict(qcfg.get("defect_prob", {}))
    defect_prob: Dict[int, Dict[Any, float]] = {}
    assembly_nodes: List[int] = []

    for node_id_str, per_cls in defect_prob_raw.items():
        node_id = int(node_id_str)
        if node_id not in id_to_idx:
            raise ValueError(f"quality_model.defect_prob references unknown node id {node_id}")
        idx = id_to_idx[node_id]
        assembly_nodes.append(idx)
        defect_prob[idx] = dict(per_cls)

    # figure out sinks by name (best effort)
    fg_idx = name_to_idx.get("FinishedGoods", -1)
    scrap_idx = name_to_idx.get("Scrap", -1)
    rework_idx = name_to_idx.get("Rework", -1)

    quality_cfg = {
        "assembly_nodes": assembly_nodes,
        "rework_node": int(rework_idx),
        "fg_node": int(fg_idx),
        "scrap_node": int(scrap_idx),
        "defect_prob": defect_prob,
        "rework_success_prob": float(qcfg.get("rework_success_prob", 1.0)),
    }

    conf = QueueingNetworkConf(
        n_nodes=n,
        arrivals=arrivals,
        services=services,
        servers=servers,
        capacity=capacity,
        discipline=[None] * n,  # not used by sim/env right now; keep placeholder if you want. Note: I left discipline=[None]*n because your sim/env currently use dispatch_policy directly and ignore per-node discipline. If you prefer, change it to [FIFO() for _ in range(n)] and keep it for later.
        overflow_mode=str(overflow_mode),
        due_date=due_date,
        job_class=job_class,
        dispatch_policy=dispatch_pol,
        routing_policy=routing_policy,
    )

    aux = {
        "name_to_idx": name_to_idx,
        "id_to_idx": id_to_idx,
        "idx_to_name": idx_to_name,
        "arrival_node_idx": arrival_node_idx,
    }
    return conf, quality_cfg, aux