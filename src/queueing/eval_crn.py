# queueing/eval_crn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from queueing.config_compile import compile_queue_network_config
from queueing.core_types import QueueingNetworkKPIs
from queueing.dispatch import DispatchPolicy
from queueing.rl_env import RoutingObsSpec, RoutingOnlyEnv
from queueing.routing_policies import RouteDecisionPolicy
from queueing.sim import QueueingNetworkSim

# ----------------------------
# Style notes:
# ----------------------------
"""
	•	old style: {"name": {"dp_id": RouteDecisionPolicy, ...}} (routing-only), or
	•	new style: {"name": PolicySpec(...)}
"""
# ----------------------------
# Metric specification
# ----------------------------

@dataclass(frozen=True)
class MetricSpec:
    name: str
    fn: Callable[[QueueingNetworkKPIs], float]
    higher_is_better: bool


DEFAULT_KPI_METRICS: List[MetricSpec] = [
    MetricSpec("throughput", lambda k: float(k.throughput), True),
    MetricSpec("mean_cycle_time", lambda k: float(k.mean_cycle_time), False),
    MetricSpec("mean_waiting_time", lambda k: float(k.mean_waiting_time), False),
    MetricSpec("mean_tardiness", lambda k: float(k.mean_tardiness), False),
    MetricSpec("on_time_delivery", lambda k: float(k.on_time_delivery), True),
    MetricSpec("wip_time_avg", lambda k: float(k.wip_time_avg), False),
    MetricSpec("fill_rate", lambda k: float(k.fill_rate), True),
]


# ----------------------------
# PolicySpec: routing + dispatch
# ----------------------------

@dataclass(frozen=True)
class PolicySpec:
    """
    Full policy specification for a queueing network experiment:
      - routing_dp_policies: mapping dp_id -> RouteDecisionPolicy
      - dispatch_by_node_name: mapping node_name -> DispatchPolicy (optional)
    """
    routing_dp_policies: Dict[str, RouteDecisionPolicy]
    dispatch_by_node_name: Optional[Dict[str, DispatchPolicy]] = None


# Allow old-style policies dict where value is just routing dp policies
PoliciesInput = Dict[str, Union[PolicySpec, Dict[str, RouteDecisionPolicy]]]


def _normalize_policies(policies: PoliciesInput) -> Dict[str, PolicySpec]:
    out: Dict[str, PolicySpec] = {}
    for name, spec in policies.items():
        if isinstance(spec, PolicySpec):
            out[name] = spec
        elif isinstance(spec, dict):
            # old style: routing only
            out[name] = PolicySpec(routing_dp_policies=spec, dispatch_by_node_name=None)
        else:
            raise TypeError(f"Policy '{name}' has unsupported type: {type(spec)}")
    return out


def _wins(delta: np.ndarray, higher_is_better: bool) -> float:
    # delta = candidate - baseline
    if higher_is_better:
        return float(np.mean(delta > 0))
    else:
        return float(np.mean(delta < 0))


def _summary(x: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
    }


# ----------------------------
# RoutingOnlyEnv episode helper
# ----------------------------

def _env_episode_total_reward(
    cfg: Dict[str, Any],
    *,
    pol: PolicySpec,
    T: float,
    seed: int,
    overflow_mode: str,
    obs_spec: RoutingObsSpec,
    max_events: int,
) -> float:
    conf, quality_cfg, _aux = compile_queue_network_config(
        cfg,
        routing_dp_policies=pol.routing_dp_policies,
        dispatch_by_node_name=pol.dispatch_by_node_name,
        overflow_mode=overflow_mode,
    )

    env = RoutingOnlyEnv(
        conf,
        cfg=cfg,
        quality_cfg=quality_cfg,
        obs_spec=obs_spec,
        seed=int(seed),
        T=float(T),
        max_events=int(max_events),
    )

    ctx, r, done, _info = env.reset_and_get_first()
    total = float(r)

    while not done:
        # choose action using the dp policy for this dp_id
        dp_pol = conf.routing_policy.dp_policy[ctx.dp_id]  # type: ignore[union-attr]
        state = env._snapshot()  # internal snapshot; OK for now

        # IMPORTANT: if your dp_pol uses randomness, the env itself already ensures exogenous CRN.
        # For purely random policies, prefer deterministic rules or implement keyed randomness inside the policy.
        a = int(
            dp_pol.select(
                rng=np.random.default_rng(0),
                dp_id=ctx.dp_id,
                job=env.jobs[ctx.job_id],
                from_node=ctx.from_node,
                actions=ctx.actions,
                state=state,
            )
        )

        env.apply_action(ctx, a)
        ctx, r, done, _info = env.step_until_decision()
        total += float(r)

    return float(total)


# ----------------------------
# Strict-CRN evaluation
# ----------------------------

def evaluate_policies_crn_mc_queue(
    cfg: Dict[str, Any],
    *,
    policies: PoliciesInput,
    baseline_name: str,
    T: float,
    n_rep: int,
    seed0: int = 0,
    overflow_mode: str = "block",
    kpi_metrics: Optional[List[MetricSpec]] = None,
    include_env_reward: bool = True,
    env_reward_name: str = "env_total_reward",
    env_T: Optional[float] = None,
    env_obs_spec: Optional[RoutingObsSpec] = None,
    max_events: int = 2_000_000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Strict-CRN MC evaluation for queueing networks that supports BOTH routing and dispatch variation.

    policies accepts:
      - new style: {"name": PolicySpec(...)}
      - old style: {"name": {dp_id: RouteDecisionPolicy, ...}}  (routing only)
    """
    pols = _normalize_policies(policies)

    if baseline_name not in pols:
        raise ValueError(f"baseline_name='{baseline_name}' not in policies keys={list(pols.keys())}")
    if n_rep <= 0:
        raise ValueError("n_rep must be > 0")

    if kpi_metrics is None:
        kpi_metrics = DEFAULT_KPI_METRICS

    policy_names = list(pols.keys())
    metric_names: List[str] = [m.name for m in kpi_metrics]
    if include_env_reward:
        metric_names.append(env_reward_name)

    # default obs spec (only needed for env reward)
    if include_env_reward:
        if env_T is None:
            env_T = float(T)
        if env_obs_spec is None:
            n_nodes = len(cfg["nodes"])
            classes = list(cfg.get("job_classes", {}).keys()) or ["A"]
            env_obs_spec = RoutingObsSpec(node_ids=list(range(n_nodes)), class_labels=classes)

    # results[policy][metric] -> np.ndarray
    results: Dict[str, Dict[str, np.ndarray]] = {
        pn: {mn: np.zeros(n_rep, dtype=float) for mn in metric_names} for pn in policy_names
    }

    for r in range(n_rep):
        ep_seed = int(seed0 + r)
        if verbose and (r % max(1, n_rep // 10) == 0):
            print(f"[CRN] replication {r+1}/{n_rep} seed={ep_seed}")

        for pn in policy_names:
            pol = pols[pn]

            # --- DES KPIs ---
            conf, quality_cfg, _aux = compile_queue_network_config(
                cfg,
                routing_dp_policies=pol.routing_dp_policies,
                dispatch_by_node_name=pol.dispatch_by_node_name,
                overflow_mode=overflow_mode,
            )
            sim = QueueingNetworkSim(conf, quality_cfg=quality_cfg)
            kpis = sim.run(T=float(T), seed=ep_seed, max_events=int(max_events))

            for m in kpi_metrics:
                results[pn][m.name][r] = float(m.fn(kpis))

            # --- Env objective (PPO scalar) ---
            if include_env_reward:
                rew = _env_episode_total_reward(
                    cfg,
                    pol=pol,
                    T=float(env_T),
                    seed=ep_seed,
                    overflow_mode=overflow_mode,
                    obs_spec=env_obs_spec,  # type: ignore[arg-type]
                    max_events=int(max_events),
                )
                results[pn][env_reward_name][r] = float(rew)

    baseline = results[baseline_name]

    report: Dict[str, Any] = {
        "baseline_name": baseline_name,
        "policy_names": policy_names,
        "n_rep": n_rep,
        "seed0": seed0,
        "T": float(T),
        "overflow_mode": overflow_mode,
        "metrics": metric_names,
        "kpi_metrics": [m.name for m in kpi_metrics],
        "include_env_reward": include_env_reward,
        "env_reward_name": env_reward_name,
        "raw": results,
        "summary": {},
        "paired": {},
        "higher_is_better": {m.name: bool(m.higher_is_better) for m in kpi_metrics},
    }

    if include_env_reward:
        report["higher_is_better"][env_reward_name] = True  # reward is higher-is-better

    # summary
    for pn in policy_names:
        report["summary"][pn] = {}
        for mn in metric_names:
            report["summary"][pn][mn] = _summary(results[pn][mn])

    # paired deltas
    for pn in policy_names:
        if pn == baseline_name:
            continue
        report["paired"][pn] = {}
        for mn in metric_names:
            cand = results[pn][mn]
            base = baseline[mn]
            delta = cand - base
            report["paired"][pn][mn] = {
                "delta_mean": float(np.mean(delta)),
                "delta_std": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
                "win_rate": _wins(delta, report["higher_is_better"][mn]),
            }

    return report


def print_strict_crn_report_queue(report: Dict[str, Any], *, metrics: Optional[List[str]] = None) -> None:
    baseline = report["baseline_name"]
    names = report["policy_names"]
    if metrics is None:
        metrics = report["metrics"]

    hib = report["higher_is_better"]

    print("\n## Strict-CRN Monte Carlo policy evaluation (Queueing Networks)")
    print(f"n_rep={report['n_rep']}  seed0={report['seed0']}  T={report['T']:.1f}  overflow={report['overflow_mode']}")
    print(f"baseline='{baseline}'")
    print()

    for m in metrics:
        dir_txt = "higher=better" if hib[m] else "lower=better"
        print(f"### Metric: {m} ({dir_txt})")
        for pn in names:
            s = report["summary"][pn][m]
            print(f"  - {pn:>18s}: mean={s['mean']:.6f}  std={s['std']:.6f}")
        for pn in names:
            if pn == baseline:
                continue
            p = report["paired"][pn][m]
            print(
                f"    Δ({pn}-{baseline}): mean={p['delta_mean']:.6f}  std={p['delta_std']:.6f}  win_rate={p['win_rate']:.3f}"
            )
        print()