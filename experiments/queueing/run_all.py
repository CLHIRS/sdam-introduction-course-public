# experiments/queueing/run_all.py
from __future__ import annotations

import argparse
import importlib
import os
from dataclasses import dataclass
from typing import Callable, List, Optional

from helpers.bootstrap import ensure_importable


# Ensure the installed package imports resolve (fallback: inject `src/`).
ensure_importable("scenarios")
ensure_importable("queueing")


@dataclass(frozen=True)
class Scenario:
    module_path: str
    name: str


SCENARIOS: List[Scenario] = [
    Scenario("scenarios.scenario_01_mm1", "Scenario 01 — M/M/1 basics"),
    Scenario("scenarios.scenario_02_parallel", "Scenario 02 — Routing to parallel servers"),
    Scenario("scenarios.scenario_03_dispatch_within_node", "Scenario 03 — Dispatch within node (FIFO vs EDD)"),
    Scenario("scenarios.scenario_04_rework_loop", "Scenario 04 — Rework loop / quality"),
    Scenario("scenarios.scenario_05_fast_vs_clean_rework", "Scenario 05 — Fast vs clean + rework"),
    Scenario("scenarios.scenario_06_final_ppo_ready_manufacturing", "Scenario 06 — Final PPO-ready manufacturing"),
]


def _call_if_exists(mod, fn: str, *args, **kwargs) -> bool:
    f: Optional[Callable] = getattr(mod, fn, None)
    if f is None:
        return False
    f(*args, **kwargs)
    return True


def _run_one(mod_path: str, *, des_seed: int, crn_seed0: int, run_env: bool) -> None:
    mod = importlib.import_module(mod_path)

    # Common simple entrypoint in scenarios 01/02/05
    if _call_if_exists(mod, "run_demo"):
        return

    # Paired DES + strict-CRN entrypoints in scenarios 03/04
    did_any = False
    did_any |= _call_if_exists(mod, "run_des_demo", seed=des_seed)
    did_any |= _call_if_exists(mod, "run_crn_demo", seed0=crn_seed0)
    if did_any:
        return

    # Scenario 06 style
    cfg = None
    if hasattr(mod, "build_cfg"):
        cfg = mod.build_cfg()
    if cfg is not None and hasattr(mod, "run_des_kpis"):
        mod.run_des_kpis(cfg)
        if run_env and hasattr(mod, "run_env_one_episode"):
            mod.run_env_one_episode(cfg, seed=des_seed)
        return

    # Fallback
    if _call_if_exists(mod, "main"):
        return

    raise RuntimeError(
        f"{mod_path} has no supported entrypoint. Expected one of: "
        "run_demo(), run_des_demo(seed=...)/run_crn_demo(seed0=...), "
        "build_cfg()+run_des_kpis(cfg) (+ optional run_env_one_episode), or main()."
    )


def run_all(*, des_seed: int = 0, crn_seed0: int = 0, only: Optional[List[str]] = None, skip: Optional[List[str]] = None, run_env: bool = True) -> None:
    print("\n=== Running all scenarios ===")
    print(f"DES seed={des_seed} | CRN seed0={crn_seed0}\n")

    only = list(only or [])
    skip = set(skip or [])

    def _selected(s: Scenario) -> bool:
        if s.module_path in skip or s.name in skip:
            return False
        if not only:
            return True
        return (s.module_path in only) or (s.name in only)

    selected = [s for s in SCENARIOS if _selected(s)]
    if not selected:
        raise RuntimeError("No scenarios selected. Check --only/--skip arguments.")

    for s in selected:
        print("\n" + "=" * 80)
        print(s.name)
        print("=" * 80)

        _run_one(s.module_path, des_seed=des_seed, crn_seed0=crn_seed0, run_env=run_env)

    print("\n=== Done ===")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run all queueing scenarios.")
    p.add_argument("--des-seed", type=int, default=0)
    p.add_argument("--crn-seed0", type=int, default=0)
    p.add_argument("--fast", action="store_true", help="Run reduced-workload scenarios (intended for smoke tests).")
    p.add_argument("--only", action="append", default=[], help="Run only a specific scenario (repeatable): module path or display name")
    p.add_argument("--skip", action="append", default=[], help="Skip a scenario (repeatable): module path or display name")
    p.add_argument("--no-env", action="store_true", help="Do not run env episode (scenario 06).")
    args = p.parse_args()

    if bool(args.fast):
        os.environ["SDAM_FAST"] = "1"

    run_all(des_seed=args.des_seed, crn_seed0=args.crn_seed0, only=args.only, skip=args.skip, run_env=(not args.no_env))