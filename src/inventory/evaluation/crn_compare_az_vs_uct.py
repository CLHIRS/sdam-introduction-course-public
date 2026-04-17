"""Strict-CRN sanity-check comparison.

Defaults to the original seasonal-demand comparison (Hybrid AlphaZero vs UCT).
Also supports the Lecture 17 regime-demand case (where the notebook demonstrates
gate ACCEPTs and learning).

Run from repo root:
    poetry run python -m inventory.evaluation.crn_compare_az_vs_uct
    poetry run python -m inventory.evaluation.crn_compare_az_vs_uct --problem regime --opponent orderupto

Uses the strict-CRN evaluator in `inventory.core.dynamics.DynamicSystemMVP`.
"""

from __future__ import annotations

import argparse

import numpy as np
from inventory.policies.alphazero import Hybrid_AlphaZero
from inventory.policies.baselines import OrderUpToPolicy
from inventory.policies.dla_mcts import DLA_MCTS_UCT
from inventory.problems.demand_models import PoissonRegimeDemand, PoissonSeasonalDemand
from inventory.problems.inventory_mvp import make_inventory_mvp_system


def _build_problem(problem: str):
    if problem == "seasonal":
        T = 30
        S0 = np.array([150.0], dtype=float)
        exo = PoissonSeasonalDemand(base=300.0, amp=90.0, period=20, spike_every=50, spike_add=120.0)
        system = make_inventory_mvp_system(exogenous_model=exo, d_s=1, sim_seed=42)
        return system, S0, T

    if problem == "regime":
        # Matches the Lecture 17 notebook regime example (observable regime in state).
        P = np.array([[0.92, 0.08], [0.12, 0.88]], dtype=float)
        exo = PoissonRegimeDemand(lam_by_regime=[220, 460], P=P, regime_index=1)
        system = make_inventory_mvp_system(exogenous_model=exo, d_s=2, sim_seed=42)
        S0 = np.array([300.0, 0.0], dtype=float)
        T = 30
        return system, S0, T

    raise ValueError(f"Unknown problem: {problem}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--problem",
        choices=["seasonal", "regime"],
        default="seasonal",
        help="Which demand model to use.",
    )
    ap.add_argument(
        "--opponent",
        choices=["uct", "orderupto"],
        default=None,
        help="What to compare against. Default depends on problem.",
    )
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=20260209)
    ap.add_argument("--n-sims", type=int, default=None, help="Simulations per decision for both planners.")
    args = ap.parse_args()

    system, S0, T = _build_problem(args.problem)
    n_episodes = int(args.episodes)
    seed0 = int(args.seed0)

    opponent = args.opponent
    if opponent is None:
        opponent = "uct" if args.problem == "seasonal" else "orderupto"

    # --- policies ---
    if args.problem == "seasonal":
        H = 6
        n_sims = int(args.n_sims) if args.n_sims is not None else 180
        pol_az = Hybrid_AlphaZero(
            system,
            x_max=480,
            dx=10,
            H=H,
            n_sims=n_sims,
            c_puct=1.5,
            gamma=1.0,
            seed=123,
            tau_eval=0.0,
        )
    else:
        # Slightly larger horizon in the regime case (matches notebook-style runs).
        H = 10
        n_sims = int(args.n_sims) if args.n_sims is not None else 160
        pol_az = Hybrid_AlphaZero(
            system,
            x_max=480,
            dx=10,
            H=H,
            n_sims=n_sims,
            c_puct=1.6,
            gamma=1.0,
            seed=0,
            tau_eval=0.0,
        )

    policies = {"Hybrid_AlphaZero": pol_az}

    if opponent == "uct":
        pol_opp = DLA_MCTS_UCT(
            system,
            horizon=H,
            n_simulations=n_sims,
            uct_c=1.4,
            x_max=480,
            dx=10,
            planning_seed=123,
        )
        policies["DLA_MCTS_UCT"] = pol_opp
    else:
        # A fixed target level baseline (cannot adapt to regime unless you add regime-conditioned logic).
        policies["PolicyOrderUpTo"] = OrderUpToPolicy(target_level=340, x_max=480, dx=10)

    results, rollouts = system.evaluate_policies_crn_mc(
        policies,
        S0,
        T=T,
        n_episodes=n_episodes,
        seed0=seed0,
        info={"T": T},
    )

    print(f"CRN MC comparison | problem={args.problem} | T={T} | episodes={n_episodes} | sims={n_sims}")
    for name, res in results.items():
        print(f"- {name:14s} mean total cost = {res.mean:10.3f} | std = {res.std:9.3f}")

    names = list(policies.keys())
    if len(names) == 2:
        a, b = names[0], names[1]
        d = results[a].totals - results[b].totals
        print(f"\nPaired diff ({a} - {b}):")
        print(f"- mean diff = {float(d.mean()):.3f} (negative => {a} better)")
        print(f"- std  diff = {float(d.std(ddof=1)):.3f}")
        print(f"- win rate  = {float((d < 0).mean()):.3f}")

    for nm in names:
        tot = float(rollouts[nm]["costs"].sum())
        print(f"Example episode total cost ({nm}): {tot:.3f}")


if __name__ == "__main__":
    main()
