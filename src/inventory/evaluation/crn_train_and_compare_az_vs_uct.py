"""Train-then-compare under strict CRN: Hybrid AlphaZero vs UCT MCTS.

This is a pragmatic sanity-check for Lecture 17/18 work-in-progress.

Run from repo root:
    poetry run python -m inventory.evaluation.crn_train_and_compare_az_vs_uct

Notes:
- Uses the strict-CRN evaluator in `inventory.core.dynamics.DynamicSystemMVP`.
- Runs a small AlphaZero-style self-play training loop with an evaluation gate.
- Then re-evaluates vs `DLA_MCTS_UCT` under the same CRN seed paths.

Regime-demand mode (Lecture 17 notebook-style learning):
    poetry run python -m inventory.evaluation.crn_train_and_compare_az_vs_uct --problem regime --opponent orderupto
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from inventory.policies.alphazero import Hybrid_AlphaZero
from inventory.policies.baselines import OrderUpToPolicy
from inventory.policies.dla_mcts import DLA_MCTS_UCT
from inventory.problems.demand_models import PoissonRegimeDemand, PoissonSeasonalDemand
from inventory.problems.inventory_mvp import make_inventory_mvp_system


def _crn_eval(system, policies, S0, *, T: int, n_episodes: int, seed0: int):
    results, _rollouts = system.evaluate_policies_crn_mc(
        policies,
        S0,
        T=T,
        n_episodes=n_episodes,
        seed0=seed0,
        info={"T": T},
    )
    return results


def _print_results(results, *, title: str) -> None:
    print(title)
    for name, res in results.items():
        print(f"- {name:18s} mean total cost = {res.mean:10.3f} | std = {res.std:9.3f}")


def _build_problem(problem: str):
    if problem == "seasonal":
        T = 30
        S0 = np.array([150.0], dtype=float)
        exo = PoissonSeasonalDemand(base=300.0, amp=90.0, period=20, spike_every=50, spike_add=120.0)
        system = make_inventory_mvp_system(exogenous_model=exo, d_s=1, sim_seed=42)
        return system, S0, T

    if problem == "regime":
        P = np.array([[0.92, 0.08], [0.12, 0.88]], dtype=float)
        exo = PoissonRegimeDemand(lam_by_regime=[220, 460], P=P, regime_index=1)
        system = make_inventory_mvp_system(exogenous_model=exo, d_s=2, sim_seed=42)
        S0 = np.array([300.0, 0.0], dtype=float)
        T = 30
        return system, S0, T

    raise ValueError(f"Unknown problem: {problem}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["seasonal", "regime"], default="seasonal")
    ap.add_argument("--opponent", choices=["uct", "orderupto"], default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--eval-seed0", type=int, default=20260209)
    ap.add_argument("--n-iterations", type=int, default=None)
    ap.add_argument("--episodes-per-iter", type=int, default=None)
    ap.add_argument("--train-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gate-episodes", type=int, default=None)
    ap.add_argument("--tau-train", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--n-sims", type=int, default=None)
    args = ap.parse_args()

    system, S0, T = _build_problem(args.problem)

    opponent = args.opponent
    if opponent is None:
        opponent = "uct" if args.problem == "seasonal" else "orderupto"

    # Defaults are chosen to match prior script behavior (seasonal) and to mirror
    # the Lecture 17 notebook-style learning run (regime).
    if args.problem == "seasonal":
        H = 6
        n_sims = int(args.n_sims) if args.n_sims is not None else 120
        n_eval_episodes = int(args.eval_episodes) if args.eval_episodes is not None else 200
        n_iterations = int(args.n_iterations) if args.n_iterations is not None else 4
        episodes_per_iter = int(args.episodes_per_iter) if args.episodes_per_iter is not None else 10
        train_steps = int(args.train_steps) if args.train_steps is not None else 450
        gate_episodes = int(args.gate_episodes) if args.gate_episodes is not None else 25
        tau_train = float(args.tau_train) if args.tau_train is not None else 1.0
        lr = float(args.lr) if args.lr is not None else 0.02
        az_seed = 123
        c_puct = 1.5
    else:
        H = 10
        n_sims = int(args.n_sims) if args.n_sims is not None else 160
        n_eval_episodes = int(args.eval_episodes) if args.eval_episodes is not None else 120
        n_iterations = int(args.n_iterations) if args.n_iterations is not None else 6
        episodes_per_iter = int(args.episodes_per_iter) if args.episodes_per_iter is not None else 10
        train_steps = int(args.train_steps) if args.train_steps is not None else 700
        gate_episodes = int(args.gate_episodes) if args.gate_episodes is not None else 20
        tau_train = float(args.tau_train) if args.tau_train is not None else 1.0
        lr = float(args.lr) if args.lr is not None else 0.03
        az_seed = 0
        c_puct = 1.6

    eval_seed0 = int(args.eval_seed0)
    batch_size = int(args.batch_size)

    # --- policies ---
    az = Hybrid_AlphaZero(
        system,
        x_max=480,
        dx=10,
        H=H,
        n_sims=n_sims,
        c_puct=c_puct,
        gamma=1.0,
        seed=az_seed,
        tau_eval=0.0,
    )

    if opponent == "uct":
        opp = DLA_MCTS_UCT(
            system,
            horizon=H,
            n_simulations=n_sims,
            uct_c=1.4,
            x_max=480,
            dx=10,
            planning_seed=123,
        )
        policies = {"Hybrid_AlphaZero": az, "DLA_MCTS_UCT": opp}
    else:
        opp = OrderUpToPolicy(target_level=340, x_max=480, dx=10)
        policies = {"PolicyOrderUpTo": opp, "Hybrid_AlphaZero": az}

    # --- pre-training evaluation ---
    pre = _crn_eval(
        system,
        policies,
        S0,
        T=T,
        n_episodes=n_eval_episodes,
        seed0=eval_seed0,
    )
    _print_results(
        pre,
        title=(
            f"PRE-TRAIN | CRN MC | problem={args.problem} | T={T} | episodes={n_eval_episodes} | sims={n_sims}"
        ),
    )

    # --- train AlphaZero ---
    print(
        "\nTraining Hybrid_AlphaZero via fit_self_play() "
        f"(problem={args.problem}, iters={n_iterations}, eps/iter={episodes_per_iter}, "
        f"train_steps={train_steps}, gate_eps={gate_episodes}, lr={lr})"
    )
    t0 = time.time()
    _hist = az.fit_self_play(
        S0=S0,
        T=T,
        n_iterations=n_iterations,
        episodes_per_iter=episodes_per_iter,
        buffer_max=8000,
        tau_train=tau_train,
        lr=lr,
        value_weight=0.5,
        train_steps=train_steps,
        batch_size=batch_size,
        gate_episodes=gate_episodes,
        gate_seed0=2026,
        seed0=0,
        verbose=True,
        info={"T": T},
    )
    t1 = time.time()
    print(f"Training wall time: {t1 - t0:.2f}s")

    # --- post-training evaluation ---
    post = _crn_eval(
        system,
        policies,
        S0,
        T=T,
        n_episodes=n_eval_episodes,
        seed0=eval_seed0,
    )
    _print_results(
        post,
        title=(
            f"\nPOST-TRAIN | CRN MC | problem={args.problem} | T={T} | episodes={n_eval_episodes} | sims={n_sims}"
        ),
    )

    # Paired diffs (post) when exactly two policies are present
    names = list(policies.keys())
    if len(names) == 2:
        a, b = names[0], names[1]
        d = post[a].totals - post[b].totals
        print(f"\nPOST-TRAIN paired diff ({a} - {b}):")
        print(f"- mean diff = {float(d.mean()):.3f} (negative => {a} better)")
        print(f"- std  diff = {float(d.std(ddof=1)):.3f}")
        print(f"- win rate  = {float((d < 0).mean()):.3f}")


if __name__ == "__main__":
    main()
