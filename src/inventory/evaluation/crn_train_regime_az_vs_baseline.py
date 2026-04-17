"""Replicate Lecture 17 regime-case learning (package version).

Goal: sanity-check that `inventory.policies.alphazero.Hybrid_AlphaZero` can pass the
strict-CRN evaluation gate and improve vs a fixed Order-Up-To baseline in the
*regime* demand setting (where adaptivity matters).

Run from repo root:
    poetry run python -m inventory.evaluation.crn_train_regime_az_vs_baseline

This mirrors the hyperparameters shown in `lectures/lecture_17_b_V0.1.ipynb`.
"""

from __future__ import annotations

import numpy as np
from inventory.policies.alphazero import Hybrid_AlphaZero
from inventory.policies.baselines import OrderUpToPolicy
from inventory.problems.demand_models import PoissonRegimeDemand
from inventory.problems.inventory_mvp import make_inventory_mvp_system


def main() -> None:
    # --- regime setup (matches Lecture 17 notebook) ---
    P = np.array([[0.92, 0.08], [0.12, 0.88]], dtype=float)
    exog = PoissonRegimeDemand(lam_by_regime=[220, 460], P=P, regime_index=1)

    system = make_inventory_mvp_system(exogenous_model=exog, d_s=2, sim_seed=42)

    S0 = np.array([300.0, 0.0], dtype=float)  # inv=300, start regime 0
    T = 30

    dx = 10
    x_max = 480

    # --- baseline: single target (cannot adapt to regime) ---
    baseline = OrderUpToPolicy(target_level=340, x_max=x_max, dx=dx)

    # --- Hybrid AlphaZero (PUCT-MCTS + tiny net) ---
    az = Hybrid_AlphaZero(
        system,
        x_max=x_max,
        dx=dx,
        H=10,
        n_sims=160,
        c_puct=1.6,
        seed=0,
        tau_eval=0.0,
    )

    # --- train ---
    history = az.fit_self_play(
        S0=S0,
        T=T,
        n_iterations=6,
        episodes_per_iter=10,
        buffer_max=5000,
        tau_train=1.0,
        lr=0.03,
        value_weight=0.5,
        train_steps=700,
        batch_size=64,
        gate_episodes=20,
        gate_seed0=2026,
        seed0=0,
        verbose=True,
        info={"T": T},
    )

    # --- evaluate under strict CRN ---
    results, _rollouts = system.evaluate_policies_crn_mc(
        policies={"PolicyOrderUpTo": baseline, "Hybrid_AlphaZero": az},
        S0=S0,
        T=T,
        n_episodes=120,
        seed0=1234,
        info={"T": T},
    )

    print("=== Final CRN-MC totals (lower is better) ===")
    for name, r in results.items():
        print(f"{name:15s} mean={r.mean:.3f}  std={r.std:.3f}")

    _ = history


if __name__ == "__main__":
    main()
