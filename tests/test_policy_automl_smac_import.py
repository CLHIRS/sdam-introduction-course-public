from __future__ import annotations

import numpy as np

from inventory.evaluation.policy_automl import (
    _require_smac3,
    make_policy_automl_demo_problem,
    run_policy_automl_smac3,
    smac3_available,
)


def test_smac3_available_handles_sklearn_dtype_compatibility() -> None:
    assert smac3_available() is True


def test_require_smac3_restores_missing_sklearn_dtype(monkeypatch) -> None:
    import sklearn.tree._tree as sklearn_tree

    monkeypatch.delattr(sklearn_tree, "DTYPE", raising=False)
    assert not hasattr(sklearn_tree, "DTYPE")

    _require_smac3()

    assert hasattr(sklearn_tree, "DTYPE")
    assert sklearn_tree.DTYPE is np.float32


def test_run_policy_automl_smac3_smoke() -> None:
    problem = make_policy_automl_demo_problem(
        lam=20.0,
        T=4,
        initial_inventory=20.0,
        x_max=60,
        dx=10,
        s_max=60,
        sim_seed=7,
    )

    result = run_policy_automl_smac3(
        problem,
        n_trials=2,
        n_episodes=3,
        seed0=123,
        scenario_seed=456,
        families=("order_up_to",),
        target_level_bounds=(0, 60),
    )

    assert result.n_trials_requested == 2
    assert result.n_trials_finished >= 1
    assert result.incumbent_label.startswith("OrderUpTo[")
    assert result.runhistory_rows
