import numpy as np

from queueing.eval_crn import evaluate_policies_crn_mc_queue
from scenarios.scenario_01_mm1 import build_config


def test_queueing_strict_crn_determinism() -> None:
    cfg = build_config(lam=0.6, mu=1.0)

    # No decision points in this scenario; policies are never queried.
    policies = {"A": {}, "B": {}}

    kwargs = dict(
        policies=policies,
        baseline_name="A",
        T=200.0,
        n_rep=5,
        seed0=0,
        overflow_mode="block",
        include_env_reward=False,
        verbose=False,
    )

    rep1 = evaluate_policies_crn_mc_queue(cfg, **kwargs)
    rep2 = evaluate_policies_crn_mc_queue(cfg, **kwargs)

    for pn in rep1["policy_names"]:
        for mn in rep1["metrics"]:
            x1 = np.asarray(rep1["raw"][pn][mn], dtype=float)
            x2 = np.asarray(rep2["raw"][pn][mn], dtype=float)
            assert np.isfinite(x1).all()
            assert np.isfinite(x2).all()
            assert np.allclose(x1, x2, atol=1e-12, rtol=0.0)
