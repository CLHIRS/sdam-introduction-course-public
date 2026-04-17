import numpy as np

from inventory.policies.baselines import OrderUpToPolicy
from inventory.problems.demand_models import PoissonConstantDemand
from inventory.problems.inventory_mvp import make_inventory_mvp_system


def test_inventory_strict_crn_determinism() -> None:
    exo = PoissonConstantDemand(lam=5)
    system = make_inventory_mvp_system(exogenous_model=exo, sim_seed=0, d_s=1)

    policies = {"base": OrderUpToPolicy(target_level=10.0)}
    S0 = np.array([10.0], dtype=float)

    kwargs = dict(T=10, n_episodes=8, seed0=1234, info={"deterministic": True})

    r1, _ = system.evaluate_policies_crn_mc(policies, S0, **kwargs)
    r2, _ = system.evaluate_policies_crn_mc(policies, S0, **kwargs)

    t1 = r1["base"].totals
    t2 = r2["base"].totals

    assert np.isfinite(t1).all()
    assert np.isfinite(t2).all()
    assert float(r1["base"].runtime_sec) >= 0.0
    assert float(r2["base"].runtime_sec) >= 0.0

    assert np.allclose(t1, t2, atol=1e-12, rtol=0.0)
