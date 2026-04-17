import io
from contextlib import redirect_stdout

import numpy as np

from inventory.evaluation import make_eval_info
from inventory.evaluation.notebooks.crn_runs import run_crn_mc
from inventory.policies.baselines import OrderUpToPolicy
from inventory.problems.demand_models import PoissonConstantDemand
from inventory.problems.inventory_mvp import make_inventory_mvp_system


def test_run_crn_mc_includes_runtime_in_summary_and_results() -> None:
    system = make_inventory_mvp_system(exogenous_model=PoissonConstantDemand(lam=5), sim_seed=0, d_s=1)
    policies = {"base": OrderUpToPolicy(target_level=10.0)}
    S0 = np.array([10.0], dtype=float)

    buf = io.StringIO()
    with redirect_stdout(buf):
        run = run_crn_mc(
            system=system,
            policies=policies,
            S0=S0,
            T=6,
            n_episodes=5,
            seed0=1234,
            info=make_eval_info(T=6),
            print_summary=True,
        )

    assert "runtime=" in buf.getvalue()
    assert float(run.results["base"].runtime_sec) >= 0.0
    assert float(run.totals_summary["base"]["runtime_sec"]) >= 0.0