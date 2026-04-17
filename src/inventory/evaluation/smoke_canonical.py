from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class SmokeCaseResult:
    name: str
    ok: bool
    seconds: float
    error: Optional[str] = None


def _run_case(name: str, fn: Callable[[], None]) -> SmokeCaseResult:
    t0 = time.time()
    try:
        fn()
    except Exception as e:  # pragma: no cover
        dt = time.time() - t0
        return SmokeCaseResult(name=name, ok=False, seconds=dt, error=f"{type(e).__name__}: {e}")
    dt = time.time() - t0
    return SmokeCaseResult(name=name, ok=True, seconds=dt)


def _assert_close(a: np.ndarray, b: np.ndarray, *, atol: float = 0.0, rtol: float = 0.0, msg: str = "") -> None:
    a = np.asarray(a)
    b = np.asarray(b)
    if not np.allclose(a, b, atol=float(atol), rtol=float(rtol)):
        raise AssertionError(msg or f"Arrays differ: max|a-b|={float(np.max(np.abs(a-b)))}")


def _case_core_foundations() -> None:
    """Lecture 01_d + Lecture 03_b: simulate, strict-CRN reproducibility, optional capacity cap."""

    from inventory.core.dynamics import DynamicSystemMVP
    from inventory.policies.baselines import OrderUpToPolicy
    from inventory.problems.demand_models import PoissonSeasonalDemand
    from inventory.problems.inventory_mvp import (
        inventory_cost,
        inventory_cost_extended,
        inventory_transition,
        make_inventory_mvp_system,
    )

    exo = PoissonSeasonalDemand(base=300.0, amp=90.0, period=20, spike_every=50, spike_add=120.0)

    # Notebook-style dimension aliases
    sys1 = DynamicSystemMVP(
        transition_func=inventory_transition,
        cost_func=inventory_cost,
        exogenous_model=exo,
        sim_seed=42,
        dS=1,
        dX=1,
        dW=1,
    )

    pol = OrderUpToPolicy(target_level=350, x_max=480, dx=10)
    S0 = np.array([250.0], dtype=float)

    traj, costs, actions, step_seeds = sys1.simulate(pol, S0, T=8, seed=1)
    assert traj.shape == (9, 1)
    assert costs.shape == (8,)
    assert actions.shape == (8, 1)
    assert step_seeds.shape == (8,)

    # strict-CRN: simulate_crn called twice yields identical rollout
    traj2, costs2, actions2, step_seeds2 = sys1.simulate_crn(pol, S0, step_seeds)
    _assert_close(step_seeds, step_seeds2, msg="step seeds changed")
    _assert_close(traj, traj2, msg="CRN traj mismatch")
    _assert_close(costs, costs2, msg="CRN costs mismatch")
    _assert_close(actions, actions2, msg="CRN actions mismatch")

    # capacity-capped transition via factory
    sys_cap = make_inventory_mvp_system(exogenous_model=exo, d_s=1, s_max=480, sim_seed=42)
    traj3, costs3, actions3, _ = sys_cap.simulate(pol, np.array([470.0], dtype=float), T=8, seed=1)
    assert (traj3[:, 0] <= 480.0 + 1e-9).all(), "capacity cap violated"
    assert costs3.shape == (8,)
    assert actions3.shape == (8, 1)

    # Evaluation-mode convention: CRN MC evaluation defaults to deterministic=True
    class _ToyStochasticPolicy:
        def act(self, state: np.ndarray, t: int, info: Optional[dict] = None) -> np.ndarray:
            info = info or {}
            if bool(info.get("deterministic", False)):
                return np.array([0.0], dtype=float)
            seed = int(info.get("crn_step_seed", 0))
            rng = np.random.default_rng(seed)
            return np.array([float(10 * int(rng.integers(0, 2)))], dtype=float)

    toy = _ToyStochasticPolicy()
    totals_default = sys1.evaluate_policy_crn_mc(toy, S0, T=8, n_episodes=30, seed0=7)
    totals_explicit = sys1.evaluate_policy_crn_mc(toy, S0, T=8, n_episodes=30, seed0=7, info={"deterministic": True})
    _assert_close(totals_default, totals_explicit, msg="evaluate_policy_crn_mc default deterministic mismatch")

    # Extended cost function (setup cost) sanity
    s = np.array([10.0], dtype=float)
    w = np.array([3.0], dtype=float)
    c0 = inventory_cost_extended(s, np.array([0.0], dtype=float), w, 0)
    c1 = inventory_cost_extended(s, np.array([1.0], dtype=float), w, 0)
    assert float(c1) > float(c0), "setup cost should increase cost when ordering"


def _case_lecture01_bayesian_eval() -> None:
    """Lecture 01_d: Bayesian evaluation helpers (CRN + vanilla + normal) import/run."""

    from inventory.evaluation import (
        bayesian_AB_test_crn,
        bayesian_ab_test_normal,
        bayesian_ab_test_vanilla,
        print_bayesian_ab_report_normal,
        print_bayesian_ab_report_vanilla,
        print_bayesian_report_AB_crn,
    )

    # Just verify they run with tiny settings (keep this very fast).
    rng = np.random.default_rng(0)
    A = rng.normal(loc=1000.0, scale=50.0, size=30)
    B = rng.normal(loc=995.0, scale=50.0, size=30)
    deltas = B - A

    assert callable(print_bayesian_report_AB_crn)
    assert callable(print_bayesian_ab_report_vanilla)
    assert callable(print_bayesian_ab_report_normal)

    res_crn = bayesian_AB_test_crn(
        deltas,
        totals_A=A,
        totals_B=B,
        n_draws=2_000,
        cred_level=0.9,
        rope=5.0,
        random_state=0,
    )
    assert "posterior" in res_crn and "data_summary" in res_crn
    assert "mean_delta_(E[B-A])" in res_crn["posterior"]

    res_van = bayesian_ab_test_vanilla(
        A,
        B,
        n_draws=2_000,
        n_draws_superiority=200,
        m_pairs=200,
        random_state=0,
    )
    assert "posterior" in res_van and "posterior_totals" in res_van

    res_norm = bayesian_ab_test_normal(
        A,
        B,
        n_draws=2_000,
        random_state=0,
        ratio_and_uplift=False,
    )
    assert "posterior" in res_norm and "posterior_episode_superiority_predictive" in res_norm


def _case_lecture01_frequentist_eval() -> None:
    """Lecture 01_d: Frequentist evaluation helpers (paired deltas) import/run."""

    from inventory.evaluation import (
        format_frequentist_report_AB,
        frequentist_analysis_AB,
        print_frequentist_report_AB,
    )

    rng = np.random.default_rng(0)
    A = rng.normal(loc=1000.0, scale=50.0, size=30)
    B = rng.normal(loc=995.0, scale=50.0, size=30)
    deltas = B - A

    assert callable(print_frequentist_report_AB)

    try:
        res = frequentist_analysis_AB(
            deltas,
            objective="cost",
            delta_convention="treatment-control",
            n_perm=400,
            n_boot=400,
            random_state=0,
        )
    except ImportError:
        # If scipy isn't installed in a minimal environment, treat as non-blocking for canonical imports.
        return

    assert set(res.keys()) >= {"setup", "descriptives", "tests", "bootstrap", "recommendation"}

    txt = format_frequentist_report_AB(res)
    assert isinstance(txt, str) and "Frequentist" in txt

    # Ensure the printing wrapper doesn't crash, but keep smoke output clean.
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        _ = print_frequentist_report_AB(
            deltas,
            objective="cost",
            delta_convention="treatment-control",
            n_perm=50,
            n_boot=50,
            random_state=0,
        )


def _case_lecture01_delta_helpers() -> None:
    """Lecture 01_d: CRN delta helpers + normality diagnostics import/run."""

    from inventory.evaluation import (
        crn_delta_totals_from_totals_by_policy,
        normality_diagnostics,
        normality_diagnostics_for_deltas,
        print_totals_summary,
    )
    from inventory.evaluation.plotting import plot_totals_hist_and_box

    rng = np.random.default_rng(0)
    A = rng.normal(loc=1000.0, scale=50.0, size=25)
    B = rng.normal(loc=995.0, scale=50.0, size=25)
    totals_by_policy = {"A": A, "B": B}

    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        summ = print_totals_summary(totals_by_policy)
    assert set(summ.keys()) == {"A", "B"}
    assert set(summ["A"].keys()) >= {"n", "mean", "std", "min", "max"}

    d = crn_delta_totals_from_totals_by_policy(totals_by_policy, "A", "B", plot=False)
    _assert_close(d, B - A, msg="CRN delta totals mismatch")

    # plotting helper (skip if matplotlib missing)
    try:
        fig, _axes = plot_totals_hist_and_box(totals_by_policy, bins=10, show=False)
        try:
            fig.canvas.draw_idle()
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass
    except ImportError:
        pass

    # Pure numpy/matplotlib (may be missing matplotlib in minimal env)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = normality_diagnostics_for_deltas(d, alpha=0.10)
        assert set(out.keys()) >= {"n", "mean", "std", "jb", "jb_p"}
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:
            pass
    except ImportError:
        pass

    # SciPy + matplotlib diagnostic (skip if deps missing)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out2 = normality_diagnostics(d, alpha=0.10, verbose=False)
        assert set(out2.keys()) >= {"shapiro_p", "k2_p", "fig"}
        try:
            fig = out2.get("fig")
            if fig is not None:
                fig.canvas.draw_idle()
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:
            pass
    except ImportError:
        pass


def _case_lecture03_dp() -> None:
    """Lecture 03_b: DP solve + derived policy act() shape sanity (kept tiny)."""

    from inventory.policies.dp import DynamicProgrammingSolver1D
    from inventory.problems.demand_models import PoissonSeasonalDemand

    exo = PoissonSeasonalDemand(base=35.0, amp=10.0, period=10, spike_every=0, spike_add=0.0)
    solver = DynamicProgrammingSolver1D(
        exo,
        T=4,
        S_max=60,
        x_max=40,
        dx=10,
        p=2.0,
        c=0.5,
        h=0.03,
        b=1.0,
        terminal_cost_per_unit=0.03,
        expectation_mode="scenarios",
        K=5,
    )
    sol = solver.solve()
    pol = solver.to_policy(sol)

    a = pol.act(np.array([25.0], dtype=float), 0)
    assert np.asarray(a).shape == (1,)


def _case_lecture12_leadtime_problem() -> None:
    """Lecture 12_b: lead-time transition/cost helpers import + basic behavior."""

    from inventory.problems.inventory_leadtime import inventory_cost_leadtime, inventory_transition_leadtime

    L = 2
    s = np.array([10.0, 3.0, 4.0], dtype=float)  # I + pipeline of length L
    a = np.array([5.0], dtype=float)
    w = np.array([12.0], dtype=float)

    s2 = inventory_transition_leadtime(s, a, w, 0, L=L)
    assert np.asarray(s2).shape == (1 + L,)
    c = inventory_cost_leadtime(s, a, w, 0, L=L)
    assert np.isfinite(float(c))


def _case_problem_backorders() -> None:
    """Backorders problem module: import + tiny simulate sanity."""

    from inventory.policies.baselines import OrderUpToPolicy
    from inventory.problems.inventory_backorders import ExogenousPoissonDemand, make_inventory_backorders_system

    exo = ExogenousPoissonDemand(lam=5.0)
    system = make_inventory_backorders_system(exogenous_model=exo, sim_seed=0)
    pol = OrderUpToPolicy(target_level=5.0, x_max=20, dx=1)

    S0 = np.array([0.0], dtype=float)
    traj, costs, actions, _seeds = system.simulate(pol, S0, T=3, seed=1)
    assert traj.shape == (4, 1)
    assert costs.shape == (3,)
    assert actions.shape == (3, 1)


def _case_lecture11_pfas() -> None:
    from inventory.policies.pfa import (
        OrderUpTo_blackbox_PFA,
        OrderUpTo_regime_table_PFA,
        OrderUpTo_state_dependent_PFA_2,
        OrderUpToBlackboxPFA,
        OrderUpToRegimeTablePFA,
        OrderUpToStateDependentPFA,
    )
    from inventory.problems.demand_models import PoissonRegimeDemand, PoissonSeasonalDemand
    from inventory.problems.inventory_mvp import make_inventory_mvp_system

    # Notebook-compatibility alias checks (Lecture 11_b naming)
    assert OrderUpTo_blackbox_PFA is OrderUpToBlackboxPFA
    assert OrderUpTo_regime_table_PFA is OrderUpToRegimeTablePFA
    assert OrderUpTo_state_dependent_PFA_2 is OrderUpToStateDependentPFA

    class _DummyRegressor:
        def fit(self, X: np.ndarray, y: np.ndarray):
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            X = np.asarray(X)
            return np.zeros((X.shape[0],), dtype=float)

    exo = PoissonSeasonalDemand(base=300.0, amp=90.0, period=20)
    system = make_inventory_mvp_system(exogenous_model=exo, d_s=1, sim_seed=42)

    P = np.array([[0.95, 0.05], [0.10, 0.90]], dtype=float)
    exo_reg = PoissonRegimeDemand(lam_by_regime=[250, 450], P=P, regime_index=1)
    system_reg = make_inventory_mvp_system(exogenous_model=exo_reg, d_s=2, sim_seed=42)

    # Basic call/shape sanity; these PFAs are deterministic.
    s = np.array([300.0], dtype=float)
    s_reg = np.array([300.0, 1.0], dtype=float)

    p1 = OrderUpToBlackboxPFA(system=system, theta=np.array([350.0], dtype=float), x_max=480, dx=10)
    a1 = p1.act(s, 0)
    assert np.asarray(a1).shape == (1,)

    p2 = OrderUpToStateDependentPFA(system=system, x_max=480, dx=10, rounding="round", regressor=_DummyRegressor())
    a2 = p2.act(s, 0)
    assert np.asarray(a2).shape == (1,)

    p3 = OrderUpToRegimeTablePFA(system=system_reg, theta=np.array([330.0, 380.0], dtype=float), x_max=480, dx=10)
    a3 = p3.act(s_reg, 0)
    assert np.asarray(a3).shape == (1,)


def _case_lecture12_cfas_and_forecasters() -> None:
    from inventory.forecasters.factory import FitConfig, fit_ml_forecaster_from_exogenous, forecast_with_default_state
    from inventory.forecasters.ml import (
        MlAr1RegimeDemandForecaster,
        MlDemandForecaster,
        MlRegimeDemandForecaster,
        MultiRegimeAr1FeatureAdapter,
        MultiRegimeFeatureAdapter,
        RegimeFeatureAdapter,
        SeasonalFeatureAdapter,
    )
    from inventory.forecasters.naive import ExpertDemandForecasterConstant350
    from inventory.policies.cfa_milp import Order_MILP_CFA, OrderMilpCfaPolicy
    from inventory.problems.demand_models import PoissonRegimeDemand, PoissonSeasonalDemand

    # Notebook-compatibility alias check (Lecture 12 naming)
    assert Order_MILP_CFA is OrderMilpCfaPolicy

    exo = PoissonSeasonalDemand(base=300.0, amp=90.0, period=20, spike_every=50, spike_add=120.0)

    adapter = SeasonalFeatureAdapter(exo)
    ml = MlDemandForecaster(adapter, model_type="linear", random_state=0)
    # keep this tiny (we just want: trains + forecasts)
    X, y = adapter.generate_dataset(n_samples=200, seed=1)
    ml.fit(X, y)

    S = np.array([250.0], dtype=float)
    mu = ml.forecast_mean_path(S, 0, 5)
    assert mu.shape == (5,)
    assert np.isfinite(mu).all()

    # One-liner workflow (adapter + ML forecaster + fit) for seasonal
    f_seasonal, rep_seasonal = fit_ml_forecaster_from_exogenous(
        exo,
        model_type="linear",
        random_state=0,
        fit=FitConfig(n_samples=300, seed=3, val_samples=100, val_seed=4),
    )
    mu1 = forecast_with_default_state(f_seasonal, exo, t=0, H=4)
    assert mu1.shape == (4,)
    assert "train" in rep_seasonal and "val" in rep_seasonal

    # Regime adapter + ML forecaster (observable regime stored in state index 1)
    P = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)
    exo_reg = PoissonRegimeDemand(lam_by_regime=[220.0, 460.0], P=P, regime_index=1)
    adapter_reg = RegimeFeatureAdapter(exo_reg)
    Xr, yr = adapter_reg.generate_dataset(n_samples=300, seed=2)
    ml_reg = MlRegimeDemandForecaster(adapter_reg, model_type="linear", random_state=0)
    ml_reg.fit(Xr, yr)
    mu_r = ml_reg.forecast_mean_path(np.array([300.0, 0.0], dtype=float), 0, 6)
    assert mu_r.shape == (6,)
    assert np.isfinite(mu_r).all()

    # Multi-regime adapter (season/day/weather) dataset shape sanity
    from inventory.problems.demand_models import ExogenousPoissonMultiRegime

    exo_mr = ExogenousPoissonMultiRegime(season_index=1, day_index=2, weather_index=3, season_period=90)
    adapter_mr = MultiRegimeFeatureAdapter(exo_mr)

    Xm, ym = adapter_mr.generate_dataset(n_samples=120, seed=7, season0=2, day0=0, weather0=2)
    assert Xm.shape[0] == 120 and ym.shape == (120,)
    assert Xm.shape[1] == 1 + 4 + 7 + 3, "Expected [1, pi_season(4), pi_day(7), pi_weather(3)]"
    assert np.isfinite(Xm).all() and np.isfinite(ym).all()

    # Multi-step features shape sanity
    Xh = adapter_mr.features_path(np.array([300.0, 2.0, 0.0, 2.0], dtype=float), 0, 8)
    assert Xh.shape == (8, 1 + 4 + 7 + 3)

    # Multi-regime AR(1) adapter + forecaster sanity
    adapter_mr_ar1 = MultiRegimeAr1FeatureAdapter(exo_mr, lag_scale=100.0)
    Xm_ar1, ym_ar1 = adapter_mr_ar1.generate_dataset(n_samples=120, seed=9, season0=2, day0=0, weather0=2)
    assert Xm_ar1.shape[0] == 120 and ym_ar1.shape == (120,)
    assert Xm_ar1.shape[1] == (1 + 4 + 7 + 3 + 1)
    assert np.isfinite(Xm_ar1).all() and np.isfinite(ym_ar1).all()

    ml_mr_ar1 = MlAr1RegimeDemandForecaster(adapter_mr_ar1, model_type="linear", random_state=0)
    ml_mr_ar1.fit(Xm_ar1, ym_ar1)
    mu_mr_ar1 = ml_mr_ar1.forecast_mean_path(
        np.array([300.0, 2.0, 0.0, 2.0], dtype=float),
        0,
        5,
        info={"last_demand": 420.0},
    )
    assert mu_mr_ar1.shape == (5,)
    assert np.isfinite(mu_mr_ar1).all()

    # Plot helper should run (skip if matplotlib unavailable)
    try:
        from inventory.evaluation.plotting import plot_multi_regime_adapter_sample_paths

        fig, _axes, _ys, _mu = plot_multi_regime_adapter_sample_paths(
            adapter_mr,
            n_samples=25,
            seed=11,
            t_start=0,
            n_paths=2,
            r0=(2, 0, 2),
            show_pi=False,
            show=False,
        )
        try:
            fig.canvas.draw_idle()
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass
    except ImportError:
        pass

    # One-liner workflow for regime
    f_reg, rep_reg = fit_ml_forecaster_from_exogenous(
        exo_reg,
        model_type="linear",
        random_state=0,
        fit=FitConfig(n_samples=300, seed=5, val_samples=100, val_seed=6),
    )
    mu2 = forecast_with_default_state(f_reg, exo_reg, t=0, H=4, regime=0)
    assert mu2.shape == (4,)
    assert "train" in rep_reg and "val" in rep_reg

    # MILP CFA should produce a valid action even if MILP backend fails (fallback exists)
    policy_ml = OrderMilpCfaPolicy(forecaster=ml, H=3, x_max=480, dx=10, S_max=480)
    a = policy_ml.act(S, 0)
    assert np.asarray(a).shape == (1,)

    expert = ExpertDemandForecasterConstant350()
    policy_expert = OrderMilpCfaPolicy(forecaster=expert, H=3, x_max=480, dx=10, S_max=480)
    a2 = policy_expert.act(S, 0)
    assert np.asarray(a2).shape == (1,)

    # Forecast plotting helper (skip if matplotlib missing)
    try:
        from inventory.evaluation import show_forecast_paths

        class _TinyForecaster:
            def __init__(self, mu: float):
                self.mu = float(mu)

            def forecast_mean_path(self, S: np.ndarray, t: int, H: int, info: Optional[dict] = None) -> np.ndarray:
                _ = (S, t, info)
                return np.full((int(H),), self.mu, dtype=float)

        f1 = _TinyForecaster(mu=10.0)
        f2 = _TinyForecaster(mu=12.0)
        fig, _ax = show_forecast_paths(f1, f2, np.array([0.0], dtype=float), times=[0, 2], H=3, show=False, verbose=False)
        try:
            fig.canvas.draw_idle()
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass
    except ImportError:
        pass


def _case_lecture13_vfas() -> None:
    from inventory.core.dynamics import DynamicSystemMVP
    from inventory.policies.vfa import FqiGreedyVfaPolicy, PostDecisionGreedyVfaPolicy
    from inventory.problems.demand_models import PoissonSeasonalDemand
    from inventory.problems.inventory_mvp import inventory_cost, inventory_transition

    exo = PoissonSeasonalDemand(base=300.0, amp=90.0, period=20)
    sys1 = DynamicSystemMVP(
        transition_func=inventory_transition,
        cost_func=inventory_cost,
        exogenous_model=exo,
        sim_seed=42,
        d_s=1,
        d_x=1,
        d_w=1,
    )

    S0 = np.array([200.0], dtype=float)

    vfa_td = PostDecisionGreedyVfaPolicy(sys1, x_max=60, dx=10, alpha=0.1, expectation_samples=64)
    a = vfa_td.act(S0, 0)
    assert np.asarray(a).shape == (1,)
    vfa_td.train_td_value(S0=S0, T=5, n_episodes=5, seed0=2026, epsilon=0.2)
    assert vfa_td.w is not None and np.isfinite(vfa_td.w).all()

    fqi = FqiGreedyVfaPolicy(sys1, x_max=60, dx=10, mode="ridge", ridge_alpha=1.0)
    fqi.train_fqi(S0=S0, T=5, n_episodes=8, seed0=2026, n_iterations=2, behavior="random")
    a2 = fqi.act(S0, 0)
    assert np.asarray(a2).shape == (1,)


def _case_lecture14_dlas() -> None:
    from inventory.policies.dla_mcts import DlaMctsUctPolicy
    from inventory.policies.dla_milp import DlaMpcMilpPolicy
    from inventory.problems.demand_models import PoissonSeasonalDemand
    from inventory.problems.inventory_mvp import make_inventory_mvp_system

    exo = PoissonSeasonalDemand(base=300.0, amp=90.0, period=20)
    system = make_inventory_mvp_system(exogenous_model=exo, d_s=1, sim_seed=42)
    S = np.array([250.0], dtype=float)

    uct = DlaMctsUctPolicy(system, horizon=4, n_simulations=40, uct_c=1.4, x_max=80, dx=10, planning_seed=7)
    a1 = uct.act(S, 0, info={"crn_step_seed": 123})
    a2 = uct.act(S, 0, info={"crn_step_seed": 123})
    _assert_close(a1, a2, msg="UCT not reproducible under fixed step seed")

    milp = DlaMpcMilpPolicy(system, H=3, x_max=80, dx=10, S_max=480)
    a3 = milp.act(S, 0, info={"crn_step_seed": 123})
    assert np.asarray(a3).shape == (1,)


def _case_lecture15_hybrids() -> None:
    from inventory.forecasters.path import ConstantMeanPathForecaster
    from inventory.policies.baselines import OrderUpToPolicy
    from inventory.policies.hybrids import (
        Hybrid_DLA_CFA_BufferedForecast,
        Hybrid_DLA_Rollout_BasePFA,
        Hybrid_DLA_VFA_Terminal,
        HybridDlaCfaBufferedForecastPolicy,
        HybridDlaRolloutBasePfaPolicy,
        HybridDlaVfaTerminalPolicy,
        TerminalVFA_Linear,
        TerminalVfaLinear,
    )
    from inventory.problems.demand_models import PoissonSeasonalDemand
    from inventory.problems.inventory_mvp import make_inventory_mvp_system

    # Notebook-compatibility alias checks (Lecture 15 naming)
    assert Hybrid_DLA_CFA_BufferedForecast is HybridDlaCfaBufferedForecastPolicy
    assert Hybrid_DLA_VFA_Terminal is HybridDlaVfaTerminalPolicy
    assert Hybrid_DLA_Rollout_BasePFA is HybridDlaRolloutBasePfaPolicy
    assert TerminalVFA_Linear is TerminalVfaLinear

    exo = PoissonSeasonalDemand(base=300.0, amp=90.0, period=20)
    system = make_inventory_mvp_system(exogenous_model=exo, d_s=1, sim_seed=42)

    forecaster = ConstantMeanPathForecaster(mu=350.0)
    S = np.array([250.0], dtype=float)

    pol1 = HybridDlaCfaBufferedForecastPolicy(system, forecast=forecaster, H=4, dx=10, x_max=80, s_max=480)
    a1 = pol1.act(S, 0, info={"crn_step_seed": 123})
    assert np.asarray(a1).shape == (1,)

    terminal = TerminalVfaLinear(theta0=0.0, theta1=-0.1)
    pol2 = HybridDlaVfaTerminalPolicy(system, forecast=forecaster, terminal_vfa=terminal, H=4, dx=10, x_max=80, s_max=480)
    a2 = pol2.act(S, 0, info={"crn_step_seed": 123})
    assert np.asarray(a2).shape == (1,)

    base = OrderUpToPolicy(target_level=350, x_max=80, dx=10)
    pol3 = HybridDlaRolloutBasePfaPolicy(system, base_policy=base, H=4, dx=10, x_max=80, n_rollouts=10)
    a3 = pol3.act(S, 0, info={"crn_step_seed": 123})
    assert np.asarray(a3).shape == (1,)


def _case_lecture16_ppo() -> None:
    from inventory.policies.ppo import HybridPpoPolicy

    ppo = HybridPpoPolicy(d_s=2, s_max=480, x_max=80, dx=10, seed=0)
    S = np.array([250.0, 1.0], dtype=float)

    a_mean = ppo.act(S, 0, info={"deterministic": True, "det_mode": "mean"})
    a_amax = ppo.act(S, 0, info={"deterministic": True, "det_mode": "argmax"})
    assert np.asarray(a_mean).shape == (1,)
    assert np.asarray(a_amax).shape == (1,)

    # strict-CRN decision sampling when stochastic
    a1 = ppo.act(S, 0, info={"crn_step_seed": 123})
    a2 = ppo.act(S, 0, info={"crn_step_seed": 123})
    _assert_close(a1, a2, msg="PPO stochastic decision not reproducible under CRN seed")


def _case_lecture16_ppo_train_slow() -> None:
    """Tiny PPO train-then-act check (kept short for CI/smoke)."""

    from inventory.core.dynamics import DynamicSystemMVP
    from inventory.policies.baselines import OrderUpToPolicy
    from inventory.policies.ppo import HybridPpoPolicy, PPOHyperParams, train_ppo_with_eval_gate
    from inventory.problems.demand_models import PoissonSeasonalDemand
    from inventory.problems.inventory_mvp import inventory_cost, inventory_transition

    exo = PoissonSeasonalDemand(base=300.0, amp=90.0, period=20)
    system = DynamicSystemMVP(
        transition_func=inventory_transition,
        cost_func=inventory_cost,
        exogenous_model=exo,
        sim_seed=42,
        d_s=1,
        d_x=1,
        d_w=1,
    )

    hp = PPOHyperParams(lr=3e-4, n_epochs=2, minibatch_size=64)
    ppo = HybridPpoPolicy(d_s=1, s_max=480, x_max=80, dx=10, seed=0, hparams=hp)
    baseline = OrderUpToPolicy(target_level=350, x_max=80, dx=10)
    S0 = np.array([250.0], dtype=float)

    hist = train_ppo_with_eval_gate(
        system=system,
        ppo=ppo,
        baseline_policy=baseline,
        S0=S0,
        T=6,
        total_train_episodes=4,
        chunk_episodes=2,
        eval_episodes=4,
        eval_seed0=999,
        train_seed0=1234,
        verbose=False,
    )
    assert isinstance(hist, list) and len(hist) == 2

    # Gate can accept or reject; just validate expected record structure.
    for row in hist:
        assert set(row.keys()) >= {"gate", "baseline_mean", "candidate_ppo_mean", "best_ppo_mean", "accepted"}
        assert isinstance(row["accepted"], bool)

    # Validate save/restore works exactly.
    saved = ppo.get_params()
    import torch

    with torch.no_grad():
        p = next(iter(ppo.net.parameters()))
        orig = p.detach().clone()
        p.add_(1e-3)
        assert not torch.allclose(p, orig)
    ppo.set_params(saved)
    with torch.no_grad():
        p2 = next(iter(ppo.net.parameters()))
        assert torch.allclose(p2, orig)

    a = ppo.act(S0, 0, info={"deterministic": True, "det_mode": "mean"})
    assert np.asarray(a).shape == (1,)


def _case_lecture16_forecast_augmented_ppo_train_slow() -> None:
    """Tiny forecast-augmented PPO train-then-act check (kept short for CI/smoke)."""

    from inventory.core.dynamics import DynamicSystemMVP
    from inventory.policies.baselines import OrderUpToPolicy
    from inventory.policies.ppo import ForecastAugmentedHybridPpoPolicy, PPOHyperParams, train_ppo_with_eval_gate
    from inventory.problems.demand_models import PoissonRegimeDemand
    from inventory.problems.inventory_mvp import inventory_cost, inventory_transition

    class _TinyForecaster:
        def forecast_mean_path(self, state: np.ndarray, t: int, H: int, info: Optional[dict] = None) -> np.ndarray:
            _ = t, info
            return np.full(H, 30.0 + 0.1 * float(state[0]), dtype=float)

    P = np.array([[0.92, 0.08], [0.12, 0.88]], dtype=float)
    exo = PoissonRegimeDemand(lam_by_regime=[220, 460], P=P, regime_index=1)
    system = DynamicSystemMVP(
        transition_func=inventory_transition,
        cost_func=inventory_cost,
        exogenous_model=exo,
        sim_seed=42,
        d_s=2,
        d_x=1,
        d_w=2,
    )

    hp = PPOHyperParams(lr=3e-4, n_epochs=2, minibatch_size=32)
    ppo = ForecastAugmentedHybridPpoPolicy(
        forecaster=_TinyForecaster(),
        raw_state_dim=2,
        forecast_horizon=3,
        demand_scale=100.0,
        s_max=480,
        x_max=80,
        dx=10,
        seed=0,
        hparams=hp,
    )
    baseline = OrderUpToPolicy(target_level=350, x_max=80, dx=10)
    S0 = np.array([250.0, 0.0], dtype=float)

    hist = train_ppo_with_eval_gate(
        system=system,
        ppo=ppo,
        baseline_policy=baseline,
        S0=S0,
        T=6,
        total_train_episodes=4,
        chunk_episodes=2,
        eval_episodes=4,
        eval_seed0=999,
        train_seed0=1234,
        verbose=False,
    )
    assert isinstance(hist, list) and len(hist) == 2

    a = ppo.act(S0, 0, info={"deterministic": True, "det_mode": "mean"})
    assert np.asarray(a).shape == (1,)


def _case_lecture17_alphazero() -> None:
    from inventory.policies.alphazero import HybridAlphaZeroPolicy
    from inventory.problems.demand_models import PoissonMultiRegimeDemand, PoissonRegimeDemand
    from inventory.problems.inventory_mvp import make_inventory_multi_regime_system, make_inventory_mvp_system

    P = np.array([[0.92, 0.08], [0.12, 0.88]], dtype=float)
    exo = PoissonRegimeDemand(lam_by_regime=[220, 460], P=P, regime_index=1)
    system = make_inventory_mvp_system(exogenous_model=exo, d_s=2, sim_seed=42)

    az = HybridAlphaZeroPolicy(
        system,
        x_max=80,
        dx=10,
        H=6,
        n_sims=30,
        c_puct=1.5,
        gamma=1.0,
        seed=0,
        tau_eval=0.0,
    )

    S = np.array([300.0, 0.0], dtype=float)
    a = az.act(S, 0, info={"crn_step_seed": 123})
    assert np.asarray(a).shape == (1,)

    exo_multi = PoissonMultiRegimeDemand(season_index=1, day_index=2, weather_index=3, season_period=90)
    system_multi = make_inventory_multi_regime_system(exogenous_model=exo_multi, sim_seed=42)

    az_multi = HybridAlphaZeroPolicy(
        system_multi,
        x_max=80,
        dx=10,
        H=4,
        n_sims=12,
        c_puct=1.5,
        gamma=1.0,
        seed=0,
        tau_eval=0.0,
    )

    S_multi = np.array([300.0, 2.0, 0.0, 2.0], dtype=float)
    feats = az_multi._featurize(S_multi, 0, 10)
    assert feats.shape == (7,)
    _assert_close(feats[:4], np.array([300.0 / 80.0, 2.0 / 3.0, 0.0, 2.0 / 2.0]), msg="AlphaZero should expose normalized multi-regime state features")

    S_multi_alt = np.array([300.0, 2.0, 1.0, 0.0], dtype=float)
    assert az_multi._decision_seed(S_multi, 0) == az_multi._decision_seed(S_multi_alt, 0)

    az_multi_full = HybridAlphaZeroPolicy(
        system_multi,
        x_max=80,
        dx=10,
        H=4,
        n_sims=12,
        c_puct=1.5,
        gamma=1.0,
        seed=0,
        tau_eval=0.0,
        transposition_key="full_state",
    )
    assert az_multi_full._decision_seed(S_multi, 0) != az_multi_full._decision_seed(S_multi_alt, 0)

    a_multi = az_multi.act(S_multi, 0, info={"crn_step_seed": 123, "T": 10})
    assert np.asarray(a_multi).shape == (1,)


def _case_lecture17_alphazero_train_slow() -> None:
    """Tiny AlphaZero self-play iteration + gate (very small numbers)."""

    from inventory.policies.alphazero import HybridAlphaZeroPolicy
    from inventory.problems.demand_models import PoissonRegimeDemand
    from inventory.problems.inventory_mvp import make_inventory_mvp_system

    P = np.array([[0.92, 0.08], [0.12, 0.88]], dtype=float)
    exo = PoissonRegimeDemand(lam_by_regime=[220, 460], P=P, regime_index=1)
    system = make_inventory_mvp_system(exogenous_model=exo, d_s=2, sim_seed=42)

    az = HybridAlphaZeroPolicy(
        system,
        x_max=80,
        dx=10,
        H=5,
        n_sims=8,
        c_puct=1.5,
        gamma=1.0,
        seed=0,
        tau_eval=0.0,
        value_scale=2000.0,
    )

    S0 = np.array([300.0, 0.0], dtype=float)
    # Baseline action before training
    a0 = az.act(S0, 0, info={"T": 6, "crn_step_seed": 123})

    hist = az.fit_self_play(
        S0=S0,
        T=6,
        n_iterations=1,
        episodes_per_iter=2,
        buffer_max=500,
        tau_train=1.0,
        lr=0.02,
        value_weight=0.5,
        train_steps=30,
        batch_size=16,
        gate_episodes=4,
        gate_seed0=2026,
        seed0=0,
        verbose=False,
        info={"T": 6},
    )
    assert "accepted" in hist and len(hist["accepted"]) == 1

    a1 = az.act(S0, 0, info={"T": 6, "crn_step_seed": 123})
    assert np.asarray(a1).shape == (1,)
    # Don't assert improvement or action change; just ensure train path executes.
    _ = (a0, a1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke tests for canonical extracted inventory lectures.")
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Run only the fast subset (default).",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Run all checks (includes --slow).",
    )
    ap.add_argument(
        "--slow",
        action="store_true",
        help="Include tiny train-then-act checks (PPO + AlphaZero).",
    )
    args = ap.parse_args()

    cases: list[tuple[str, Callable[[], None]]] = [
        ("core_foundations", _case_core_foundations),
        ("lecture01_bayesian_eval", _case_lecture01_bayesian_eval),
        ("lecture01_frequentist_eval", _case_lecture01_frequentist_eval),
        ("lecture01_delta_helpers", _case_lecture01_delta_helpers),
        ("lecture03_dp", _case_lecture03_dp),
        ("lecture11_pfas", _case_lecture11_pfas),
        ("problem_backorders", _case_problem_backorders),
        ("lecture12_leadtime", _case_lecture12_leadtime_problem),
        ("lecture12_cfas", _case_lecture12_cfas_and_forecasters),
        ("lecture13_vfas", _case_lecture13_vfas),
        ("lecture14_dlas", _case_lecture14_dlas),
        ("lecture15_hybrids", _case_lecture15_hybrids),
        ("lecture16_ppo", _case_lecture16_ppo),
        ("lecture17_alphazero", _case_lecture17_alphazero),
    ]

    if args.slow or args.all:
        cases.extend(
            [
                ("lecture16_ppo_train", _case_lecture16_ppo_train_slow),
                ("lecture16_forecast_augmented_ppo_train", _case_lecture16_forecast_augmented_ppo_train_slow),
                ("lecture17_az_train", _case_lecture17_alphazero_train_slow),
            ]
        )

    results: list[SmokeCaseResult] = []
    for name, fn in cases:
        res = _run_case(name, fn)
        results.append(res)
        if res.ok:
            print(f"[OK]   {name:18s} ({res.seconds:5.2f}s)")
        else:
            print(f"[FAIL] {name:18s} ({res.seconds:5.2f}s) :: {res.error}")

    n_fail = sum(1 for r in results if not r.ok)
    print(f"\nSmoke summary: {len(results) - n_fail}/{len(results)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
