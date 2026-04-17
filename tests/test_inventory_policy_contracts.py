import numpy as np

from inventory.core.dynamics import DynamicSystemMVP
from inventory.policies.baselines import OrderUpToPolicy
from inventory.policies.alphazero import HybridAlphaZeroPolicy
from inventory.policies.ppo import ForecastAugmentedHybridPpoPolicy, PPOHyperParams
from inventory.problems.demand_models import ExogenousPoissonMultiRegime
from inventory.problems.inventory_mvp import inventory_cost, inventory_transition, make_inventory_multi_regime_system


def test_order_up_to_policy_act_contract() -> None:
    pol = OrderUpToPolicy(target_level=105.0, x_max=200.0, dx=10)

    a = pol.act(np.array([50.0], dtype=float), t=0, info={"deterministic": True})

    assert isinstance(a, np.ndarray)
    assert a.shape == (1,)
    assert np.isfinite(a).all()

    x = int(round(float(a[0])))
    assert 0 <= x <= 200
    assert x % 10 == 0


def test_hybrid_alphazero_full_state_features_and_transposition_modes() -> None:
    exo = ExogenousPoissonMultiRegime(season_index=1, day_index=2, weather_index=3, season_period=90)
    system = make_inventory_multi_regime_system(exogenous_model=exo, sim_seed=42)
    state = np.array([300.0, 2.0, 0.0, 2.0], dtype=float)
    alt_state = np.array([300.0, 2.0, 1.0, 0.0], dtype=float)

    pol_auto = HybridAlphaZeroPolicy(
        system,
        x_max=80,
        dx=10,
        s_max=480.0,
        net_backend="numpy",
        H=4,
        n_sims=12,
        seed=0,
        transposition_key="auto",
    )
    feats = pol_auto._featurize(state, 0, 10)

    assert feats.shape == (7,)
    np.testing.assert_allclose(feats[:4], np.array([300.0 / 480.0, 2.0 / 3.0, 0.0, 1.0]))
    assert pol_auto._decision_seed(state, 0) == pol_auto._decision_seed(alt_state, 0)

    pol_full = HybridAlphaZeroPolicy(
        system,
        x_max=80,
        dx=10,
        s_max=480.0,
        net_backend="numpy",
        H=4,
        n_sims=12,
        seed=0,
        transposition_key="full_state",
    )
    assert pol_full._decision_seed(state, 0) != pol_full._decision_seed(alt_state, 0)

    action = pol_full.act(state, 0, info={"deterministic": True, "T": 10})
    assert isinstance(action, np.ndarray)
    assert action.shape == (1,)


def test_forecast_augmented_hybrid_ppo_contract() -> None:
    class _TinyForecaster:
        def forecast_mean_path(self, state: np.ndarray, t: int, H: int, info=None) -> np.ndarray:
            _ = t, info
            return np.full(H, float(state[0]) / 10.0 + 5.0, dtype=float)

    system = DynamicSystemMVP(
        transition_func=inventory_transition,
        cost_func=inventory_cost,
        exogenous_model=ExogenousPoissonMultiRegime(season_index=1, day_index=2, weather_index=3, season_period=90),
        sim_seed=42,
        d_s=4,
        d_x=1,
        d_w=4,
    )
    hp = PPOHyperParams(lr=3e-4, n_epochs=1, minibatch_size=16)
    pol = ForecastAugmentedHybridPpoPolicy(
        forecaster=_TinyForecaster(),
        raw_state_dim=4,
        forecast_horizon=3,
        demand_scale=100.0,
        s_max=480,
        x_max=80,
        dx=10,
        hparams=hp,
        seed=0,
    )
    state = np.array([300.0, 2.0, 0.0, 2.0], dtype=float)

    obs = pol._observation_vector(state, 0, info=None)
    assert obs.shape == (7,)
    np.testing.assert_allclose(obs[:4], state)
    np.testing.assert_allclose(obs[4:], np.full(3, 0.35, dtype=float))

    action = pol.act(state, 0, info={"deterministic": True, "det_mode": "mean"})
    assert isinstance(action, np.ndarray)
    assert action.shape == (1,)
