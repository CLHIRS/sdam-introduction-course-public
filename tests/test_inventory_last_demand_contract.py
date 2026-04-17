import numpy as np

from inventory.core.dynamics import DynamicSystemMVP


class _DeterministicDemand:
    def sample(self, state: np.ndarray, action: np.ndarray, t: int, rng: np.random.Generator) -> np.ndarray:
        _ = state, action, rng
        return np.array([10.0 + float(t)], dtype=float)


class _RecordingPolicy:
    def __init__(self) -> None:
        self.seen_info: list[dict] = []

    def act(self, state: np.ndarray, t: int, info: dict | None = None) -> np.ndarray:
        _ = state, t
        self.seen_info.append({} if info is None else dict(info))
        return np.array([0.0], dtype=float)


def test_rollout_injects_last_demand_from_previous_step() -> None:
    system = DynamicSystemMVP(
        transition_func=lambda s, a, w, t: s,
        cost_func=lambda s, a, w, t: 0.0,
        exogenous_model=_DeterministicDemand(),
        sim_seed=42,
        d_s=1,
        d_x=1,
        d_w=1,
    )
    policy = _RecordingPolicy()

    system.simulate(policy, np.array([0.0], dtype=float), T=4, seed=7, info={"deterministic": True})

    assert len(policy.seen_info) == 4
    assert "last_demand" not in policy.seen_info[0]
    assert float(policy.seen_info[1]["last_demand"]) == 10.0
    assert float(policy.seen_info[2]["last_demand"]) == 11.0
    assert float(policy.seen_info[3]["last_demand"]) == 12.0


def test_rollout_preserves_other_info_keys_when_injecting_last_demand() -> None:
    system = DynamicSystemMVP(
        transition_func=lambda s, a, w, t: s,
        cost_func=lambda s, a, w, t: 0.0,
        exogenous_model=_DeterministicDemand(),
        sim_seed=42,
        d_s=1,
        d_x=1,
        d_w=1,
    )
    policy = _RecordingPolicy()

    system.simulate(policy, np.array([0.0], dtype=float), T=2, seed=7, info={"deterministic": True, "tag": "keep-me"})

    assert policy.seen_info[0]["tag"] == "keep-me"
    assert policy.seen_info[1]["tag"] == "keep-me"
    assert float(policy.seen_info[1]["last_demand"]) == 10.0


def test_rollout_optionally_injects_bounded_demand_history() -> None:
    system = DynamicSystemMVP(
        transition_func=lambda s, a, w, t: s,
        cost_func=lambda s, a, w, t: 0.0,
        exogenous_model=_DeterministicDemand(),
        sim_seed=42,
        d_s=1,
        d_x=1,
        d_w=1,
        demand_history_window=2,
    )
    policy = _RecordingPolicy()

    system.simulate(policy, np.array([0.0], dtype=float), T=4, seed=7, info={"deterministic": True})

    assert len(policy.seen_info) == 4
    assert "demand_history" not in policy.seen_info[0]
    assert np.allclose(policy.seen_info[1]["demand_history"], np.array([10.0], dtype=float), atol=0.0, rtol=0.0)
    assert np.allclose(policy.seen_info[2]["demand_history"], np.array([10.0, 11.0], dtype=float), atol=0.0, rtol=0.0)
    assert np.allclose(policy.seen_info[3]["demand_history"], np.array([11.0, 12.0], dtype=float), atol=0.0, rtol=0.0)
    assert float(policy.seen_info[3]["last_demand"]) == 12.0