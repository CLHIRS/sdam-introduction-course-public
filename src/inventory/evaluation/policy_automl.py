from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np

from inventory.core.dynamics import DynamicSystemMVP
from inventory.core.policy import Policy
from inventory.evaluation.info import make_eval_info
from inventory.forecasters.naive import ExogenousAwareMeanForecaster
from inventory.policies.baselines import OrderUpToPolicy
from inventory.policies.cfa_milp import OrderMilpCfaPolicy
from inventory.policies.hybrid_dla import HybridDlaRolloutComposablePolicy
from inventory.policies.dla_mcts import DlaMctsUctPolicy
from inventory.policies.dla_milp import DlaMpcMilpPolicy
from inventory.policies.dp import DPSolverMultiRegimeScenarios, DynamicProgrammingSolver1D
from inventory.policies.pfa import DirectActionPFA, OrderUpToBlackboxPFA
from inventory.policies.vfa import _build_supervised_regressor
from inventory.problems.demand_models import PoissonConstantDemand, PoissonMultiRegimeDemand
from inventory.problems.inventory_mvp import make_inventory_multi_regime_system, make_inventory_mvp_system


@dataclass(frozen=True)
class PolicyAutoMLDemoProblem:
    """Small, repo-native problem bundle for policy-selection demos.

    This deliberately keeps the problem simple so that notebooks can focus on the
    AutoML workflow:
    - define a search space over policy families + hyperparameters
    - evaluate every candidate with strict CRN
    - rank candidates by mean total cost (optionally with a runtime penalty)
    """

    name: str
    system: DynamicSystemMVP
    exogenous_model: Any
    S0: np.ndarray
    T: int
    x_max: int
    dx: int
    s_max: int
    p: float
    c: float
    h: float
    b: float
    K: float
    eval_info: dict[str, Any]
    train_info: Mapping[str, Any] | None = None
    forecaster_registry: Mapping[str, Any] | None = None
    default_cfa_forecaster_key: str | None = None
    default_dla_forecaster_key: str | None = None
    feature_fn_registry: Mapping[str, Callable[[Any, int], Any]] | None = None
    prefit_policy_registry: Mapping[str, Policy] | None = None
    component_policy_registry: Mapping[str, Policy] | None = None


@dataclass(frozen=True)
class PolicyAutoMLSearchResult:
    label: str
    family: str
    config: dict[str, Any]
    mean_total_cost: float
    std_total_cost: float
    runtime_sec: float
    score: float
    totals: np.ndarray


@dataclass(frozen=True)
class PolicyAutoMLSmacResult:
    incumbent_config: dict[str, Any]
    incumbent_label: str
    incumbent_score: float
    incumbent_policy: Policy | None
    runhistory_rows: list[dict[str, Any]]
    n_trials_finished: int
    n_trials_requested: int
    scenario_seed: int


def smac3_available() -> bool:
    """Return True if the optional SMAC3 stack is importable."""

    try:
        import ConfigSpace  # noqa: F401
        import smac  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _require_smac3() -> tuple[Any, Any, Any, Any, Any]:
    """Import SMAC3 + ConfigSpace lazily so the base repo stays usable without them."""

    try:
        from ConfigSpace import AndConjunction, ConfigurationSpace, EqualsCondition
        from smac import HyperparameterOptimizationFacade, Scenario
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only when optional deps are absent
        raise ModuleNotFoundError(
            "SMAC3 integration requires the optional `smac` dependency. "
            "Install it in this repository with `poetry add smac`."
        ) from exc

    return ConfigurationSpace, EqualsCondition, AndConjunction, HyperparameterOptimizationFacade, Scenario


def make_policy_automl_demo_problem(
    *,
    lam: float = 300.0,
    T: int = 20,
    initial_inventory: float = 150.0,
    x_max: int = 480,
    dx: int = 10,
    s_max: int = 480,
    sim_seed: int = 42,
    p: float = 2.0,
    c: float = 0.5,
    h: float = 0.03,
    b: float = 1.0,
) -> PolicyAutoMLDemoProblem:
    """Create a didactic constant-demand inventory problem for AutoML demos."""

    exo = PoissonConstantDemand(lam=float(lam))
    system = make_inventory_mvp_system(
        exogenous_model=exo,
        d_s=1,
        s_max=int(s_max),
        sim_seed=int(sim_seed),
    )
    S0 = np.array([float(initial_inventory)], dtype=float)
    eval_info = make_eval_info(det_mode="argmax", T=int(T))
    return PolicyAutoMLDemoProblem(
        name="constant_inventory",
        system=system,
        exogenous_model=exo,
        S0=S0,
        T=int(T),
        x_max=int(x_max),
        dx=int(dx),
        s_max=int(s_max),
        p=float(p),
        c=float(c),
        h=float(h),
        b=float(b),
        K=0.0,
        eval_info=eval_info,
    )


def make_policy_automl_demo_problem_multi_regime(
    *,
    T: int = 12,
    initial_inventory: float = 300.0,
    initial_season: int = 2,
    initial_day: int = 0,
    initial_weather: int = 2,
    x_max: int = 480,
    dx: int = 10,
    s_max: int = 480,
    sim_seed: int = 42,
    p: float = 2.0,
    c: float = 0.5,
    h: float = 0.03,
    b: float = 1.0,
    K: float = 0.0,
    season_period: int = 90,
) -> PolicyAutoMLDemoProblem:
    """Create a didactic observable multi-regime inventory problem for AutoML demos.

    The defaults intentionally set `K=0.0` so the exact DP benchmark remains
    comparable to the simulated system used in the notebook.
    """

    exo = PoissonMultiRegimeDemand(
        season_index=1,
        day_index=2,
        weather_index=3,
        season_period=int(season_period),
    )
    system = make_inventory_multi_regime_system(
        exogenous_model=exo,
        s_max=int(s_max),
        sim_seed=int(sim_seed),
        p=float(p),
        c=float(c),
        h=float(h),
        b=float(b),
        K=float(K),
    )
    S0 = np.array(
        [
            float(initial_inventory),
            float(initial_season),
            float(initial_day),
            float(initial_weather),
        ],
        dtype=float,
    )
    eval_info = make_eval_info(det_mode="argmax", T=int(T))
    return PolicyAutoMLDemoProblem(
        name="multi_regime_inventory",
        system=system,
        exogenous_model=exo,
        S0=S0,
        T=int(T),
        x_max=int(x_max),
        dx=int(dx),
        s_max=int(s_max),
        p=float(p),
        c=float(c),
        h=float(h),
        b=float(b),
        K=float(K),
        eval_info=eval_info,
    )


def make_policy_automl_demo_search_space(problem: PolicyAutoMLDemoProblem) -> list[dict[str, Any]]:
    """Return a compact search space across several existing policy classes.

    The search space is intentionally small and explicit so students can inspect
    every candidate. In a real SMAC run, the same choices would be encoded as a
    conditional configuration space rather than an explicit list.
    """

    x_max = int(problem.x_max)
    target_levels = [260, 320, 380]
    H_short = [1, 3, 5]

    configs: list[dict[str, Any]] = []

    for target in target_levels:
        configs.append(
            {
                "label": f"OrderUpTo[S={target}]",
                "family": "order_up_to",
                "target_level": int(target),
            }
        )

    for target in target_levels:
        configs.append(
            {
                "label": f"BlackboxPFA[theta={target}]",
                "family": "pfa_blackbox",
                "theta": float(target),
                "theta_min": 0.0,
                "theta_max": float(x_max),
            }
        )

    for H in H_short:
        configs.append(
            {
                "label": f"CFA_MILP[H={H}]",
                "family": "cfa_milp",
                "H": int(H),
            }
        )

    for H in [2, 4]:
        configs.append(
            {
                "label": f"DLA_MPC[H={H}]",
                "family": "dla_mpc",
                "H": int(H),
            }
        )

    for horizon, n_sims, uct_c in [
        (2, 24, 0.7),
        (4, 40, 1.4),
        (6, 80, 1.4),
    ]:
        configs.append(
            {
                "label": f"DLA_MCTS[h={horizon}, sims={n_sims}, c={uct_c}]",
                "family": "dla_mcts",
                "horizon": int(horizon),
                "n_simulations": int(n_sims),
                "uct_c": float(uct_c),
                "planning_seed": 123,
            }
        )

    return configs


def make_policy_automl_demo_search_space_multi_regime(problem: PolicyAutoMLDemoProblem) -> list[dict[str, Any]]:
    """Return a compact multi-regime search space across existing policy classes."""

    x_max = int(problem.x_max)
    target_levels = [260, 320, 380]
    H_short = [1, 3, 5]

    configs: list[dict[str, Any]] = []

    for target in target_levels:
        configs.append(
            {
                "label": f"OrderUpTo[S={target}]",
                "family": "order_up_to",
                "target_level": int(min(target, x_max)),
            }
        )

    for target in target_levels:
        configs.append(
            {
                "label": f"BlackboxPFA[theta={target}]",
                "family": "pfa_blackbox",
                "theta": float(min(target, x_max)),
                "theta_min": 0.0,
                "theta_max": float(x_max),
            }
        )

    for H in H_short:
        configs.append(
            {
                "label": f"CFA_MILP[H={H}]",
                "family": "cfa_milp",
                "H": int(H),
            }
        )

    for H in [2, 4]:
        configs.append(
            {
                "label": f"DLA_MPC[H={H}]",
                "family": "dla_mpc",
                "H": int(H),
            }
        )

    for horizon, n_sims, uct_c in [
        (2, 24, 0.7),
        (4, 40, 1.4),
        (6, 80, 1.4),
    ]:
        configs.append(
            {
                "label": f"DLA_MCTS[h={horizon}, sims={n_sims}, c={uct_c}]",
                "family": "dla_mcts",
                "horizon": int(horizon),
                "n_simulations": int(n_sims),
                "uct_c": float(uct_c),
                "planning_seed": 123,
            }
        )

    return configs


def _config_like_to_dict(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)

    get_dictionary = getattr(config, "get_dictionary", None)
    if callable(get_dictionary):
        return dict(get_dictionary())

    items = getattr(config, "items", None)
    if callable(items):
        return dict(items())

    return dict(config)


def _clip_and_round_to_grid(value: float, *, lower: float, upper: float, step: int) -> float:
    x = float(np.clip(float(value), float(lower), float(upper)))
    if int(step) > 1:
        x = float(step) * float(np.round(x / float(step)))
        x = float(np.clip(x, float(lower), float(upper)))
    return float(x)


def _normalize_for_cache(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple((str(k), _normalize_for_cache(v)) for k, v in sorted(value.items(), key=lambda kv: str(kv[0])))
    if isinstance(value, np.ndarray):
        return ("ndarray", tuple(np.asarray(value).reshape(-1).tolist()))
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_for_cache(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _policy_config_cache_key(
    config: Mapping[str, Any],
    *,
    training_seed0: int,
    train_info: Mapping[str, Any] | None,
) -> Any:
    return (
        _normalize_for_cache(dict(config)),
        int(training_seed0),
        _normalize_for_cache(dict(train_info or {})),
    )


def _resolve_train_info(
    problem: PolicyAutoMLDemoProblem,
    train_info: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source = train_info
    if source is None:
        source = problem.train_info
    if source is None:
        source = problem.eval_info
    return dict(source)


def _configspace_range_or_singleton(lower: int | float, upper: int | float) -> Any:
    if lower == upper:
        return [lower]
    return (lower, upper)


def smac_policy_menu_to_search_space_config(policy_menu: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a notebook-facing policy menu into helper-layer SMAC kwargs.

    Preferred teaching-facing structure:

        {
            "tunable_families": {
                "order_up_to": {...},
                "cfa_milp": {...},
                "dla_mpc": {...},
            },
            "trainable_families": {
                "direct_action_pfa": {...},
            },
            "composite_families": {
                "hybrid_rollout_choice": {...},
            },
            "prefit_variants": {
                "DirectActionPFA_Tree": {...},
            },
        }

    Minimal semantics:
    - `tunable_families` lists policy families that SMAC may instantiate and tune
      directly through the shared helper constructors in `build_policy_from_config(...)`.
    - `trainable_families` lists policy families that must be instantiated and
      then trained/materialized inside the search loop.
    - `composite_families` lists policy families that assemble multiple existing
      policy objects into one composite/searchable architecture.
    - `prefit_variants` lists already-built policy artifacts that should be treated
      as categorical choices inside the SMAC search space.
    - Family-specific `bounds` entries are translated into the corresponding SMAC
      ConfigSpace parameter ranges.
    - Prefit variant names are translated into the helper-layer family
      `prefit_policy`, which performs a registry lookup instead of re-training.

    Backward-compatible fallback:
        a flat family -> spec mapping, where "prefit_policy" contains a "choices" field.
    """

    if (
        "tunable_families" in policy_menu
        or "trainable_families" in policy_menu
        or "composite_families" in policy_menu
        or "prefit_variants" in policy_menu
    ):
        tunable = dict(policy_menu.get("tunable_families", {}))
        trainable = dict(policy_menu.get("trainable_families", {}))
        composite = dict(policy_menu.get("composite_families", {}))
        prefit = dict(policy_menu.get("prefit_variants", {}))
    else:
        raw = dict(policy_menu)
        tunable = {k: v for k, v in raw.items() if str(k) != "prefit_policy"}
        trainable = {}
        composite = {}
        prefit_spec = dict(raw.get("prefit_policy", {}))
        prefit = {
            str(name): {"policy_class": "PrefitPolicyRegistryLookup"}
            for name in tuple(prefit_spec.get("choices", ()))
        }

    families = (
        [str(name) for name in tunable.keys()]
        + [str(name) for name in trainable.keys()]
        + [str(name) for name in composite.keys()]
    )
    cfg: dict[str, Any] = {"families": tuple(families)}

    if "order_up_to" in tunable and "bounds" in tunable["order_up_to"]:
        cfg["target_level_bounds"] = tuple(tunable["order_up_to"]["bounds"])
    if "pfa_blackbox" in tunable and "bounds" in tunable["pfa_blackbox"]:
        cfg["theta_bounds"] = tuple(tunable["pfa_blackbox"]["bounds"])
    if "cfa_milp" in tunable and "bounds" in tunable["cfa_milp"]:
        cfg["H_cfa_bounds"] = tuple(tunable["cfa_milp"]["bounds"])
    if "dla_mpc" in tunable and "bounds" in tunable["dla_mpc"]:
        cfg["H_mpc_bounds"] = tuple(tunable["dla_mpc"]["bounds"])
    if "dla_mcts" in tunable:
        spec = dict(tunable["dla_mcts"])
        if "horizon_bounds" in spec:
            cfg["horizon_mcts_bounds"] = tuple(spec["horizon_bounds"])
        if "n_simulations_bounds" in spec:
            cfg["n_simulations_mcts_bounds"] = tuple(spec["n_simulations_bounds"])
        if "uct_c_bounds" in spec:
            cfg["uct_c_mcts_bounds"] = tuple(spec["uct_c_bounds"])
    if "direct_action_pfa" in trainable:
        spec = dict(trainable["direct_action_pfa"])
        if "regressor_modes" in spec:
            cfg["direct_action_pfa_regressor_choices"] = tuple(str(mode) for mode in spec["regressor_modes"])
        if "train_n_episodes_bounds" in spec:
            cfg["direct_action_pfa_train_episodes_bounds"] = tuple(spec["train_n_episodes_bounds"])
        if "train_n_iter_bounds" in spec:
            cfg["direct_action_pfa_train_iter_bounds"] = tuple(spec["train_n_iter_bounds"])
        if "H" in spec:
            cfg["direct_action_pfa_H"] = int(spec["H"])
        if "feature_fn_key" in spec:
            cfg["direct_action_pfa_feature_fn_key"] = str(spec["feature_fn_key"])
        model_hyper = dict(spec.get("model_hyperparameters", {}))
        tree_spec = dict(model_hyper.get("decision_tree", {}))
        elastic_spec = dict(model_hyper.get("elastic", {}))
        if "tree_max_depth_bounds" in tree_spec:
            cfg["direct_action_pfa_tree_max_depth_bounds"] = tuple(tree_spec["tree_max_depth_bounds"])
        if "elastic_alpha_bounds" in elastic_spec:
            cfg["direct_action_pfa_elastic_alpha_bounds"] = tuple(elastic_spec["elastic_alpha_bounds"])
        if "elastic_l1_ratio_bounds" in elastic_spec:
            cfg["direct_action_pfa_elastic_l1_ratio_bounds"] = tuple(elastic_spec["elastic_l1_ratio_bounds"])
    if "hybrid_rollout_choice" in composite:
        spec = dict(composite["hybrid_rollout_choice"])
        if "rollout_policy_choices" in spec:
            cfg["hybrid_rollout_rollout_policy_choices"] = tuple(str(name) for name in spec["rollout_policy_choices"])
        if "candidate_policy_choices" in spec:
            cfg["hybrid_rollout_candidate_policy_choices"] = tuple(str(name) for name in spec["candidate_policy_choices"])
        fixed_params = dict(spec.get("fixed_params", {}))
        if "H" in fixed_params:
            cfg["hybrid_rollout_H"] = int(fixed_params["H"])
        if "n_rollouts" in fixed_params:
            cfg["hybrid_rollout_n_rollouts"] = int(fixed_params["n_rollouts"])
        if "candidate_radius_steps" in fixed_params:
            cfg["hybrid_rollout_candidate_radius_steps"] = int(fixed_params["candidate_radius_steps"])
        if "include_full_action_grid" in fixed_params:
            cfg["hybrid_rollout_include_full_action_grid"] = bool(fixed_params["include_full_action_grid"])
        if "decision_seed" in fixed_params:
            cfg["hybrid_rollout_decision_seed"] = int(fixed_params["decision_seed"])

    if prefit:
        cfg["families"] = tuple([*families, "prefit_policy"])
        cfg["prefit_policy_names"] = tuple(str(name) for name in prefit.keys())

    return cfg


def _resolve_prefit_policy_names(
    problem: PolicyAutoMLDemoProblem,
    prefit_policy_names: Sequence[str] | None,
) -> list[str]:
    registry = dict(problem.prefit_policy_registry or {})
    if prefit_policy_names is None:
        names = list(registry.keys())
    else:
        names = [str(name) for name in prefit_policy_names]

    missing = [name for name in names if name not in registry]
    if missing:
        raise ValueError(f"Requested prefit policy names are not in the registry: {missing}")

    return names


def _resolve_component_policy_names(
    problem: PolicyAutoMLDemoProblem,
    component_policy_names: Sequence[str] | None,
) -> list[str]:
    registry = dict(problem.component_policy_registry or {})
    if component_policy_names is None:
        names = list(registry.keys())
    else:
        names = [str(name) for name in component_policy_names]

    missing = [name for name in names if name not in registry]
    if missing:
        raise ValueError(f"Requested component policy names are not in the registry: {missing}")

    return names


def _resolve_feature_fn(
    problem: PolicyAutoMLDemoProblem,
    feature_fn_key: str | None,
) -> Callable[[Any, int], Any] | None:
    if feature_fn_key is None:
        return None

    registry = dict(problem.feature_fn_registry or {})
    if feature_fn_key not in registry:
        raise ValueError(f"Requested feature function key is not in the registry: {feature_fn_key}")
    return registry[feature_fn_key]


def _resolve_smac_families(
    problem: PolicyAutoMLDemoProblem,
    families: Sequence[str] | None,
    *,
    prefit_policy_names: Sequence[str] | None = None,
    hybrid_rollout_rollout_policy_choices: Sequence[str] | None = None,
    hybrid_rollout_candidate_policy_choices: Sequence[str] | None = None,
) -> list[str]:
    allowed = {
        "order_up_to",
        "pfa_blackbox",
        "cfa_milp",
        "dla_mpc",
        "dla_mcts",
        "direct_action_pfa",
        "hybrid_rollout_choice",
        "prefit_policy",
    }

    if families is None:
        resolved = ["order_up_to", "pfa_blackbox", "cfa_milp", "dla_mpc", "dla_mcts"]
        if problem.prefit_policy_registry:
            resolved.append("prefit_policy")
    else:
        resolved = [str(family) for family in families]

    unknown = [family for family in resolved if family not in allowed]
    if unknown:
        raise ValueError(f"Unsupported SMAC3 policy family request: {unknown}")

    if "prefit_policy" in resolved and not _resolve_prefit_policy_names(problem, prefit_policy_names):
        raise ValueError("Family 'prefit_policy' requires at least one registered prefit policy")
    if "hybrid_rollout_choice" in resolved:
        if not _resolve_component_policy_names(problem, hybrid_rollout_rollout_policy_choices):
            raise ValueError("Family 'hybrid_rollout_choice' requires at least one rollout-policy choice")
        if not _resolve_component_policy_names(problem, hybrid_rollout_candidate_policy_choices):
            raise ValueError("Family 'hybrid_rollout_choice' requires at least one candidate-policy choice")

    return resolved


def make_policy_automl_smac3_configspace(
    problem: PolicyAutoMLDemoProblem,
    *,
    families: Sequence[str] | None = None,
    prefit_policy_names: Sequence[str] | None = None,
    target_level_bounds: tuple[int, int] | None = None,
    theta_bounds: tuple[float, float] | None = None,
    H_cfa_bounds: tuple[int, int] = (1, 6),
    H_mpc_bounds: tuple[int, int] = (1, 6),
    horizon_mcts_bounds: tuple[int, int] = (2, 6),
    n_simulations_mcts_bounds: tuple[int, int] = (16, 96),
    uct_c_mcts_bounds: tuple[float, float] = (0.3, 2.5),
    direct_action_pfa_regressor_choices: Sequence[str] | None = None,
    direct_action_pfa_train_episodes_bounds: tuple[int, int] = (80, 200),
    direct_action_pfa_train_iter_bounds: tuple[int, int] = (1, 3),
    direct_action_pfa_tree_max_depth_bounds: tuple[int, int] = (3, 10),
    direct_action_pfa_elastic_alpha_bounds: tuple[float, float] = (0.001, 1.0),
    direct_action_pfa_elastic_l1_ratio_bounds: tuple[float, float] = (0.1, 0.9),
    hybrid_rollout_rollout_policy_choices: Sequence[str] | None = None,
    hybrid_rollout_candidate_policy_choices: Sequence[str] | None = None,
) -> Any:
    """Create a conditional ConfigSpace search space for real SMAC3 runs."""

    ConfigurationSpace, EqualsCondition, AndConjunction, _, _ = _require_smac3()

    max_target = int(max(0, problem.s_max))
    low_target, high_target = target_level_bounds or (0, max_target)
    low_theta, high_theta = theta_bounds or (0.0, float(problem.x_max))
    resolved_families = _resolve_smac_families(
        problem,
        families,
        prefit_policy_names=prefit_policy_names,
        hybrid_rollout_rollout_policy_choices=hybrid_rollout_rollout_policy_choices,
        hybrid_rollout_candidate_policy_choices=hybrid_rollout_candidate_policy_choices,
    )
    resolved_direct_action_pfa_choices = tuple(
        str(mode) for mode in (direct_action_pfa_regressor_choices or ("decision_tree", "elastic", "mlp"))
    )
    resolved_hybrid_rollout_rollout_choices = tuple(
        str(name) for name in _resolve_component_policy_names(problem, hybrid_rollout_rollout_policy_choices)
    ) if "hybrid_rollout_choice" in resolved_families else tuple()
    resolved_hybrid_rollout_candidate_choices = tuple(
        str(name) for name in _resolve_component_policy_names(problem, hybrid_rollout_candidate_policy_choices)
    ) if "hybrid_rollout_choice" in resolved_families else tuple()
    space_def: dict[str, Any] = {
        "family": resolved_families,
        "target_level": _configspace_range_or_singleton(int(low_target), int(high_target)),
        "theta": _configspace_range_or_singleton(float(low_theta), float(high_theta)),
        "H_cfa": _configspace_range_or_singleton(int(H_cfa_bounds[0]), int(H_cfa_bounds[1])),
        "H_mpc": _configspace_range_or_singleton(int(H_mpc_bounds[0]), int(H_mpc_bounds[1])),
        "horizon_mcts": _configspace_range_or_singleton(int(horizon_mcts_bounds[0]), int(horizon_mcts_bounds[1])),
        "n_simulations_mcts": _configspace_range_or_singleton(
            int(n_simulations_mcts_bounds[0]), int(n_simulations_mcts_bounds[1])
        ),
        "uct_c_mcts": _configspace_range_or_singleton(float(uct_c_mcts_bounds[0]), float(uct_c_mcts_bounds[1])),
    }
    if "direct_action_pfa" in resolved_families:
        space_def["regressor_mode"] = list(resolved_direct_action_pfa_choices)
        space_def["train_n_episodes"] = _configspace_range_or_singleton(
            int(direct_action_pfa_train_episodes_bounds[0]),
            int(direct_action_pfa_train_episodes_bounds[1]),
        )
        space_def["train_n_iter"] = _configspace_range_or_singleton(
            int(direct_action_pfa_train_iter_bounds[0]),
            int(direct_action_pfa_train_iter_bounds[1]),
        )
        if "decision_tree" in resolved_direct_action_pfa_choices:
            space_def["tree_max_depth"] = _configspace_range_or_singleton(
                int(direct_action_pfa_tree_max_depth_bounds[0]),
                int(direct_action_pfa_tree_max_depth_bounds[1]),
            )
        if "elastic" in resolved_direct_action_pfa_choices:
            space_def["elastic_alpha"] = _configspace_range_or_singleton(
                float(direct_action_pfa_elastic_alpha_bounds[0]),
                float(direct_action_pfa_elastic_alpha_bounds[1]),
            )
            space_def["elastic_l1_ratio"] = _configspace_range_or_singleton(
                float(direct_action_pfa_elastic_l1_ratio_bounds[0]),
                float(direct_action_pfa_elastic_l1_ratio_bounds[1]),
            )
    if "hybrid_rollout_choice" in resolved_families:
        space_def["rollout_policy_key"] = list(resolved_hybrid_rollout_rollout_choices)
        space_def["candidate_policy_key"] = list(resolved_hybrid_rollout_candidate_choices)
    if "prefit_policy" in resolved_families:
        space_def["prefit_name"] = list(_resolve_prefit_policy_names(problem, prefit_policy_names))

    cs = ConfigurationSpace(space_def)

    conditions = []
    if "order_up_to" in resolved_families:
        conditions.append(EqualsCondition(cs["target_level"], cs["family"], "order_up_to"))
    if "pfa_blackbox" in resolved_families:
        conditions.append(EqualsCondition(cs["theta"], cs["family"], "pfa_blackbox"))
    if "cfa_milp" in resolved_families:
        conditions.append(EqualsCondition(cs["H_cfa"], cs["family"], "cfa_milp"))
    if "dla_mpc" in resolved_families:
        conditions.append(EqualsCondition(cs["H_mpc"], cs["family"], "dla_mpc"))
    if "dla_mcts" in resolved_families:
        conditions.extend(
            [
                EqualsCondition(cs["horizon_mcts"], cs["family"], "dla_mcts"),
                EqualsCondition(cs["n_simulations_mcts"], cs["family"], "dla_mcts"),
                EqualsCondition(cs["uct_c_mcts"], cs["family"], "dla_mcts"),
            ]
        )
    if "direct_action_pfa" in resolved_families:
        conditions.extend(
            [
                EqualsCondition(cs["regressor_mode"], cs["family"], "direct_action_pfa"),
                EqualsCondition(cs["train_n_episodes"], cs["family"], "direct_action_pfa"),
                EqualsCondition(cs["train_n_iter"], cs["family"], "direct_action_pfa"),
            ]
        )
        if "tree_max_depth" in cs:
            conditions.append(
                AndConjunction(
                    EqualsCondition(cs["tree_max_depth"], cs["family"], "direct_action_pfa"),
                    EqualsCondition(cs["tree_max_depth"], cs["regressor_mode"], "decision_tree"),
                )
            )
        if "elastic_alpha" in cs:
            conditions.append(
                AndConjunction(
                    EqualsCondition(cs["elastic_alpha"], cs["family"], "direct_action_pfa"),
                    EqualsCondition(cs["elastic_alpha"], cs["regressor_mode"], "elastic"),
                )
            )
        if "elastic_l1_ratio" in cs:
            conditions.append(
                AndConjunction(
                    EqualsCondition(cs["elastic_l1_ratio"], cs["family"], "direct_action_pfa"),
                    EqualsCondition(cs["elastic_l1_ratio"], cs["regressor_mode"], "elastic"),
                )
            )
    if "hybrid_rollout_choice" in resolved_families:
        conditions.extend(
            [
                EqualsCondition(cs["rollout_policy_key"], cs["family"], "hybrid_rollout_choice"),
                EqualsCondition(cs["candidate_policy_key"], cs["family"], "hybrid_rollout_choice"),
            ]
        )
    if "prefit_policy" in resolved_families:
        conditions.append(EqualsCondition(cs["prefit_name"], cs["family"], "prefit_policy"))

    if conditions:
        cs.add(conditions)
    return cs


def smac3_config_to_policy_config(
    config: Mapping[str, Any] | Any,
    problem: PolicyAutoMLDemoProblem,
    *,
    prefit_policy_names: Sequence[str] | None = None,
    direct_action_pfa_H: int = 5,
    direct_action_pfa_feature_fn_key: str | None = None,
    hybrid_rollout_rollout_policy_choices: Sequence[str] | None = None,
    hybrid_rollout_candidate_policy_choices: Sequence[str] | None = None,
    hybrid_rollout_H: int = 3,
    hybrid_rollout_n_rollouts: int = 5,
    hybrid_rollout_candidate_radius_steps: int = 1,
    hybrid_rollout_include_full_action_grid: bool = False,
    hybrid_rollout_decision_seed: int = 2026,
) -> dict[str, Any]:
    """Translate one SMAC3/ConfigSpace configuration into a repo-native policy config."""

    values = _config_like_to_dict(config)
    family = str(values["family"])

    if family == "order_up_to":
        target = _clip_and_round_to_grid(
            float(values["target_level"]),
            lower=0.0,
            upper=float(problem.s_max),
            step=int(problem.dx),
        )
        target_i = int(np.round(target))
        return {
            "label": f"OrderUpTo[S={target_i}]",
            "family": family,
            "target_level": target_i,
        }

    if family == "pfa_blackbox":
        theta = float(np.clip(float(values["theta"]), 0.0, float(problem.x_max)))
        return {
            "label": f"BlackboxPFA[theta={theta:.1f}]",
            "family": family,
            "theta": theta,
            "theta_min": 0.0,
            "theta_max": float(problem.x_max),
        }

    if family == "cfa_milp":
        H = int(max(1, int(np.round(float(values["H_cfa"])))))
        return {
            "label": f"CFA_MILP[H={H}]",
            "family": family,
            "H": H,
        }

    if family == "dla_mpc":
        H = int(max(1, int(np.round(float(values["H_mpc"])))))
        return {
            "label": f"DLA_MPC[H={H}]",
            "family": family,
            "H": H,
        }

    if family == "dla_mcts":
        horizon = int(max(1, int(np.round(float(values["horizon_mcts"])))))
        n_simulations = int(max(1, int(np.round(float(values["n_simulations_mcts"])))))
        uct_c = float(max(1e-6, float(values["uct_c_mcts"])))
        return {
            "label": f"DLA_MCTS[h={horizon}, sims={n_simulations}, c={uct_c:.3f}]",
            "family": family,
            "horizon": horizon,
            "n_simulations": n_simulations,
            "uct_c": uct_c,
            "planning_seed": 123,
        }

    if family == "direct_action_pfa":
        regressor_mode = str(values["regressor_mode"])
        train_n_episodes = int(max(1, int(np.round(float(values["train_n_episodes"])))))
        train_n_iter = int(max(1, int(np.round(float(values["train_n_iter"])))))
        H = int(max(1, int(direct_action_pfa_H)))
        cfg = {
            "family": family,
            "regressor_mode": regressor_mode,
            "train_n_episodes": train_n_episodes,
            "train_n_iter": train_n_iter,
            "H": H,
        }
        if direct_action_pfa_feature_fn_key is not None:
            cfg["feature_fn_key"] = str(direct_action_pfa_feature_fn_key)
        label_parts = [regressor_mode, f"neps={train_n_episodes}", f"niter={train_n_iter}"]
        if regressor_mode == "decision_tree" and values.get("tree_max_depth") is not None:
            tree_max_depth = int(max(1, int(np.round(float(values["tree_max_depth"])))))
            cfg["tree_max_depth"] = tree_max_depth
            label_parts.append(f"depth={tree_max_depth}")
        if regressor_mode == "elastic":
            if values.get("elastic_alpha") is not None:
                elastic_alpha = float(values["elastic_alpha"])
                cfg["elastic_alpha"] = elastic_alpha
                label_parts.append(f"alpha={elastic_alpha:.3g}")
            if values.get("elastic_l1_ratio") is not None:
                elastic_l1_ratio = float(values["elastic_l1_ratio"])
                cfg["elastic_l1_ratio"] = elastic_l1_ratio
                label_parts.append(f"l1={elastic_l1_ratio:.3g}")
        cfg["label"] = f"DirectActionPFA[{', '.join(label_parts)}]"
        return cfg

    if family == "hybrid_rollout_choice":
        rollout_choices = _resolve_component_policy_names(problem, hybrid_rollout_rollout_policy_choices)
        candidate_choices = _resolve_component_policy_names(problem, hybrid_rollout_candidate_policy_choices)
        rollout_policy_key = str(values["rollout_policy_key"])
        candidate_policy_key = str(values["candidate_policy_key"])
        if rollout_policy_key not in rollout_choices:
            raise ValueError(f"Unknown rollout_policy_key for SMAC3 config: {rollout_policy_key}")
        if candidate_policy_key not in candidate_choices:
            raise ValueError(f"Unknown candidate_policy_key for SMAC3 config: {candidate_policy_key}")
        return {
            "label": f"HybridRollout[rollout={rollout_policy_key}, candidate={candidate_policy_key}]",
            "family": family,
            "rollout_policy_key": rollout_policy_key,
            "candidate_policy_key": candidate_policy_key,
            "H": int(max(1, int(hybrid_rollout_H))),
            "n_rollouts": int(max(1, int(hybrid_rollout_n_rollouts))),
            "candidate_radius_steps": int(max(0, int(hybrid_rollout_candidate_radius_steps))),
            "include_full_action_grid": bool(hybrid_rollout_include_full_action_grid),
            "decision_seed": int(hybrid_rollout_decision_seed),
        }

    if family == "prefit_policy":
        names = _resolve_prefit_policy_names(problem, prefit_policy_names)
        name = str(values["prefit_name"])
        if name not in names:
            raise ValueError(f"Unknown prefit policy name for SMAC3 config: {name}")
        return {
            "label": f"PrefitPolicy[{name}]",
            "family": family,
            "prefit_name": name,
        }

    raise ValueError(f"Unsupported SMAC3 policy family: {family}")


def _default_forecaster(
    problem: PolicyAutoMLDemoProblem,
    *,
    role: str,
) -> Any:
    registry = dict(problem.forecaster_registry or {})

    if role == "cfa_milp":
        key = problem.default_cfa_forecaster_key
    elif role == "dla_mpc":
        key = problem.default_dla_forecaster_key or problem.default_cfa_forecaster_key
    else:
        key = None

    if key is not None and key in registry:
        return registry[key]

    return ExogenousAwareMeanForecaster(exogenous_model=problem.exogenous_model)


def _build_direct_action_pfa_regressor(
    config: Mapping[str, Any],
    *,
    random_state: int,
):
    return _build_supervised_regressor(
        mode=str(config["regressor_mode"]),
        ridge_alpha=1.0,
        elastic_alpha=float(config.get("elastic_alpha", 1.0)),
        elastic_l1_ratio=float(config.get("elastic_l1_ratio", 0.5)),
        tree_max_depth=int(config.get("tree_max_depth", 4)),
        tree_learning_rate=float(config.get("tree_learning_rate", 0.08)),
        tree_max_iter=int(config.get("tree_max_iter", 400)),
        tree_min_samples_leaf=int(config.get("tree_min_samples_leaf", 20)),
        mlp_hidden=tuple(int(h) for h in config.get("mlp_hidden", (64, 64))),
        mlp_max_iter=int(config.get("mlp_max_iter", 500)),
        random_state=int(random_state),
    )


def _train_direct_action_pfa_from_config(
    config: Mapping[str, Any],
    problem: PolicyAutoMLDemoProblem,
    *,
    training_seed0: int,
    train_info: Mapping[str, Any] | None = None,
) -> DirectActionPFA:
    regressor = _build_direct_action_pfa_regressor(
        config,
        random_state=int(config.get("random_state", training_seed0)),
    )
    policy = DirectActionPFA(
        system=problem.system,
        regressor=regressor,
        x_max=float(problem.x_max),
        dx=int(problem.dx),
        H=int(config.get("H", 5)),
        feature_fn=_resolve_feature_fn(problem, str(config["feature_fn_key"])) if config.get("feature_fn_key") is not None else None,
    )
    policy.fit_via_rollout_improvement(
        S0=problem.S0,
        T=int(problem.T),
        n_episodes=int(config.get("train_n_episodes", 80)),
        seed0=int(training_seed0),
        n_iter=int(config.get("train_n_iter", 1)),
        info=_resolve_train_info(problem, train_info),
    )
    return policy


def _build_hybrid_rollout_choice_from_config(
    config: Mapping[str, Any],
    problem: PolicyAutoMLDemoProblem,
) -> HybridDlaRolloutComposablePolicy:
    registry = dict(problem.component_policy_registry or {})
    rollout_policy_key = str(config["rollout_policy_key"])
    candidate_policy_key = str(config["candidate_policy_key"])
    if rollout_policy_key not in registry:
        raise ValueError(f"Unknown rollout policy key: {rollout_policy_key}")
    if candidate_policy_key not in registry:
        raise ValueError(f"Unknown candidate policy key: {candidate_policy_key}")
    return HybridDlaRolloutComposablePolicy(
        system=problem.system,
        rollout_policy=registry[rollout_policy_key],
        candidate_policy=registry[candidate_policy_key],
        H=int(config.get("H", 3)),
        n_rollouts=int(config.get("n_rollouts", 5)),
        dx=int(problem.dx),
        x_max=int(problem.x_max),
        decision_seed=int(config.get("decision_seed", 2026)),
        candidate_radius_steps=int(config.get("candidate_radius_steps", 1)),
        include_full_action_grid=bool(config.get("include_full_action_grid", False)),
    )


def build_policy_from_config(config: Mapping[str, Any], problem: PolicyAutoMLDemoProblem) -> Policy:
    """Instantiate one policy candidate from a simple config dict."""

    family = str(config["family"])
    x_max = int(problem.x_max)
    dx = int(problem.dx)
    s_max = int(problem.s_max)
    system = problem.system

    if family == "order_up_to":
        return OrderUpToPolicy(
            target_level=float(config["target_level"]),
            x_max=x_max,
            dx=dx,
        )

    if family == "pfa_blackbox":
        return OrderUpToBlackboxPFA(
            system=system,
            theta=np.array([float(config["theta"])], dtype=float),
            x_max=x_max,
            dx=dx,
            theta_min=float(config.get("theta_min", 0.0)),
            theta_max=float(config.get("theta_max", x_max)),
        )

    if family == "cfa_milp":
        return OrderMilpCfaPolicy(
            forecaster=_default_forecaster(problem, role="cfa_milp"),
            H=int(config["H"]),
            x_max=x_max,
            dx=dx,
            S_max=s_max,
            p=float(problem.p),
            c=float(problem.c),
            h=float(problem.h),
            b=float(problem.b),
            K=float(problem.K),
        )

    if family == "dla_mpc":
        return DlaMpcMilpPolicy(
            model=system,
            H=int(config["H"]),
            x_max=x_max,
            dx=dx,
            S_max=s_max,
            p=float(problem.p),
            c=float(problem.c),
            h=float(problem.h),
            b=float(problem.b),
            K=float(problem.K),
            forecaster=_default_forecaster(problem, role="dla_mpc"),
        )

    if family == "dla_mcts":
        return DlaMctsUctPolicy(
            model=system,
            horizon=int(config["horizon"]),
            n_simulations=int(config["n_simulations"]),
            uct_c=float(config["uct_c"]),
            x_max=x_max,
            dx=dx,
            planning_seed=int(config.get("planning_seed", 0)),
        )

    if family == "direct_action_pfa":
        return _train_direct_action_pfa_from_config(
            config,
            problem,
            training_seed0=int(config.get("training_seed0", 1234)),
            train_info=problem.train_info,
        )

    if family == "hybrid_rollout_choice":
        return _build_hybrid_rollout_choice_from_config(config, problem)

    if family == "prefit_policy":
        registry = dict(problem.prefit_policy_registry or {})
        name = str(config["prefit_name"])
        if name not in registry:
            raise ValueError(f"Unknown prefit policy name: {name}")
        return registry[name]

    raise ValueError(f"Unsupported policy family: {family}")


def materialize_policy_from_config(
    config: Mapping[str, Any],
    problem: PolicyAutoMLDemoProblem,
    *,
    training_seed0: int = 1234,
    train_info: Mapping[str, Any] | None = None,
    policy_cache: MutableMapping[Any, Policy] | None = None,
) -> Policy:
    """Instantiate or train one policy candidate, with optional config-based caching."""

    resolved_train_info = _resolve_train_info(problem, train_info)
    cache_key = _policy_config_cache_key(
        config,
        training_seed0=int(training_seed0),
        train_info=resolved_train_info,
    )
    if policy_cache is not None and cache_key in policy_cache:
        return policy_cache[cache_key]

    family = str(config["family"])
    if family == "direct_action_pfa":
        policy = _train_direct_action_pfa_from_config(
            config,
            problem,
            training_seed0=int(training_seed0),
            train_info=resolved_train_info,
        )
    elif family == "hybrid_rollout_choice":
        policy = _build_hybrid_rollout_choice_from_config(config, problem)
    else:
        policy = build_policy_from_config(config, problem)

    if policy_cache is not None:
        policy_cache[cache_key] = policy
    return policy


def build_exact_dp_benchmark(
    problem: PolicyAutoMLDemoProblem,
    *,
    terminal_cost_per_unit: float = 0.0,
    expectation_mode: str = "truncation",
) -> Policy:
    """Build an exact DP benchmark for supported demo problems."""

    if isinstance(problem.exogenous_model, PoissonConstantDemand):
        solver = DynamicProgrammingSolver1D(
            problem.exogenous_model,
            T=int(problem.T),
            S_max=int(problem.s_max),
            x_max=int(problem.x_max),
            dx=int(problem.dx),
            p=float(problem.p),
            c=float(problem.c),
            h=float(problem.h),
            b=float(problem.b),
            terminal_cost_per_unit=float(terminal_cost_per_unit),
            expectation_mode=str(expectation_mode),
        )
        sol = solver.solve()
        return solver.to_policy(sol)

    if isinstance(problem.exogenous_model, PoissonMultiRegimeDemand):
        solver = DPSolverMultiRegimeScenarios(
            problem.exogenous_model,
            T=int(problem.T),
            S_max=int(problem.s_max),
            x_max=int(problem.x_max),
            dx=int(problem.dx),
            p=float(problem.p),
            c=float(problem.c),
            h=float(problem.h),
            b=float(problem.b),
            terminal_cost_per_unit=float(terminal_cost_per_unit),
            gamma=1.0,
            K=15,
        )
        sol = solver.solve()
        return solver.to_policy(sol)

    raise ValueError(f"Unsupported exogenous model for exact DP benchmark: {type(problem.exogenous_model).__name__}")


def objective_mean_total_cost(
    config: Mapping[str, Any],
    problem: PolicyAutoMLDemoProblem,
    *,
    n_episodes: int = 80,
    seed0: int = 20260209,
    eval_info: Mapping[str, Any] | None = None,
) -> float:
    """Black-box objective that a SMAC-style optimizer would minimize."""

    res = evaluate_policy_config(
        config,
        problem,
        n_episodes=int(n_episodes),
        seed0=int(seed0),
        eval_info=eval_info,
    )
    return float(res.mean_total_cost)


def make_policy_automl_smac3_target_function(
    problem: PolicyAutoMLDemoProblem,
    *,
    n_episodes: int = 80,
    seed0: int = 20260209,
    training_seed0: int = 20265209,
    runtime_penalty: float = 0.0,
    eval_info: Mapping[str, Any] | None = None,
    train_info: Mapping[str, Any] | None = None,
    policy_cache: MutableMapping[Any, Policy] | None = None,
    prefit_policy_names: Sequence[str] | None = None,
    direct_action_pfa_H: int = 5,
    direct_action_pfa_feature_fn_key: str | None = None,
    hybrid_rollout_rollout_policy_choices: Sequence[str] | None = None,
    hybrid_rollout_candidate_policy_choices: Sequence[str] | None = None,
    hybrid_rollout_H: int = 3,
    hybrid_rollout_n_rollouts: int = 5,
    hybrid_rollout_candidate_radius_steps: int = 1,
    hybrid_rollout_include_full_action_grid: bool = False,
    hybrid_rollout_decision_seed: int = 2026,
) -> Callable[[Any, int], float]:
    """Return a real SMAC3 target function over the repo's policy objective.

    The target is deterministic on purpose: every configuration is evaluated on
    the same strict-CRN seed bundle so that SMAC compares policies fairly.
    """

    def _target(config: Any, seed: int = 0) -> float:
        _ = int(seed)
        policy_config = smac3_config_to_policy_config(
            config,
            problem,
            prefit_policy_names=prefit_policy_names,
            direct_action_pfa_H=direct_action_pfa_H,
            direct_action_pfa_feature_fn_key=direct_action_pfa_feature_fn_key,
            hybrid_rollout_rollout_policy_choices=hybrid_rollout_rollout_policy_choices,
            hybrid_rollout_candidate_policy_choices=hybrid_rollout_candidate_policy_choices,
            hybrid_rollout_H=hybrid_rollout_H,
            hybrid_rollout_n_rollouts=hybrid_rollout_n_rollouts,
            hybrid_rollout_candidate_radius_steps=hybrid_rollout_candidate_radius_steps,
            hybrid_rollout_include_full_action_grid=hybrid_rollout_include_full_action_grid,
            hybrid_rollout_decision_seed=hybrid_rollout_decision_seed,
        )
        res = evaluate_policy_config(
            policy_config,
            problem,
            n_episodes=int(n_episodes),
            seed0=int(seed0),
            training_seed0=int(training_seed0),
            runtime_penalty=float(runtime_penalty),
            eval_info=eval_info,
            train_info=train_info,
            policy_cache=policy_cache,
        )
        return float(res.score)

    return _target


def run_policy_automl_smac3(
    problem: PolicyAutoMLDemoProblem,
    *,
    n_trials: int = 24,
    scenario_seed: int = 123,
    n_episodes: int = 80,
    seed0: int = 20260209,
    training_seed0: int | None = None,
    runtime_penalty: float = 0.0,
    eval_info: Mapping[str, Any] | None = None,
    train_info: Mapping[str, Any] | None = None,
    overwrite: bool = True,
    policy_menu: Mapping[str, Any] | None = None,
    families: Sequence[str] | None = None,
    prefit_policy_names: Sequence[str] | None = None,
    target_level_bounds: tuple[int, int] | None = None,
    theta_bounds: tuple[float, float] | None = None,
    H_cfa_bounds: tuple[int, int] = (1, 6),
    H_mpc_bounds: tuple[int, int] = (1, 6),
    horizon_mcts_bounds: tuple[int, int] = (2, 6),
    n_simulations_mcts_bounds: tuple[int, int] = (16, 96),
    uct_c_mcts_bounds: tuple[float, float] = (0.3, 2.5),
    direct_action_pfa_regressor_choices: Sequence[str] | None = None,
    direct_action_pfa_train_episodes_bounds: tuple[int, int] = (80, 200),
    direct_action_pfa_train_iter_bounds: tuple[int, int] = (1, 3),
    direct_action_pfa_tree_max_depth_bounds: tuple[int, int] = (3, 10),
    direct_action_pfa_elastic_alpha_bounds: tuple[float, float] = (0.001, 1.0),
    direct_action_pfa_elastic_l1_ratio_bounds: tuple[float, float] = (0.1, 0.9),
    direct_action_pfa_H: int = 5,
    direct_action_pfa_feature_fn_key: str | None = None,
    hybrid_rollout_rollout_policy_choices: Sequence[str] | None = None,
    hybrid_rollout_candidate_policy_choices: Sequence[str] | None = None,
    hybrid_rollout_H: int = 3,
    hybrid_rollout_n_rollouts: int = 5,
    hybrid_rollout_candidate_radius_steps: int = 1,
    hybrid_rollout_include_full_action_grid: bool = False,
    hybrid_rollout_decision_seed: int = 2026,
) -> PolicyAutoMLSmacResult:
    """Run an actual SMAC3 search over a conditional policy configuration space."""

    # `policy_menu` is the preferred notebook-facing interface for EIC/EIMR-style
    # teaching notebooks. It may contain:
    # - `tunable_families`: searchable policy families with family-specific bounds
    # - `prefit_variants`: already-trained policy artifacts selectable by name
    #
    # Example:
    # {
    #     "tunable_families": {
    #         "order_up_to": {"bounds": (0, 480)},
    #         "cfa_milp": {"bounds": (1, 5)},
    #         "dla_mpc": {"bounds": (1, 5)},
    #     },
    #     "trainable_families": {
    #         "direct_action_pfa": {
    #             "regressor_modes": ("decision_tree", "elastic", "mlp"),
    #             "train_n_episodes_bounds": (80, 200),
    #             "train_n_iter_bounds": (1, 3),
    #             "H": 5,
    #             "model_hyperparameters": {
    #                 "decision_tree": {"tree_max_depth_bounds": (3, 10)},
    #                 "elastic": {
    #                     "elastic_alpha_bounds": (0.001, 1.0),
    #                     "elastic_l1_ratio_bounds": (0.1, 0.9),
    #                 },
    #             },
    #         },
    #     },
    #     "composite_families": {
    #         "hybrid_rollout_choice": {
    #             "rollout_policy_choices": ("OrderUpTo_350", "DirectActionPFA_Tree"),
    #             "candidate_policy_choices": ("OrderUpTo_350", "DirectActionPFA_Tree"),
    #             "fixed_params": {
    #                 "H": 3,
    #                 "n_rollouts": 5,
    #                 "candidate_radius_steps": 1,
    #                 "include_full_action_grid": False,
    #                 "decision_seed": 2026,
    #             },
    #         },
    #     },
    #     "prefit_variants": {
    #         "DirectActionPFA_Tree": {"policy_class": "DirectActionPFA"},
    #     },
    # }

    _, _, _, HyperparameterOptimizationFacade, Scenario = _require_smac3()

    menu_cfg = smac_policy_menu_to_search_space_config(policy_menu) if policy_menu is not None else {}
    resolved_families = families if families is not None else menu_cfg.get("families")
    resolved_prefit_names = prefit_policy_names if prefit_policy_names is not None else menu_cfg.get("prefit_policy_names")
    resolved_target_bounds = target_level_bounds if target_level_bounds is not None else menu_cfg.get("target_level_bounds")
    resolved_theta_bounds = theta_bounds if theta_bounds is not None else menu_cfg.get("theta_bounds")
    resolved_H_cfa_bounds = H_cfa_bounds if H_cfa_bounds != (1, 6) else menu_cfg.get("H_cfa_bounds", H_cfa_bounds)
    resolved_H_mpc_bounds = H_mpc_bounds if H_mpc_bounds != (1, 6) else menu_cfg.get("H_mpc_bounds", H_mpc_bounds)
    resolved_horizon_mcts_bounds = (
        horizon_mcts_bounds
        if horizon_mcts_bounds != (2, 6)
        else menu_cfg.get("horizon_mcts_bounds", horizon_mcts_bounds)
    )
    resolved_n_sims_bounds = (
        n_simulations_mcts_bounds
        if n_simulations_mcts_bounds != (16, 96)
        else menu_cfg.get("n_simulations_mcts_bounds", n_simulations_mcts_bounds)
    )
    resolved_uct_c_bounds = (
        uct_c_mcts_bounds
        if uct_c_mcts_bounds != (0.3, 2.5)
        else menu_cfg.get("uct_c_mcts_bounds", uct_c_mcts_bounds)
    )
    resolved_direct_action_pfa_choices = (
        direct_action_pfa_regressor_choices
        if direct_action_pfa_regressor_choices is not None
        else menu_cfg.get("direct_action_pfa_regressor_choices")
    )
    resolved_direct_action_pfa_train_episodes_bounds = (
        direct_action_pfa_train_episodes_bounds
        if direct_action_pfa_train_episodes_bounds != (80, 200)
        else menu_cfg.get("direct_action_pfa_train_episodes_bounds", direct_action_pfa_train_episodes_bounds)
    )
    resolved_direct_action_pfa_train_iter_bounds = (
        direct_action_pfa_train_iter_bounds
        if direct_action_pfa_train_iter_bounds != (1, 3)
        else menu_cfg.get("direct_action_pfa_train_iter_bounds", direct_action_pfa_train_iter_bounds)
    )
    resolved_direct_action_pfa_tree_max_depth_bounds = (
        direct_action_pfa_tree_max_depth_bounds
        if direct_action_pfa_tree_max_depth_bounds != (3, 10)
        else menu_cfg.get("direct_action_pfa_tree_max_depth_bounds", direct_action_pfa_tree_max_depth_bounds)
    )
    resolved_direct_action_pfa_elastic_alpha_bounds = (
        direct_action_pfa_elastic_alpha_bounds
        if direct_action_pfa_elastic_alpha_bounds != (0.001, 1.0)
        else menu_cfg.get("direct_action_pfa_elastic_alpha_bounds", direct_action_pfa_elastic_alpha_bounds)
    )
    resolved_direct_action_pfa_elastic_l1_ratio_bounds = (
        direct_action_pfa_elastic_l1_ratio_bounds
        if direct_action_pfa_elastic_l1_ratio_bounds != (0.1, 0.9)
        else menu_cfg.get("direct_action_pfa_elastic_l1_ratio_bounds", direct_action_pfa_elastic_l1_ratio_bounds)
    )
    resolved_direct_action_pfa_H = (
        int(direct_action_pfa_H)
        if int(direct_action_pfa_H) != 5
        else int(menu_cfg.get("direct_action_pfa_H", direct_action_pfa_H))
    )
    resolved_direct_action_pfa_feature_fn_key = (
        str(direct_action_pfa_feature_fn_key)
        if direct_action_pfa_feature_fn_key is not None
        else (
            str(menu_cfg["direct_action_pfa_feature_fn_key"])
            if menu_cfg.get("direct_action_pfa_feature_fn_key") is not None
            else None
        )
    )
    resolved_hybrid_rollout_rollout_choices = (
        hybrid_rollout_rollout_policy_choices
        if hybrid_rollout_rollout_policy_choices is not None
        else menu_cfg.get("hybrid_rollout_rollout_policy_choices")
    )
    resolved_hybrid_rollout_candidate_choices = (
        hybrid_rollout_candidate_policy_choices
        if hybrid_rollout_candidate_policy_choices is not None
        else menu_cfg.get("hybrid_rollout_candidate_policy_choices")
    )
    resolved_hybrid_rollout_H = (
        int(hybrid_rollout_H)
        if int(hybrid_rollout_H) != 3
        else int(menu_cfg.get("hybrid_rollout_H", hybrid_rollout_H))
    )
    resolved_hybrid_rollout_n_rollouts = (
        int(hybrid_rollout_n_rollouts)
        if int(hybrid_rollout_n_rollouts) != 5
        else int(menu_cfg.get("hybrid_rollout_n_rollouts", hybrid_rollout_n_rollouts))
    )
    resolved_hybrid_rollout_candidate_radius_steps = (
        int(hybrid_rollout_candidate_radius_steps)
        if int(hybrid_rollout_candidate_radius_steps) != 1
        else int(menu_cfg.get("hybrid_rollout_candidate_radius_steps", hybrid_rollout_candidate_radius_steps))
    )
    resolved_hybrid_rollout_include_full_action_grid = (
        bool(hybrid_rollout_include_full_action_grid)
        if bool(hybrid_rollout_include_full_action_grid) is not False
        else bool(menu_cfg.get("hybrid_rollout_include_full_action_grid", hybrid_rollout_include_full_action_grid))
    )
    resolved_hybrid_rollout_decision_seed = (
        int(hybrid_rollout_decision_seed)
        if int(hybrid_rollout_decision_seed) != 2026
        else int(menu_cfg.get("hybrid_rollout_decision_seed", hybrid_rollout_decision_seed))
    )
    resolved_training_seed0 = int(seed0) + 50_000 if training_seed0 is None else int(training_seed0)
    resolved_train_info = dict(problem.train_info or problem.eval_info) if train_info is None else dict(train_info)
    policy_cache: dict[Any, Policy] = {}

    configspace = make_policy_automl_smac3_configspace(
        problem,
        families=resolved_families,
        prefit_policy_names=resolved_prefit_names,
        hybrid_rollout_rollout_policy_choices=resolved_hybrid_rollout_rollout_choices,
        hybrid_rollout_candidate_policy_choices=resolved_hybrid_rollout_candidate_choices,
        target_level_bounds=resolved_target_bounds,
        theta_bounds=resolved_theta_bounds,
        H_cfa_bounds=resolved_H_cfa_bounds,
        H_mpc_bounds=resolved_H_mpc_bounds,
        horizon_mcts_bounds=resolved_horizon_mcts_bounds,
        n_simulations_mcts_bounds=resolved_n_sims_bounds,
        uct_c_mcts_bounds=resolved_uct_c_bounds,
        direct_action_pfa_regressor_choices=resolved_direct_action_pfa_choices,
        direct_action_pfa_train_episodes_bounds=resolved_direct_action_pfa_train_episodes_bounds,
        direct_action_pfa_train_iter_bounds=resolved_direct_action_pfa_train_iter_bounds,
        direct_action_pfa_tree_max_depth_bounds=resolved_direct_action_pfa_tree_max_depth_bounds,
        direct_action_pfa_elastic_alpha_bounds=resolved_direct_action_pfa_elastic_alpha_bounds,
        direct_action_pfa_elastic_l1_ratio_bounds=resolved_direct_action_pfa_elastic_l1_ratio_bounds,
    )
    scenario = Scenario(
        configspace,
        deterministic=True,
        n_trials=int(n_trials),
        seed=int(scenario_seed),
    )
    target_function = make_policy_automl_smac3_target_function(
        problem,
        n_episodes=int(n_episodes),
        seed0=int(seed0),
        training_seed0=resolved_training_seed0,
        runtime_penalty=float(runtime_penalty),
        eval_info=eval_info,
        train_info=resolved_train_info,
        policy_cache=policy_cache,
        prefit_policy_names=resolved_prefit_names,
        direct_action_pfa_H=resolved_direct_action_pfa_H,
        direct_action_pfa_feature_fn_key=resolved_direct_action_pfa_feature_fn_key,
        hybrid_rollout_rollout_policy_choices=resolved_hybrid_rollout_rollout_choices,
        hybrid_rollout_candidate_policy_choices=resolved_hybrid_rollout_candidate_choices,
        hybrid_rollout_H=resolved_hybrid_rollout_H,
        hybrid_rollout_n_rollouts=resolved_hybrid_rollout_n_rollouts,
        hybrid_rollout_candidate_radius_steps=resolved_hybrid_rollout_candidate_radius_steps,
        hybrid_rollout_include_full_action_grid=resolved_hybrid_rollout_include_full_action_grid,
        hybrid_rollout_decision_seed=resolved_hybrid_rollout_decision_seed,
    )

    smac = HyperparameterOptimizationFacade(
        scenario,
        target_function,
        overwrite=bool(overwrite),
        logging_level=False,
    )
    incumbent = smac.optimize()

    incumbent_config = smac3_config_to_policy_config(
        incumbent,
        problem,
        prefit_policy_names=resolved_prefit_names,
        direct_action_pfa_H=resolved_direct_action_pfa_H,
        direct_action_pfa_feature_fn_key=resolved_direct_action_pfa_feature_fn_key,
        hybrid_rollout_rollout_policy_choices=resolved_hybrid_rollout_rollout_choices,
        hybrid_rollout_candidate_policy_choices=resolved_hybrid_rollout_candidate_choices,
        hybrid_rollout_H=resolved_hybrid_rollout_H,
        hybrid_rollout_n_rollouts=resolved_hybrid_rollout_n_rollouts,
        hybrid_rollout_candidate_radius_steps=resolved_hybrid_rollout_candidate_radius_steps,
        hybrid_rollout_include_full_action_grid=resolved_hybrid_rollout_include_full_action_grid,
        hybrid_rollout_decision_seed=resolved_hybrid_rollout_decision_seed,
    )
    incumbent_score = float(smac.runhistory.average_cost(incumbent))
    incumbent_policy = materialize_policy_from_config(
        incumbent_config,
        problem,
        training_seed0=resolved_training_seed0,
        train_info=resolved_train_info,
        policy_cache=policy_cache,
    )

    if hasattr(smac.runhistory, "get_configs"):
        configs = list(smac.runhistory.get_configs(sort_by="cost"))
    else:
        configs = list(getattr(smac.runhistory, "config_ids", {}).keys())
        configs = sorted(configs, key=lambda cfg: float(smac.runhistory.average_cost(cfg)))

    runhistory_rows: list[dict[str, Any]] = []
    for rank, cfg in enumerate(configs, start=1):
        policy_config = smac3_config_to_policy_config(
            cfg,
            problem,
            prefit_policy_names=resolved_prefit_names,
            direct_action_pfa_H=resolved_direct_action_pfa_H,
            direct_action_pfa_feature_fn_key=resolved_direct_action_pfa_feature_fn_key,
            hybrid_rollout_rollout_policy_choices=resolved_hybrid_rollout_rollout_choices,
            hybrid_rollout_candidate_policy_choices=resolved_hybrid_rollout_candidate_choices,
            hybrid_rollout_H=resolved_hybrid_rollout_H,
            hybrid_rollout_n_rollouts=resolved_hybrid_rollout_n_rollouts,
            hybrid_rollout_candidate_radius_steps=resolved_hybrid_rollout_candidate_radius_steps,
            hybrid_rollout_include_full_action_grid=resolved_hybrid_rollout_include_full_action_grid,
            hybrid_rollout_decision_seed=resolved_hybrid_rollout_decision_seed,
        )
        runhistory_rows.append(
            {
                "rank": int(rank),
                "label": str(policy_config["label"]),
                "family": str(policy_config["family"]),
                "score": round(float(smac.runhistory.average_cost(cfg)), 3),
            }
        )

    return PolicyAutoMLSmacResult(
        incumbent_config=incumbent_config,
        incumbent_label=str(incumbent_config["label"]),
        incumbent_score=incumbent_score,
        incumbent_policy=incumbent_policy,
        runhistory_rows=runhistory_rows,
        n_trials_finished=int(getattr(smac.runhistory, "finished", len(runhistory_rows))),
        n_trials_requested=int(n_trials),
        scenario_seed=int(scenario_seed),
    )


def evaluate_policy_config(
    config: Mapping[str, Any],
    problem: PolicyAutoMLDemoProblem,
    *,
    n_episodes: int = 80,
    seed0: int = 20260209,
    training_seed0: int | None = None,
    runtime_penalty: float = 0.0,
    eval_info: Mapping[str, Any] | None = None,
    train_info: Mapping[str, Any] | None = None,
    policy_cache: MutableMapping[Any, Policy] | None = None,
) -> PolicyAutoMLSearchResult:
    """Evaluate one candidate policy under strict CRN."""

    resolved_training_seed0 = int(seed0) + 50_000 if training_seed0 is None else int(training_seed0)
    policy = materialize_policy_from_config(
        config,
        problem,
        training_seed0=resolved_training_seed0,
        train_info=train_info,
        policy_cache=policy_cache,
    )
    info = dict(problem.eval_info if eval_info is None else eval_info)

    t0 = perf_counter()
    totals = problem.system.evaluate_policy_crn_mc(
        policy,
        problem.S0,
        T=int(problem.T),
        n_episodes=int(n_episodes),
        seed0=int(seed0),
        info=info,
    )
    runtime_sec = float(perf_counter() - t0)

    mean_total_cost = float(np.mean(totals)) if len(totals) else float("nan")
    std_total_cost = float(np.std(totals, ddof=1)) if len(totals) > 1 else 0.0
    score = float(mean_total_cost + float(runtime_penalty) * runtime_sec)

    return PolicyAutoMLSearchResult(
        label=str(config.get("label", config["family"])),
        family=str(config["family"]),
        config=dict(config),
        mean_total_cost=mean_total_cost,
        std_total_cost=std_total_cost,
        runtime_sec=runtime_sec,
        score=score,
        totals=np.asarray(totals, dtype=float),
    )


def evaluate_policy_search_space(
    configs: Sequence[Mapping[str, Any]],
    problem: PolicyAutoMLDemoProblem,
    *,
    n_episodes: int = 80,
    seed0: int = 20260209,
    runtime_penalty: float = 0.0,
    eval_info: Mapping[str, Any] | None = None,
) -> list[PolicyAutoMLSearchResult]:
    """Evaluate and rank a finite candidate set."""

    results = [
        evaluate_policy_config(
            cfg,
            problem,
            n_episodes=int(n_episodes),
            seed0=int(seed0),
            runtime_penalty=float(runtime_penalty),
            eval_info=eval_info,
        )
        for cfg in configs
    ]
    return sorted(results, key=lambda r: (r.score, r.runtime_sec, r.label))


def results_to_rows(results: Sequence[PolicyAutoMLSearchResult]) -> list[dict[str, Any]]:
    """Convert results into notebook-friendly rows."""

    rows: list[dict[str, Any]] = []
    for rank, res in enumerate(results, start=1):
        rows.append(
            {
                "rank": int(rank),
                "label": res.label,
                "family": res.family,
                "score": round(float(res.score), 3),
                "mean_total_cost": round(float(res.mean_total_cost), 3),
                "std_total_cost": round(float(res.std_total_cost), 3),
                "runtime_sec": round(float(res.runtime_sec), 3),
            }
        )
    return rows


__all__ = [
    "PolicyAutoMLDemoProblem",
    "PolicyAutoMLSearchResult",
    "PolicyAutoMLSmacResult",
    "smac3_available",
    "make_policy_automl_demo_problem",
    "make_policy_automl_demo_problem_multi_regime",
    "make_policy_automl_demo_search_space",
    "make_policy_automl_demo_search_space_multi_regime",
    "smac_policy_menu_to_search_space_config",
    "make_policy_automl_smac3_configspace",
    "smac3_config_to_policy_config",
    "build_policy_from_config",
    "materialize_policy_from_config",
    "build_exact_dp_benchmark",
    "objective_mean_total_cost",
    "make_policy_automl_smac3_target_function",
    "run_policy_automl_smac3",
    "evaluate_policy_config",
    "evaluate_policy_search_space",
    "results_to_rows",
]
