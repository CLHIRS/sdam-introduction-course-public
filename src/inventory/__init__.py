"""inventory

Permanent, importable code for inventory-related lectures.

Design goals:
- Keep notebook code minimal: import from `inventory` rather than copy/paste.
- Preserve didactic clarity: core system first, then policies, then evaluation.
- Enable strict CRN evaluation for fair policy comparisons.

Usage (when running from repo root):
- `poetry install` once, then use `poetry run python ...` so `import inventory` works.
"""

from inventory.core.dynamics import DynamicSystemMVP
from inventory.core.exogenous import ExogenousModel
from inventory.core.policy import Policy
from inventory.core.types import Action, Exog, State
from inventory.policies.alphazero import Hybrid_AlphaZero, HybridAlphaZeroPolicy
from inventory.policies.baselines import OrderUpToCapacityPolicy, OrderUpToPolicy, PolicyOrderUpToCapacity
from inventory.policies.cfa_milp import Order_MILP_CFA, OrderMilpCfaPolicy
from inventory.policies.dla_mcts import DLA_MCTS_UCT, DlaMctsUctPolicy
from inventory.policies.dla_milp import DLA_MPC_MILP, DlaMpcMilpPolicy
from inventory.policies.hybrids import (
    Hybrid_DLA_CFA_BufferedForecast,
    Hybrid_DLA_Rollout_BasePFA,
    Hybrid_DLA_VFA_Terminal,
    HybridDlaCfaBufferedForecastPolicy,
    HybridDlaRolloutBasePfaPolicy,
    HybridDlaVfaTerminalPolicy,
    TerminalVFA,
    TerminalVFA_Linear,
    TerminalVfaLinear,
)

try:
    from inventory.policies.ppo import (
        ForecastAugmentedHybridPpoPolicy,
        Hybrid_PPO,
        HybridPpoPolicy,
        PPOHyperParams,
        train_ppo_with_eval_gate,
    )
except Exception:  # optional dependency (torch)
    ForecastAugmentedHybridPpoPolicy = None  # type: ignore[assignment]
    Hybrid_PPO = None  # type: ignore[assignment]
    HybridPpoPolicy = None  # type: ignore[assignment]
    PPOHyperParams = None  # type: ignore[assignment]
    train_ppo_with_eval_gate = None  # type: ignore[assignment]
from inventory.evaluation.info import make_eval_info, make_train_info
from inventory.forecasters.base import DemandForecaster
from inventory.forecasters.factory import (
    FitConfig,
    fit_ml_forecaster_from_exogenous,
    forecast_with_default_state,
    make_adapter,
    make_ml_forecaster,
)
from inventory.forecasters.ml import (
    MlAr1RegimeDemandForecaster,
    MlDemandForecaster,
    MlRegimeDemandForecaster,
    MultiRegimeAr1FeatureAdapter,
    MultiRegimeFeatureAdapter,
    QuantileBoostingRegimeDemandForecaster,
    RegimeFeatureAdapter,
    SeasonalFeatureAdapter,
)
from inventory.forecasters.naive import ExpertDemandForecasterConstant350, NaiveForecaster, RollingMeanForecaster
from inventory.forecasters.path import (
    ConstantMeanPathForecaster,
    DemandPathForecaster,
    ExogenousMeanPathForecaster,
    SeasonalSinMeanPathForecaster,
)
from inventory.forecasters.ts import EtsDemandForecaster, SarimaxRegimeDemandForecaster
from inventory.policies.pfa import OrderUpToBlackboxPFA, OrderUpToRegimeTablePFA, OrderUpToStateDependentPFA
from inventory.policies.vfa import (
    FQI_Greedy_VFA,
    FqiGreedyVfaPolicy,
    PostDecision_Greedy_VFA,
    PostDecisionGreedyVfaPolicy,
)
from inventory.problems.inventory_mvp import make_inventory_multi_regime_system

__all__ = [
    "State",
    "Action",
    "Exog",
    "Policy",
    "ExogenousModel",
    "DynamicSystemMVP",
    "OrderUpToPolicy",
    "OrderUpToCapacityPolicy",
    "PolicyOrderUpToCapacity",
    "OrderMilpCfaPolicy",
    "Order_MILP_CFA",
    "DlaMpcMilpPolicy",
    "DLA_MPC_MILP",
    "DlaMctsUctPolicy",
    "DLA_MCTS_UCT",
    "HybridAlphaZeroPolicy",
    "Hybrid_AlphaZero",
    "OrderUpToBlackboxPFA",
    "OrderUpToRegimeTablePFA",
    "OrderUpToStateDependentPFA",
    "TerminalVFA",
    "TerminalVfaLinear",
    "TerminalVFA_Linear",
    "HybridDlaCfaBufferedForecastPolicy",
    "HybridDlaVfaTerminalPolicy",
    "HybridDlaRolloutBasePfaPolicy",
    "Hybrid_DLA_CFA_BufferedForecast",
    "Hybrid_DLA_VFA_Terminal",
    "Hybrid_DLA_Rollout_BasePFA",
    "ForecastAugmentedHybridPpoPolicy",
    "PPOHyperParams",
    "HybridPpoPolicy",
    "Hybrid_PPO",
    "train_ppo_with_eval_gate",
    "PostDecisionGreedyVfaPolicy",
    "FqiGreedyVfaPolicy",
    "PostDecision_Greedy_VFA",
    "FQI_Greedy_VFA",
    "DemandForecaster",
    "DemandPathForecaster",
    "ConstantMeanPathForecaster",
    "SeasonalSinMeanPathForecaster",
    "ExogenousMeanPathForecaster",
    "EtsDemandForecaster",
    "SeasonalFeatureAdapter",
    "RegimeFeatureAdapter",
    "MultiRegimeFeatureAdapter",
    "MultiRegimeAr1FeatureAdapter",
    "MlDemandForecaster",
    "MlRegimeDemandForecaster",
    "QuantileBoostingRegimeDemandForecaster",
    "SarimaxRegimeDemandForecaster",
    "MlAr1RegimeDemandForecaster",
    "NaiveForecaster",
    "RollingMeanForecaster",
    "ExpertDemandForecasterConstant350",
    "FitConfig",
    "make_adapter",
    "make_ml_forecaster",
    "fit_ml_forecaster_from_exogenous",
    "forecast_with_default_state",
    "make_eval_info",
    "make_train_info",
    "make_inventory_multi_regime_system",
]
