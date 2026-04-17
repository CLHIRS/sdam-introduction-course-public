from .alphazero import Hybrid_AlphaZero, HybridAlphaZeroPolicy
from .baselines import OrderUpToCapacityPolicy, OrderUpToPolicy, PolicyOrderUpToCapacity
from .cfa_milp import Order_MILP_CFA, Order_MILP_CFA_LeadTime, OrderMilpCfaLeadTimePolicy, OrderMilpCfaPolicy
from .dla_mcts import DLA_MCTS_UCT, DlaMctsUctPolicy
from .dla_milp import DLA_MPC_MILP, DlaMpcMilpPolicy
from .dp import (
	DPSolution1D,
	DPSolutionRegime,
	DPSolverRegimeScenarios,
	DynamicProgrammingPolicy,
	DynamicProgrammingPolicyRegime,
	DynamicProgrammingSolver1D,
)
from .hybrids import (
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
from .pfa import OrderUpToBlackboxPFA, OrderUpToRegimeTablePFA, OrderUpToStateDependentPFA

try:
	from .ppo import ForecastAugmentedHybridPpoPolicy, Hybrid_PPO, HybridPpoPolicy, PPOHyperParams, train_ppo_with_eval_gate
except Exception:  # optional dependency (torch)
	ForecastAugmentedHybridPpoPolicy = None  # type: ignore[assignment]
	Hybrid_PPO = None  # type: ignore[assignment]
	HybridPpoPolicy = None  # type: ignore[assignment]
	PPOHyperParams = None  # type: ignore[assignment]
	train_ppo_with_eval_gate = None  # type: ignore[assignment]
from .vfa import FQI_Greedy_VFA, FqiGreedyVfaPolicy, PostDecision_Greedy_VFA, PostDecisionGreedyVfaPolicy

__all__ = [
	"OrderUpToPolicy",
	"OrderUpToCapacityPolicy",
	"PolicyOrderUpToCapacity",
	"OrderMilpCfaPolicy",
	"Order_MILP_CFA",
	"OrderMilpCfaLeadTimePolicy",
	"Order_MILP_CFA_LeadTime",
	"DlaMpcMilpPolicy",
	"DLA_MPC_MILP",
	"DlaMctsUctPolicy",
	"DLA_MCTS_UCT",
	"OrderUpToBlackboxPFA",
	"OrderUpToRegimeTablePFA",
	"OrderUpToStateDependentPFA",
	"DynamicProgrammingPolicy",
	"DynamicProgrammingSolver1D",
	"DPSolution1D",
	"DynamicProgrammingPolicyRegime",
	"DPSolverRegimeScenarios",
	"DPSolutionRegime",
	"TerminalVFA",
	"TerminalVfaLinear",
	"TerminalVFA_Linear",
	"HybridDlaCfaBufferedForecastPolicy",
	"HybridDlaVfaTerminalPolicy",
	"HybridDlaRolloutBasePfaPolicy",
	"Hybrid_DLA_CFA_BufferedForecast",
	"Hybrid_DLA_VFA_Terminal",
	"Hybrid_DLA_Rollout_BasePFA",
	"HybridAlphaZeroPolicy",
	"Hybrid_AlphaZero",
	"ForecastAugmentedHybridPpoPolicy",
	"PPOHyperParams",
	"HybridPpoPolicy",
	"Hybrid_PPO",
	"train_ppo_with_eval_gate",
	"PostDecisionGreedyVfaPolicy",
	"FqiGreedyVfaPolicy",
	"PostDecision_Greedy_VFA",
	"FQI_Greedy_VFA",
]
"""Policy implementations for inventory lectures."""
