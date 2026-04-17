"""Evaluation utilities (frequentist/bayesian/reporting) for inventory lectures.

Scripts in this folder include:
- `smoke_canonical.py`: canonical notebook parity smoke tests.
- `forecast_benchmark.py`: compare forecasters fairly on shared synthetic demand paths.
"""

from inventory.evaluation.bayesian import (
	bayesian_AB_test_crn,
	bayesian_ab_test_normal,
	bayesian_ab_test_vanilla,
	print_bayesian_ab_report_normal,
	print_bayesian_ab_report_vanilla,
	print_bayesian_report_AB_crn,
)
from inventory.evaluation.deltas import (
	crn_delta_totals_from_totals_by_policy,
	normality_diagnostics,
	normality_diagnostics_for_deltas,
	print_totals_summary,
)
from inventory.evaluation.frequentist import (
	format_frequentist_report_AB,
	frequentist_analysis_AB,
	print_frequentist_report_AB,
)
from inventory.evaluation.info import make_eval_info, make_train_info
from inventory.evaluation.plotting import (
	PairedDeltaPlotSummary,
	plot_multiple_results_grid,
	plot_paired_deltas_vs_baseline,
	plot_reference_episode_rollouts,
	plot_results_grid,
	plot_totals_hist_and_box,
	show_forecast_paths,
)

__all__ = [
	"bayesian_AB_test_crn",
	"print_bayesian_report_AB_crn",
	"bayesian_ab_test_vanilla",
	"print_bayesian_ab_report_vanilla",
	"bayesian_ab_test_normal",
	"print_bayesian_ab_report_normal",
	"frequentist_analysis_AB",
	"format_frequentist_report_AB",
	"print_frequentist_report_AB",
	"crn_delta_totals_from_totals_by_policy",
	"print_totals_summary",
	"normality_diagnostics_for_deltas",
	"normality_diagnostics",
	"make_eval_info",
	"make_train_info",
	"plot_results_grid",
	"plot_multiple_results_grid",
	"show_forecast_paths",
	"plot_totals_hist_and_box",
	"PairedDeltaPlotSummary",
	"plot_paired_deltas_vs_baseline",
	"plot_reference_episode_rollouts",
]
