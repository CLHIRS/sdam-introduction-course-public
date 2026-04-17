"""Notebook-facing convenience helpers.

These helpers are intentionally small wrappers around the core evaluation code so
notebooks can stay focused on modeling/policy ideas instead of repeated glue.
"""

from inventory.evaluation.notebooks.bootstrap import ensure_inventory_imports
from inventory.evaluation.notebooks.crn_runs import CRNRun, collect_crn_rollouts_mc, run_crn_mc
from inventory.evaluation.notebooks.pipeline import CRNExperimentPipelineConfig, run_crn_experiment_pipeline
from inventory.evaluation.notebooks.reports import (
	compute_crn_deltas,
	print_bayesian_report_for_crn_deltas,
	print_frequentist_report_for_crn_deltas,
	print_paired_delta_summary,
	print_totals_summary,
	run_full_ab_workup_from_totals,
)

__all__ = [
	"ensure_inventory_imports",
	"CRNRun",
	"run_crn_mc",
	"collect_crn_rollouts_mc",
	"print_totals_summary",
	"print_paired_delta_summary",
	"compute_crn_deltas",
	"print_frequentist_report_for_crn_deltas",
	"print_bayesian_report_for_crn_deltas",
	"run_full_ab_workup_from_totals",
	"CRNExperimentPipelineConfig",
	"run_crn_experiment_pipeline",
]
