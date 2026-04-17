from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from inventory.evaluation.bayesian import print_bayesian_report_AB_crn
from inventory.evaluation.deltas import crn_delta_totals_from_totals_by_policy, normality_diagnostics, print_totals_summary
from inventory.evaluation.frequentist import (
	format_frequentist_report_AB,
	frequentist_analysis_AB,
	paired_deltas_against_baseline,
)


def print_paired_delta_summary(
	totals_by_policy: Mapping[str, np.ndarray],
	*,
	baseline_name: str,
	higher_is_better: bool,
) -> dict[str, Any]:
	"""Print the paired-delta summary vs baseline and return the reports."""

	reports = paired_deltas_against_baseline(
		dict(totals_by_policy),
		baseline_name=baseline_name,
		higher_is_better=higher_is_better,
	)
	print("=== Final CRN-MC PairedDeltaReport ===")
	for name, res in reports.items():
		direction = "higher is better" if higher_is_better else "lower is better"
		print(
			f"baseline={res.baseline:10s} | other={res.other:10s} | mean_delta={res.mean_delta:.3f} | "
			f"std_delta={res.std_delta:.3f} | win_rate={res.win_rate:.3f} | ({direction})"
		)
	return reports


def compute_crn_deltas(
	totals_by_policy: Mapping[str, Any],
	*,
	base_policy: str,
	other_policy: str,
	plot: bool = False,
) -> np.ndarray:
	"""Compute per-episode strict-CRN deltas of TOTAL cost: other - base."""

	deltas = crn_delta_totals_from_totals_by_policy(
		totals_by_policy,
		base_policy=base_policy,
		other_policy=other_policy,
		plot=plot,
	)
	return np.asarray(deltas, dtype=float)


def print_frequentist_report_for_crn_deltas(
	deltas: np.ndarray,
	*,
	objective: str = "cost",
	delta_convention: str = "treatment-control",
	claim: str = "treatment_better",
	random_state: int | None = 0,
) -> dict[str, Any]:
	"""Run and print the standard frequentist paired-delta report."""

	res = frequentist_analysis_AB(
		deltas,
		objective=objective,
		delta_convention=delta_convention,
		claim=claim,
		random_state=random_state,
	)
	print(format_frequentist_report_AB(res))
	return res


def print_bayesian_report_for_crn_deltas(
	deltas: np.ndarray,
	*,
	better: str = "lower",
	deltas_are: str = "B_minus_A",
	random_state: int = 0,
	n_draws: int = 200_000,
	cred_level: float = 0.95,
	delta: float = 0.0,
	rope: float = 0.0,
	mode: str = "full",
) -> None:
	"""Print the standard Bayesian CRN report for paired deltas."""

	_ = print_bayesian_report_AB_crn(
		deltas,
		better=better,
		deltas_are=deltas_are,
		random_state=random_state,
		n_draws=n_draws,
		cred_level=cred_level,
		delta=delta,
		rope=rope,
		mode=mode,
		return_result=False,
	)


def run_full_ab_workup_from_totals(
	totals_by_policy: Mapping[str, Any],
	*,
	baseline_name: str,
	candidate_name: str,
	higher_is_better: bool,
	plot_deltas: bool = False,
	random_state: int | None = 0,
	run_normality: bool = True,
	run_frequentist: bool = True,
	run_bayesian: bool = True,
	bayes_draws: int = 200_000,
) -> dict[str, Any]:
	"""One-liner notebook helper: totals -> paired deltas -> diagnostics + reports."""

	print_totals_summary(totals_by_policy)
	reports = print_paired_delta_summary(
		totals_by_policy,
		baseline_name=baseline_name,
		higher_is_better=higher_is_better,
	)

	deltas = compute_crn_deltas(
		totals_by_policy,
		base_policy=baseline_name,
		other_policy=candidate_name,
		plot=plot_deltas,
	)

	out: dict[str, Any] = {"paired": reports, "deltas": deltas}

	if run_normality:
		out["normality"] = normality_diagnostics(deltas)

	if run_frequentist:
		out["frequentist"] = print_frequentist_report_for_crn_deltas(
			deltas,
			random_state=random_state,
		)

	if run_bayesian:
		better = "higher" if higher_is_better else "lower"
		print_bayesian_report_for_crn_deltas(
			deltas,
			better=better,
			deltas_are="B_minus_A",
			random_state=int(random_state or 0),
			n_draws=bayes_draws,
		)

	return out
