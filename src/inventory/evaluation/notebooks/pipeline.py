from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping

import numpy as np

from inventory.evaluation.deltas import normality_diagnostics
from inventory.evaluation.notebooks.crn_runs import CRNRun, collect_crn_rollouts_mc, run_crn_mc
from inventory.evaluation.notebooks.reports import (
	compute_crn_deltas,
	print_bayesian_report_for_crn_deltas,
	print_frequentist_report_for_crn_deltas,
	print_paired_delta_summary,
)
from inventory.evaluation.plotting import (
	plot_reference_episode_rollouts_grid,
	plot_rollouts_overlay_grid,
	plot_totals_hist_and_box,
)


@dataclass(frozen=True)
class CRNExperimentPipelineConfig:
	baseline_name: str = "baseline"
	candidate_name: str = "candidate"
	higher_is_better: bool = False

	show_step_headings: bool = True

	plot_totals: bool = True
	plot_reference_episode: bool = True
	reference_figsize: tuple[float, float] = (12, 14)

	plot_overlay_rollouts: bool = True
	overlay_n_episodes: int = 50
	overlay_seed0: int = 1234
	overlay_figsize: tuple[float, float] = (12, 14)
	overlay_alpha_episode: float = 0.12
	overlay_linewidth_episode: float = 1.0
	overlay_linewidth_mean: float = 2.8

	print_paired_delta_summary: bool = True
	print_candidate_report: bool = True

	compute_deltas: bool = True

	run_normality: bool = True
	run_frequentist: bool = True
	frequentist_random_state: int | None = 0

	run_bayesian: bool = True
	bayes_better: str | None = None
	bayes_random_state: int = 0
	bayes_draws: int = 200_000
	bayes_cred_level: float = 0.95
	bayes_delta: float = 0.0
	bayes_rope: float = 0.0
	bayes_mode: str = "full"

	@classmethod
	def fast(
		cls,
		*,
		baseline_name: str = "baseline",
		candidate_name: str = "candidate",
		higher_is_better: bool = False,
	) -> "CRNExperimentPipelineConfig":
		"""Preset for quick iteration in teaching notebooks.

		Trades statistical thoroughness for speed:
		- skips overlay rollouts (often the most expensive plot)
		- skips normality diagnostics
		- reduces Bayesian draws
		"""
		return cls(
			baseline_name=baseline_name,
			candidate_name=candidate_name,
			higher_is_better=higher_is_better,
			show_step_headings=True,
			plot_overlay_rollouts=False,
			run_normality=False,
			bayes_draws=50_000,
		)


def _print_step_heading(title: str, *, enabled: bool) -> None:
	if not enabled:
		return
	print("\n" + "=" * (len(title) + 8))
	print(f"=== {title} ===")
	print("=" * (len(title) + 8))
	print("")


def _run_captured(fn: Callable[[], Any]) -> tuple[Any, str]:
	stdout_buf = io.StringIO()
	stderr_buf = io.StringIO()
	with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
		result = fn()
	text = stdout_buf.getvalue() + stderr_buf.getvalue()
	return result, text


def run_crn_experiment_pipeline(
	*,
	system: Any,
	policies: Mapping[str, Any],
	S0: np.ndarray,
	T: int,
	n_episodes: int,
	seed0: int,
	info: Any,
	config: CRNExperimentPipelineConfig | None = None,
	print_summary: bool = True,
	rollouts_collect_kwargs: MutableMapping[str, Any] | None = None,
	show_plots: bool = True,
	plot_deltas: bool = False,
) -> dict[str, Any]:
	"""End-to-end strict-CRN notebook pipeline.

	Designed for teaching notebooks: run the CRN experiment, generate the standard plots,
	compute paired deltas, and print frequentist + Bayesian reports.

	Only required inputs match :func:`run_crn_mc`.
	"""

	cfg = config or CRNExperimentPipelineConfig()

	def heading(title: str, *, gate: bool = True) -> None:
		_print_step_heading(title, enabled=bool(cfg.show_step_headings and gate))

	heading("Strict-CRN Monte Carlo run")
	run = run_crn_mc(
		system=system,
		policies=policies,
		S0=S0,
		T=T,
		n_episodes=n_episodes,
		seed0=seed0,
		info=info,
		print_summary=print_summary,
	)

	if cfg.plot_totals:
		heading("Total cost distribution", gate=True)
		plot_totals_hist_and_box(run.totals_by_policy)

	if cfg.plot_reference_episode:
		heading("Reference episode rollouts (shared episode)", gate=True)
		plot_reference_episode_rollouts_grid(
			run.rollouts,
			figsize=cfg.reference_figsize,
			show=show_plots,
			marker="o",
		)

	overlay_rollouts = None
	if cfg.plot_overlay_rollouts:
		heading(f"Overlay rollouts (CRN-MC; n={cfg.overlay_n_episodes})", gate=True)
		overlay_rollouts = collect_crn_rollouts_mc(
			system=system,
			policies=policies,
			S0=S0,
			T=T,
			n_episodes=cfg.overlay_n_episodes,
			seed0=cfg.overlay_seed0,
			info=info,
			kwargs=rollouts_collect_kwargs,
		)

		plot_rollouts_overlay_grid(
			overlay_rollouts,
			figsize=cfg.overlay_figsize,
			alpha_episode=cfg.overlay_alpha_episode,
			linewidth_episode=cfg.overlay_linewidth_episode,
			linewidth_mean=cfg.overlay_linewidth_mean,
			title_suffix=f"(CRN-MC overlay; n={cfg.overlay_n_episodes})",
			show=show_plots,
		)

	reports: dict[str, Any] | None = None
	candidate_report: Any = None
	if cfg.print_paired_delta_summary:
		heading("Paired delta summary (vs baseline)", gate=True)
		reports = print_paired_delta_summary(
			run.totals_by_policy,
			baseline_name=cfg.baseline_name,
			higher_is_better=cfg.higher_is_better,
		)

		candidate_report = reports.get(cfg.candidate_name) if reports else None
		if cfg.print_candidate_report and candidate_report is not None:
			heading(f"PairedDeltaReport[{cfg.candidate_name}]", gate=True)
			print(candidate_report)

	deltas: np.ndarray | None = None
	if cfg.compute_deltas:
		heading("CRN deltas (candidate - baseline)", gate=True)
		deltas = compute_crn_deltas(
			run.totals_by_policy,
			base_policy=cfg.baseline_name,
			other_policy=cfg.candidate_name,
			plot=plot_deltas,
		)

	normality = None
	if cfg.run_normality and deltas is not None:
		heading("Normality diagnostics of CRN deltas", gate=True)
		normality = normality_diagnostics(deltas)

	frequentist = None
	if cfg.run_frequentist and deltas is not None:
		heading("Frequentist analysis of CRN deltas", gate=True)
		frequentist, text = _run_captured(
			lambda: print_frequentist_report_for_crn_deltas(
				deltas,
				random_state=cfg.frequentist_random_state,
			)
		)
		if text:
			print(text, end="" if text.endswith("\n") else "\n")

	if cfg.run_bayesian and deltas is not None:
		heading("Bayesian analysis of CRN deltas", gate=True)
		better = cfg.bayes_better
		if better is None:
			better = "higher" if cfg.higher_is_better else "lower"
		_, text = _run_captured(
			lambda: print_bayesian_report_for_crn_deltas(
				deltas,
				better=better,
				deltas_are="B_minus_A",
				random_state=cfg.bayes_random_state,
				n_draws=cfg.bayes_draws,
				cred_level=cfg.bayes_cred_level,
				delta=cfg.bayes_delta,
				rope=cfg.bayes_rope,
				mode=cfg.bayes_mode,
			)
		)
		if text:
			print(text, end="" if text.endswith("\n") else "\n")

	return {
		"run": run,
		"results": run.results,
		"rollouts": run.rollouts,
		"totals_by_policy": run.totals_by_policy,
		"totals_summary": run.totals_summary,
		"overlay_rollouts": overlay_rollouts,
		"reports": reports,
		"candidate_report": candidate_report,
		"deltas": deltas,
		"normality": normality,
		"frequentist": frequentist,
	}
