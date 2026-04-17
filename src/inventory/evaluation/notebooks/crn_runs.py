from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

import numpy as np

from inventory.evaluation.reporting import summarize_totals


@dataclass(frozen=True)
class CRNRun:
	results: Mapping[str, Any]
	rollouts: Any
	totals_by_policy: dict[str, np.ndarray]
	totals_summary: dict[str, dict[str, float]]


def run_crn_mc(
	*,
	system: Any,
	policies: Mapping[str, Any],
	S0: np.ndarray,
	T: int,
	n_episodes: int,
	seed0: int,
	info: Any,
	print_summary: bool = True,
) -> CRNRun:
	"""Run strict-CRN Monte Carlo evaluation and return notebook-friendly outputs."""

	results, rollouts = system.evaluate_policies_crn_mc(
		policies,
		S0,
		T=T,
		n_episodes=n_episodes,
		seed0=seed0,
		info=info,
	)

	totals_by_policy: dict[str, np.ndarray] = {
		name: np.asarray(res.totals, dtype=float) for name, res in results.items()
	}
	totals_summary = summarize_totals(totals_by_policy)
	for name, res in results.items():
		totals_summary.setdefault(name, {})["runtime_sec"] = float(getattr(res, "runtime_sec", 0.0))

	if print_summary:
		print("=== Final CRN-MC totals (lower is better) ===")
		for name, r in totals_summary.items():
			print(
				f"{name:15s} | mean={r['mean']:.3f} | std={r['std']:.3f} | "
				f"n={int(r['n'])} | runtime={r.get('runtime_sec', 0.0):.3f}s"
			)

	return CRNRun(
		results=results,
		rollouts=rollouts,
		totals_by_policy=totals_by_policy,
		totals_summary=totals_summary,
	)


def collect_crn_rollouts_mc(
	*,
	system: Any,
	policies: Mapping[str, Any],
	S0: np.ndarray,
	T: int,
	n_episodes: int,
	seed0: int,
	info: Any,
	kwargs: MutableMapping[str, Any] | None = None,
) -> Any:
	"""Collect CRN rollouts for plotting overlays (many episodes per policy)."""

	return system.collect_policies_crn_rollouts_mc(
		policies,
		S0,
		T=T,
		n_episodes=n_episodes,
		seed0=seed0,
		info=info,
		**(kwargs or {}),
	)
