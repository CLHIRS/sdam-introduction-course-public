"""Bayesian evaluation helpers.

Extracted from Lecture 01_d refactor notebook.

These helpers are designed for teaching and for evaluating paired deltas (CRN) as well
as classic independent-arm A/B tests.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
	"bayesian_AB_test_crn",
	"print_bayesian_report_AB_crn",
	"bayesian_ab_test_vanilla",
	"print_bayesian_ab_report_vanilla",
	"bayesian_ab_test_normal",
	"print_bayesian_ab_report_normal",
]


def bayesian_AB_test_crn(
	deltas,
	totals_A=None,
	totals_B=None,
	*,
	delta=0.0,
	better="lower",  # costs -> "lower"
	deltas_are="B_minus_A",  # {"B_minus_A","A_minus_B"}
	method="bayes_bootstrap",  # currently: "bayes_bootstrap"
	n_draws=200_000,
	cred_level=0.95,
	rope=0.0,  # practical equivalence on mean effect
	percent_improvement=True,  # compute posterior over mean % improvement
	episode_superiority=True,  # compute posterior over P(B<A) etc (requires totals)
	eps=1e-12,  # for ratio stability
	random_state=0,
	return_draws=False,  # if True, store posterior draws for consistent plotting
):
	"""Bayesian paired-policy analysis for episode TOTAL costs with CRN pairing.

	Inputs (paired, aligned by episode/CRN seed):
	  - totals_A: episode totals for policy A (control)
	  - totals_B: episode totals for policy B (treatment)
	  - deltas: episode-wise differences (B-A by default)

	Bayesian method:
	  - Bayesian bootstrap (Dirichlet weights) over paired episodes.
	  - Produces posterior over mean effect and various probabilities.

	Interpretation (default: costs, lower is better):
	  - Δ = B - A
	  - mean-level: "B better" means E[Δ] < 0
	  - "B at least delta better" means E[Δ] <= -delta
	  - episode-level: P(B < A) and P(B <= A - delta)
	"""

	rng = np.random.default_rng(random_state)

	if method != "bayes_bootstrap":
		raise ValueError("Currently only method='bayes_bootstrap' is implemented (robust, pairing-aware).")

	# ---- sanitize deltas ----
	d = np.asarray(deltas, dtype=float)
	d = d[np.isfinite(d)]
	if d.size < 2:
		raise ValueError("Need at least 2 finite delta samples.")

	# Normalize to B-A
	if deltas_are == "A_minus_B":
		d = -d
	elif deltas_are != "B_minus_A":
		raise ValueError("deltas_are must be 'B_minus_A' or 'A_minus_B'.")

	# ---- totals (optional) ----
	A = B = None
	if totals_A is not None and totals_B is not None:
		A0 = np.asarray(totals_A, dtype=float)
		B0 = np.asarray(totals_B, dtype=float)
		m = np.isfinite(A0) & np.isfinite(B0)
		A = A0[m]
		B = B0[m]
		if A.size != d.size:
			raise ValueError(
				f"Length mismatch: deltas n={d.size}, totals_A/totals_B n={A.size}. "
				"They must be paired and aligned."
			)

	n = d.size
	delta = float(delta)
	rope = float(rope)

	# Credible interval quantiles
	alpha = 1.0 - float(cred_level)
	q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

	def summarize(draws: Any) -> dict[str, Any]:
		draws = np.asarray(draws, dtype=float)
		return {
			"mean": float(np.mean(draws)),
			"median": float(np.median(draws)),
			"sd": float(np.std(draws, ddof=1)),
			"ci": [float(np.quantile(draws, q_lo)), float(np.quantile(draws, q_hi))],
		}

	# ---- Bayesian bootstrap weights: Dirichlet(1,...,1) via Gamma(1,1) ----
	g = rng.gamma(shape=1.0, scale=1.0, size=(int(n_draws), n))
	w = g / np.sum(g, axis=1, keepdims=True)

	# Posterior over mean treatment effect E[B-A]
	mean_d = w @ d

	# Mean-level event definition
	if better == "lower":
		# costs: "B at least delta better" => E[B-A] <= -delta
		mean_event_text = f"E[B-A] <= {-delta:.6g}"
		prob_dir = float(np.mean(mean_d < 0.0))
		dir_text = "P(E[B-A] < 0)"
		prob_better_by_delta = float(np.mean(mean_d <= -delta))
	elif better == "higher":
		mean_event_text = f"E[B-A] >= {delta:.6g}"
		prob_dir = float(np.mean(mean_d > 0.0))
		dir_text = "P(E[B-A] > 0)"
		prob_better_by_delta = float(np.mean(mean_d >= delta))
	else:
		raise ValueError("better must be 'lower' or 'higher'.")

	# ROPE probability on mean effect
	prob_in_rope = float(np.mean(np.abs(mean_d) <= rope)) if rope > 0 else None

	posterior = {
		"method": "bayes_bootstrap",
		"n_draws": int(n_draws),
		"mean_level_event": mean_event_text,
		"mean_delta_(E[B-A])": summarize(mean_d),
		dir_text: prob_dir,
		"prob_B_better_by_delta_(mean_level)": prob_better_by_delta,
		"rope": rope if rope > 0 else None,
		"prob_practically_equivalent_(|E[B-A]|<=rope)": prob_in_rope,
	}

	# Optional mean totals + mean percent improvement
	totals_post = None
	if A is not None and B is not None:
		mean_A = w @ A
		mean_B = w @ B
		totals_post = {
			"E[A]": summarize(mean_A),
			"E[B]": summarize(mean_B),
		}
		if percent_improvement:
			# Positive means B improves when lower-is-better: (A - B)/|A|
			pct = (mean_A - mean_B) / (np.abs(mean_A) + float(eps))
			totals_post["mean_percent_improvement_(E-level)"] = summarize(pct)

	# Optional episode-level superiority probabilities (paired)
	superiority_post = None
	if episode_superiority and A is not None and B is not None:
		if better == "lower":
			ind0 = (B < A).astype(float)
			indD = (B <= (A - delta)).astype(float)
			k0 = "P(B < A) (episode-level)"
			kD = f"P(B <= A - {delta:.6g}) (episode-level)"
		else:
			ind0 = (B > A).astype(float)
			indD = (B >= (A + delta)).astype(float)
			k0 = "P(B > A) (episode-level)"
			kD = f"P(B >= A + {delta:.6g}) (episode-level)"

		p_sup0 = w @ ind0
		p_supD = w @ indD
		superiority_post = {
			k0: summarize(p_sup0),
			kD: summarize(p_supD),
		}

	# Decision-friendly verdict
	if prob_better_by_delta >= 0.95:
		strength = "Strong"
	elif prob_better_by_delta >= 0.80:
		strength = "Moderate"
	else:
		strength = "Weak/insufficient"
	verdict = f"{strength} evidence: P({mean_event_text}) = {prob_better_by_delta:.3f}"

	# One-page report string
	lines: list[str] = []
	lines.append("Bayesian Policy Comparison (paired CRN, episode TOTAL cost)")
	lines.append("=" * 62)
	lines.append(f"n episodes: {n}")
	lines.append("Interpretation: deltas = B - A")
	lines.append("")
	lines.append("Data summary (deltas = B-A):")
	lines.append(f"  sample mean: {np.mean(d): .6g}")
	lines.append(f"  sample sd  : {np.std(d, ddof=1): .6g}")
	lines.append(f"  min / max  : {np.min(d): .6g} / {np.max(d): .6g}")
	lines.append("")
	md = posterior["mean_delta_(E[B-A])"]
	lines.append("Posterior on mean treatment effect (E[B-A]):")
	lines.append(f"  {cred_level*100:.1f}% CrI: [{md['ci'][0]: .6g}, {md['ci'][1]: .6g}]")
	lines.append(f"  {dir_text}: {posterior[dir_text]:.3f}")
	lines.append(f"  P({mean_event_text}): {prob_better_by_delta:.3f}")
	if rope > 0:
		lines.append(f"  ROPE ±{rope:g}: P(|E[B-A]|<=rope) = {float(prob_in_rope):.3f}")
	lines.append("")
	if totals_post is not None:
		ciA = totals_post["E[A]"]["ci"]
		ciB = totals_post["E[B]"]["ci"]
		lines.append("Posterior on mean totals:")
		lines.append(f"  E[A] CrI: [{ciA[0]: .6g}, {ciA[1]: .6g}]")
		lines.append(f"  E[B] CrI: [{ciB[0]: .6g}, {ciB[1]: .6g}]")
		if percent_improvement and "mean_percent_improvement_(E-level)" in totals_post:
			cip = totals_post["mean_percent_improvement_(E-level)"]["ci"]
			lines.append(f"  Mean % improvement CrI: [{100*cip[0]: .3g}%, {100*cip[1]: .3g}%]")
		lines.append("")
	if superiority_post is not None:
		lines.append("Episode-level superiority (paired):")
		for k, s in superiority_post.items():
			ci_k = s["ci"]
			lines.append(f"  {k}: mean={s['mean']:.3f}  CrI=[{ci_k[0]:.3f}, {ci_k[1]:.3f}]")
		lines.append("")
	lines.append("Verdict:")
	lines.append(f"  {verdict}")

	report = "\n".join(lines)

	result: dict[str, Any] = {
		"inputs": {
			"better": better,
			"deltas_are": "B_minus_A",
			"delta": float(delta),
			"cred_level": float(cred_level),
			"method": method,
			"n_draws": int(n_draws),
			"rope": rope if rope > 0 else None,
			"random_state": int(random_state),
		},
		"data_summary": {
			"n": int(n),
			"sample_mean_(B-A)": float(np.mean(d)),
			"sample_sd_(B-A)": float(np.std(d, ddof=1)),
			"min_(B-A)": float(np.min(d)),
			"max_(B-A)": float(np.max(d)),
		},
		"posterior": posterior,
		"posterior_totals_optional": totals_post,
		"posterior_episode_superiority_optional": superiority_post,
		"verdict": verdict,
		"report": report,
	}

	if return_draws:
		draws: dict[str, Any] = {
			"weights": w,  # (n_draws, n)
			"mean_delta": mean_d,  # (n_draws,)
			"deltas": d,  # (n,)
			"better": better,
			"delta": float(delta),
			"rope": rope if rope > 0 else None,
		}
		if A is not None and B is not None:
			draws["totals_A"] = A
			draws["totals_B"] = B
			draws["mean_A"] = (w @ A)
			draws["mean_B"] = (w @ B)
		result["draws"] = draws

	return result


def print_bayesian_report_AB_crn(
	deltas,
	totals_A=None,
	totals_B=None,
	*,
	# Analysis knobs
	delta=0.0,
	better="lower",
	deltas_are="B_minus_A",
	cred_level=0.95,
	rope=0.0,
	n_draws=200_000,
	random_state=0,
	percent_improvement=True,
	episode_superiority=True,
	# Printing knobs
	mode="full",  # {"full","table","compact","decision_only","complete_AB_report"}
	pass_threshold=0.95,
	show_table=False,  # table requires pandas + helper; kept off by default here
	digits=4,
	prob_as_percent=True,  # (reserved; used if you add a pandas table helper)
	# Repro/metadata for complete_AB_report
	crn_description="Paired CRN episodes (same seeds/streams across A and B).",
	decision_rule_text=None,
	sensitivity_note="(Optional) Sensitivity: rerun with different n_draws or compare to a parametric model.",
	include_assumptions=True,
	assumptions_text=None,  # if None, auto-generated (1–2 sentences)
	return_result=True,
):
	"""Runs bayesian_AB_test_crn(...) and prints output depending on `mode`."""

	_ = (show_table, prob_as_percent)  # reserved

	# Run analysis ONCE with saved draws (needed for regret in complete_AB_report)
	res = bayesian_AB_test_crn(
		deltas=deltas,
		totals_A=totals_A,
		totals_B=totals_B,
		delta=delta,
		better=better,
		deltas_are=deltas_are,
		n_draws=n_draws,
		cred_level=cred_level,
		rope=rope,
		percent_improvement=percent_improvement,
		episode_superiority=episode_superiority,
		random_state=random_state,
		return_draws=True,
	)

	post = res.get("posterior", {})
	event_txt = post.get("mean_level_event", "B better by delta")
	p = post.get("prob_B_better_by_delta_(mean_level)", None)

	if p is None:
		decision_line = "Decision: (missing key probability)"
	else:
		status = "PASS ✅" if p >= pass_threshold else "NO PASS —"
		decision_line = f"Decision @ {pass_threshold:.2f}: {status}   P({event_txt}) = {p:.3f}"

	mode_l = str(mode).lower().strip()

	# ---------------- complete_AB_report ----------------
	if mode_l == "complete_ab_report":
		draws = res.get("draws", {})
		mean_d = draws.get("mean_delta", None)
		if mean_d is None:
			raise ValueError("complete_AB_report requires return_draws=True (enabled internally).")

		# Expected regret at mean-level
		if better == "lower":
			regret_choose_B = np.maximum(0.0, mean_d)
			regret_choose_A = np.maximum(0.0, -mean_d)
			dir_key = "P(E[B-A] < 0)"
			_better_label = "Lower cost is better"
		else:
			regret_choose_B = np.maximum(0.0, -mean_d)
			regret_choose_A = np.maximum(0.0, mean_d)
			dir_key = "P(E[B-A] > 0)"
			_better_label = "Higher is better"

		exp_regret_B = float(np.mean(regret_choose_B))
		exp_regret_A = float(np.mean(regret_choose_A))

		p_event = float(p) if p is not None else float("nan")
		wrong_decision_prob_proxy = float(1.0 - p_event) if np.isfinite(p_event) else float("nan")

		if decision_rule_text is None:
			decision_rule_text = f"Ship B if P({event_txt}) ≥ {pass_threshold:.2f}."

		if include_assumptions and assumptions_text is None:
			rope_str = f"ROPE ±{rope:g}" if (rope is not None and float(rope) > 0) else "no ROPE"
			assumptions_text = (
				"Assumptions / setup: We compare paired CRN episode total costs using Δ = B − A. "
				"Posterior uncertainty is estimated via the Bayesian bootstrap (nonparametric) "
				f"with practical threshold δ={delta:g} and {rope_str}."
			)

		md = post.get("mean_delta_(E[B-A])", {})
		ci = md.get("ci", [float("nan"), float("nan")])
		sup = res.get("posterior_episode_superiority_optional") or {}
		rope_val = post.get("rope", None)
		p_rope = post.get("prob_practically_equivalent_(|E[B-A]|<=rope)", None)

		print("Bayesian A/B Report (COMPLETE, paired CRN)")
		print("=" * 72)
		print(f"n paired episodes: {res['data_summary']['n']}")
		print(f"CRN pairing: {crn_description}")
		print("Method: Bayesian bootstrap (Dirichlet weights over empirical support)")
		print(f"Monte Carlo: n_draws={n_draws}, seed={random_state}")
		print(f"Decision rule: {decision_rule_text}")
		print("")
		if include_assumptions and assumptions_text:
			print(assumptions_text)
			print("")

		# A
		print("A) Effect size & uncertainty (mean-level)")
		print(
			f"  E[B-A]: mean={md.get('mean', float('nan')):.{digits}g}  {100*cred_level:.1f}% CrI=[{ci[0]:.{digits}g}, {ci[1]:.{digits}g}]"
		)
		if dir_key in post:
			print(f"  {dir_key}: {post[dir_key]:.3f}")
		print(f"  P({event_txt}): {p_event:.3f}")
		print("")

		# B
		print("B) Practical significance")
		if rope_val is not None and p_rope is not None:
			print(f"  ROPE ±{rope_val:g}: P(|E[B-A]|<=rope) = {p_rope:.3f}")
		else:
			print("  ROPE: (not used)")
		print("")

		# C
		print("C) Superiority / robustness (episode-level)")
		if sup:
			for k, s in sup.items():
				ci_k = s.get("ci", [float("nan"), float("nan")])
				print(f"  {k}: mean={s.get('mean', float('nan')):.3f}  CrI=[{ci_k[0]:.3f}, {ci_k[1]:.3f}]")
		else:
			print("  (Not available: provide totals_A and totals_B)")
		print("")

		# D
		print("D) Decision quality / risk")
		print(f"  Expected regret if choose B: {exp_regret_B:.{digits}g}")
		print(f"  Expected regret if choose A: {exp_regret_A:.{digits}g}")
		if np.isfinite(wrong_decision_prob_proxy):
			print(f"  Wrong-decision probability proxy: {wrong_decision_prob_proxy:.3f} (= 1 - P({event_txt}))")
		print("")

		# E
		print("E) Sensitivity & reproducibility")
		print(
			f"  Report n={res['data_summary']['n']}, CRN pairing, n_draws={n_draws}, seed={random_state}, threshold={pass_threshold:.2f}, δ={delta:g}"
		)
		if rope is not None and float(rope) > 0:
			print(f"  ROPE used: ±{rope:g}")
		print(f"  {sensitivity_note}")
		print("")
		print("Verdict:")
		print(f"  {res.get('verdict','')}")

		return res if return_result else None

	# ---------------- other modes ----------------
	print(decision_line)

	if mode_l == "decision_only":
		return res if return_result else None

	if mode_l == "compact":
		print("-" * len(decision_line))
		md = post.get("mean_delta_(E[B-A])", {})
		ci = md.get("ci", [float("nan"), float("nan")])
		mean = md.get("mean", float("nan"))
		print(f"E[B-A]: mean={mean:.{digits}g}  CrI=[{ci[0]:.{digits}g}, {ci[1]:.{digits}g}]")
		for k in ("P(E[B-A] < 0)", "P(E[B-A] > 0)"):
			if k in post:
				print(f"{k}: {post[k]:.3f}")
		sup = res.get("posterior_episode_superiority_optional") or {}
		if sup:
			keys = list(sup.keys())[:2]
			for k in keys:
				s = sup[k]
				ci_k = s["ci"]
				print(f"{k}: mean={s['mean']:.3f}  CrI=[{ci_k[0]:.3f}, {ci_k[1]:.3f}]")
		return res if return_result else None

	# default: "full"
	if mode_l == "full":
		print("-" * len(decision_line))
		print(res.get("report", ""))
	else:
		raise ValueError("mode must be one of: {'full','compact','decision_only','complete_AB_report'}")

	return res if return_result else None


def bayesian_ab_test_vanilla(
	totals_A,
	totals_B,
	*,
	delta=0.0,
	better="lower",  # costs -> "lower"
	cred_level=0.95,
	rope=0.0,  # practical equivalence on mean effect
	n_draws=200_000,  # draws for mean-level posteriors
	# Episode-level superiority posteriors use an inner Monte Carlo to avoid O(n^2):
	n_draws_superiority=5_000,  # posterior draws for superiority probabilities
	m_pairs=4_000,  # inner MC pairs per draw
	percent_improvement=True,  # mean-level % improvement posterior
	eps=1e-12,
	random_state=0,
	return_draws=False,
):
	"""Vanilla Bayesian A/B test for independent samples (no CRN pairing).

	Bayesian method: Bayesian bootstrap (Dirichlet(1,...,1) weights) independently per arm.
	"""

	rng = np.random.default_rng(random_state)

	A = np.asarray(totals_A, dtype=float)
	B = np.asarray(totals_B, dtype=float)
	A = A[np.isfinite(A)]
	B = B[np.isfinite(B)]
	if A.size < 2 or B.size < 2:
		raise ValueError("Need at least 2 finite samples in each arm (A and B).")

	nA, nB = A.size, B.size
	delta = float(delta)
	rope = float(rope)

	# Credible interval quantiles
	alpha = 1.0 - float(cred_level)
	q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

	def summarize(draws: Any) -> dict[str, Any]:
		draws = np.asarray(draws, dtype=float)
		return {
			"mean": float(np.mean(draws)),
			"median": float(np.median(draws)),
			"sd": float(np.std(draws, ddof=1)),
			"ci": [float(np.quantile(draws, q_lo)), float(np.quantile(draws, q_hi))],
		}

	# ----- Bayesian bootstrap draws for mean(A), mean(B) -----
	def bb_weights(n: int, nd: int) -> np.ndarray:
		g = rng.gamma(shape=1.0, scale=1.0, size=(nd, n))
		return g / np.sum(g, axis=1, keepdims=True)

	wA = bb_weights(int(nA), int(n_draws))
	wB = bb_weights(int(nB), int(n_draws))
	mean_A = wA @ A
	mean_B = wB @ B
	mean_d = mean_B - mean_A  # Δ = E[B] - E[A]

	# Mean-level events
	if better == "lower":
		# Costs: B better means mean_d < 0; "at least delta better" means mean_d <= -delta
		dir_text = "P(E[B]-E[A] < 0)"
		prob_dir = float(np.mean(mean_d < 0.0))
		event_text = f"E[B]-E[A] <= {-delta:.6g}"
		prob_better_by_delta = float(np.mean(mean_d <= -delta))
		# mean-level regret (if choose B, regret when B worse i.e., mean_d>0)
		regret_choose_B = np.maximum(0.0, mean_d)
		regret_choose_A = np.maximum(0.0, -mean_d)
		better_label = "Lower is better (cost)"
		# mean-level percent improvement: (A - B)/|A|
		pct = (mean_A - mean_B) / (np.abs(mean_A) + float(eps))
	elif better == "higher":
		dir_text = "P(E[B]-E[A] > 0)"
		prob_dir = float(np.mean(mean_d > 0.0))
		event_text = f"E[B]-E[A] >= {delta:.6g}"
		prob_better_by_delta = float(np.mean(mean_d >= delta))
		regret_choose_B = np.maximum(0.0, -mean_d)
		regret_choose_A = np.maximum(0.0, mean_d)
		better_label = "Higher is better (reward)"
		# mean-level percent improvement: (B - A)/|A|
		pct = (mean_B - mean_A) / (np.abs(mean_A) + float(eps))
	else:
		raise ValueError("better must be 'lower' or 'higher'.")

	prob_in_rope = float(np.mean(np.abs(mean_d) <= rope)) if rope > 0 else None

	posterior = {
		"method": "bayes_bootstrap (independent arms)",
		"n_draws": int(n_draws),
		"mean_level_event": event_text,
		"mean_effect_(E[B]-E[A])": summarize(mean_d),
		dir_text: prob_dir,
		"prob_B_better_by_delta_(mean_level)": prob_better_by_delta,
		"rope": rope if rope > 0 else None,
		"prob_practically_equivalent_(|E[B]-E[A]|<=rope)": prob_in_rope,
	}

	totals_post: dict[str, Any] = {
		"E[A]": summarize(mean_A),
		"E[B]": summarize(mean_B),
	}
	if percent_improvement:
		totals_post["mean_percent_improvement_(E-level)"] = summarize(pct)

	# ----- Episode-level superiority (posterior over P(B beats A)) -----
	superiority_post = None
	if n_draws_superiority and m_pairs:
		nds = int(n_draws_superiority)
		mp = int(m_pairs)

		# weights for superiority draws
		wA_s = bb_weights(int(nA), nds)
		wB_s = bb_weights(int(nB), nds)

		# preallocate
		p_sup0 = np.empty(nds, dtype=float)
		p_supD = np.empty(nds, dtype=float)

		for k in range(nds):
			# sample indices according to bootstrap weights
			ia = rng.choice(int(nA), size=mp, replace=True, p=wA_s[k])
			ib = rng.choice(int(nB), size=mp, replace=True, p=wB_s[k])

			a_s = A[ia]
			b_s = B[ib]

			if better == "lower":
				# episode-level: B wins if b < a
				wins0 = b_s < a_s
				winsD = b_s <= (a_s - delta)
				k0 = "P(B < A) (episode-level)"
				kD = f"P(B <= A - {delta:.6g}) (episode-level)"
			else:
				wins0 = b_s > a_s
				winsD = b_s >= (a_s + delta)
				k0 = "P(B > A) (episode-level)"
				kD = f"P(B >= A + {delta:.6g}) (episode-level)"

			p_sup0[k] = float(np.mean(wins0))
			p_supD[k] = float(np.mean(winsD))

		superiority_post = {
			k0: summarize(p_sup0),
			kD: summarize(p_supD),
			"superiority_mc": {"n_draws_superiority": nds, "m_pairs": mp},
		}

	# ----- Decision quality / risk -----
	exp_regret_B = float(np.mean(regret_choose_B))
	exp_regret_A = float(np.mean(regret_choose_A))
	wrong_decision_prob_proxy = float(1.0 - prob_better_by_delta)

	# Verdict
	if prob_better_by_delta >= 0.95:
		strength = "Strong"
	elif prob_better_by_delta >= 0.80:
		strength = "Moderate"
	else:
		strength = "Weak/insufficient"
	verdict = f"{strength} evidence: P({event_text}) = {prob_better_by_delta:.3f}"

	# One-page report
	lines: list[str] = []
	lines.append("Bayesian A/B Test (vanilla, independent samples)")
	lines.append("=" * 62)
	lines.append(f"n_A={nA}, n_B={nB}   ({better_label})")
	lines.append("")
	lines.append("Data summary:")
	lines.append(f"  A: mean={np.mean(A): .6g}, sd={np.std(A, ddof=1): .6g}")
	lines.append(f"  B: mean={np.mean(B): .6g}, sd={np.std(B, ddof=1): .6g}")
	lines.append("")
	md = posterior["mean_effect_(E[B]-E[A])"]
	lines.append("A) Effect size & uncertainty (mean-level)")
	lines.append(f"  E[B]-E[A]: {cred_level*100:.1f}% CrI [{md['ci'][0]: .6g}, {md['ci'][1]: .6g}]")
	lines.append(f"  {dir_text}: {posterior[dir_text]:.3f}")
	lines.append(f"  P({event_text}): {prob_better_by_delta:.3f}")
	lines.append("")
	lines.append("B) Practical significance")
	if rope > 0:
		lines.append(f"  ROPE ±{rope:g}: P(|E[B]-E[A]|<=rope) = {float(prob_in_rope):.3f}")
	else:
		lines.append("  ROPE: (not used)")
	lines.append("")
	lines.append("C) Superiority / robustness (episode-level)")
	if superiority_post is not None:
		# show only the two key metrics
		keys = [k for k in superiority_post.keys() if k.startswith("P(")]
		for k in keys[:2]:
			s = superiority_post[k]
			ci = s["ci"]
			lines.append(f"  {k}: mean={s['mean']:.3f}  CrI=[{ci[0]:.3f}, {ci[1]:.3f}]")
	else:
		lines.append("  (Not computed: set n_draws_superiority and m_pairs > 0)")
	lines.append("")
	lines.append("D) Decision quality / risk")
	lines.append(f"  Expected regret if choose B: {exp_regret_B:.6g}")
	lines.append(f"  Expected regret if choose A: {exp_regret_A:.6g}")
	lines.append(f"  Wrong-decision probability proxy: {wrong_decision_prob_proxy:.3f} (= 1 - P({event_text}))")
	lines.append("")
	lines.append("E) Sensitivity & reproducibility")
	lines.append(f"  n_draws={n_draws}, seed={random_state}")
	if superiority_post is not None:
		mc = superiority_post["superiority_mc"]
		lines.append(f"  superiority MC: n_draws_superiority={mc['n_draws_superiority']}, m_pairs={mc['m_pairs']}")
	lines.append("")
	lines.append("Verdict:")
	lines.append(f"  {verdict}")
	report = "\n".join(lines)

	result: dict[str, Any] = {
		"inputs": {
			"better": better,
			"delta": float(delta),
			"cred_level": float(cred_level),
			"rope": rope if rope > 0 else None,
			"method": "bayes_bootstrap (independent arms)",
			"n_draws": int(n_draws),
			"n_draws_superiority": int(n_draws_superiority),
			"m_pairs": int(m_pairs),
			"random_state": int(random_state),
		},
		"data_summary": {
			"n_A": int(nA),
			"n_B": int(nB),
			"sample_mean_A": float(np.mean(A)),
			"sample_mean_B": float(np.mean(B)),
			"sample_sd_A": float(np.std(A, ddof=1)),
			"sample_sd_B": float(np.std(B, ddof=1)),
		},
		"posterior": posterior,
		"posterior_totals": totals_post,
		"posterior_episode_superiority_optional": superiority_post,
		"decision_risk": {
			"expected_regret_choose_B": exp_regret_B,
			"expected_regret_choose_A": exp_regret_A,
			"wrong_decision_prob_proxy": wrong_decision_prob_proxy,
		},
		"verdict": verdict,
		"report": report,
	}

	if return_draws:
		result["draws"] = {
			"mean_A": mean_A,
			"mean_B": mean_B,
			"mean_effect": mean_d,
			"regret_choose_B": regret_choose_B,
			"regret_choose_A": regret_choose_A,
			# store full weights only if you really need them (can be large)
		}

	return result


def print_bayesian_ab_report_vanilla(
	totals_A,
	totals_B,
	*,
	delta=0.0,
	better="lower",
	cred_level=0.95,
	rope=0.0,
	n_draws=200_000,
	n_draws_superiority=5_000,
	m_pairs=4_000,
	percent_improvement=True,
	random_state=0,
	pass_threshold=0.95,
	mode="complete_AB_report",  # {"complete_AB_report","full","compact","decision_only"}
	include_assumptions=True,
	assumptions_text=None,  # if None, auto-generated (1–2 sentences)
	sensitivity_note="(Optional) Sensitivity: rerun with different n_draws, or compare to a parametric Bayesian model.",
	digits=4,
	return_result=True,
):
	"""Printer for the vanilla (independent-samples) Bayesian A/B test."""

	res = bayesian_ab_test_vanilla(
		totals_A,
		totals_B,
		delta=delta,
		better=better,
		cred_level=cred_level,
		rope=rope,
		n_draws=n_draws,
		n_draws_superiority=n_draws_superiority,
		m_pairs=m_pairs,
		percent_improvement=percent_improvement,
		random_state=random_state,
		return_draws=True,
	)

	post = res["posterior"]
	event_txt = post["mean_level_event"]
	p = post["prob_B_better_by_delta_(mean_level)"]

	status = "PASS ✅" if p >= pass_threshold else "NO PASS —"
	decision_line = f"Decision @ {pass_threshold:.2f}: {status}   P({event_txt}) = {p:.3f}"

	if assumptions_text is None and include_assumptions:
		rope_str = f"ROPE ±{rope:g}" if (rope is not None and float(rope) > 0) else "no ROPE"
		assumptions_text = (
			"Assumptions / setup: A and B are treated as independent samples and we compare mean outcomes via Δ = E[B] − E[A]. "
			f"Posterior uncertainty is estimated with the Bayesian bootstrap (nonparametric) using practical threshold δ={delta:g} and {rope_str}."
		)

	mode_l = str(mode).lower().strip()

	if mode_l == "decision_only":
		print(decision_line)
		return res if return_result else None

	if mode_l == "compact":
		print(decision_line)
		md = post["mean_effect_(E[B]-E[A])"]
		ci = md["ci"]
		print(f"E[B]-E[A]: mean={md['mean']:.{digits}g}  CrI=[{ci[0]:.{digits}g}, {ci[1]:.{digits}g}]")
		for k in ("P(E[B]-E[A] < 0)", "P(E[B]-E[A] > 0)"):
			if k in post:
				print(f"{k}: {post[k]:.3f}")
		sup = res.get("posterior_episode_superiority_optional")
		if sup:
			keys = [k for k in sup.keys() if k.startswith("P(")]
			for k in keys[:2]:
				s = sup[k]
				ci2 = s["ci"]
				print(f"{k}: mean={s['mean']:.3f}  CrI=[{ci2[0]:.3f}, {ci2[1]:.3f}]")
		return res if return_result else None

	if mode_l in {"full", "complete_ab_report"}:
		if mode_l == "full":
			print(decision_line)
			print("-" * len(decision_line))
			print(res["report"])
			return res if return_result else None

		# complete_AB_report
		md = post["mean_effect_(E[B]-E[A])"]
		ci = md["ci"]
		risk = res["decision_risk"]
		sup = res.get("posterior_episode_superiority_optional") or {}

		print("Bayesian A/B Report (COMPLETE, independent samples)")
		print("=" * 72)
		print(f"n_A={res['data_summary']['n_A']}, n_B={res['data_summary']['n_B']}")
		print(f"Method: {res['inputs']['method']}")
		print(f"Monte Carlo: n_draws={n_draws}, seed={random_state}")
		print(f"Decision rule: Ship B if P({event_txt}) ≥ {pass_threshold:.2f}.")
		print("")
		if include_assumptions and assumptions_text:
			print(assumptions_text)
			print("")

		# A
		print("A) Effect size & uncertainty (mean-level)")
		print(
			f"  E[B]-E[A]: mean={md['mean']:.{digits}g}  {100*cred_level:.1f}% CrI=[{ci[0]:.{digits}g}, {ci[1]:.{digits}g}]"
		)
		for k in ("P(E[B]-E[A] < 0)", "P(E[B]-E[A] > 0)"):
			if k in post:
				print(f"  {k}: {post[k]:.3f}")
		print(f"  P({event_txt}): {p:.3f}")
		print("")

		# B
		print("B) Practical significance")
		if post.get("rope") is not None:
			print(
				f"  ROPE ±{post['rope']:g}: P(|E[B]-E[A]|<=rope) = {post['prob_practically_equivalent_(|E[B]-E[A]|<=rope)']:.3f}"
			)
		else:
			print("  ROPE: (not used)")
		print("")

		# C
		print("C) Superiority / robustness (episode-level)")
		keys = [k for k in sup.keys() if k.startswith("P(")]
		if keys:
			for k in keys[:2]:
				s = sup[k]
				ci_k = s["ci"]
				print(f"  {k}: mean={s['mean']:.3f}  CrI=[{ci_k[0]:.3f}, {ci_k[1]:.3f}]")
			mc = sup.get("superiority_mc", {})
			if mc:
				print(f"  (MC settings: n_draws_superiority={mc['n_draws_superiority']}, m_pairs={mc['m_pairs']})")
		else:
			print("  (Not computed: set n_draws_superiority and m_pairs > 0)")
		print("")

		# D
		print("D) Decision quality / risk")
		print(f"  Expected regret if choose B: {risk['expected_regret_choose_B']:.{digits}g}")
		print(f"  Expected regret if choose A: {risk['expected_regret_choose_A']:.{digits}g}")
		print(f"  Wrong-decision probability proxy: {risk['wrong_decision_prob_proxy']:.3f} (= 1 - P({event_txt}))")
		print("")

		# E
		print("E) Sensitivity & reproducibility")
		print(f"  n_draws={n_draws}, seed={random_state}")
		print(f"  {sensitivity_note}")
		print("")
		print("Verdict:")
		print(f"  {res['verdict']}")
		print("")
		print(decision_line)

		return res if return_result else None

	raise ValueError("mode must be one of: {'complete_AB_report','full','compact','decision_only'}")


def _erf_vec(x: Any) -> np.ndarray:
	"""Vectorized error function.

	Tries SciPy if available (fast), otherwise falls back to math.erf (portable).
	"""

	try:
		from scipy.special import erf as sp_erf  # type: ignore

		return sp_erf(x)
	except Exception:
		x = np.asarray(x, dtype=float)
		# np.vectorize uses Python loops but is fine for teaching-scale draws
		return np.vectorize(math.erf)(x)


def _norm_cdf_vec(x: Any) -> np.ndarray:
	"""Vectorized Normal CDF."""

	return 0.5 * (1.0 + _erf_vec(np.asarray(x, dtype=float) / np.sqrt(2.0)))


def bayesian_ab_test_normal(
	totals_A,
	totals_B,
	*,
	delta=0.0,
	better="lower",  # costs -> "lower"
	cred_level=0.95,
	rope=0.0,  # practical equivalence on mean effect (mean-level)
	n_draws=200_000,
	random_state=0,
	percent_improvement=True,
	ratio_and_uplift=True,
	# Conjugate prior hyperparameters (weakly-informative defaults)
	mu0_A=None,
	mu0_B=None,
	kappa0=1e-3,
	alpha0=2.0,
	beta0=None,
	return_draws=False,
):
	"""Bayesian A/B test with parametric Normal model (independent samples)."""

	rng = np.random.default_rng(random_state)

	A = np.asarray(totals_A, dtype=float)
	B = np.asarray(totals_B, dtype=float)
	A = A[np.isfinite(A)]
	B = B[np.isfinite(B)]
	if A.size < 2 or B.size < 2:
		raise ValueError("Need at least 2 finite samples in each arm (A and B).")

	nA, nB = A.size, B.size
	delta = float(delta)
	rope = float(rope)

	# Credible interval quantiles
	alpha = 1.0 - float(cred_level)
	q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

	def summarize(draws: Any) -> dict[str, Any]:
		draws = np.asarray(draws, dtype=float)
		return {
			"mean": float(np.mean(draws)),
			"median": float(np.median(draws)),
			"sd": float(np.std(draws, ddof=1)),
			"ci": [float(np.quantile(draws, q_lo)), float(np.quantile(draws, q_hi))],
		}

	# Set prior centers mu0 to sample means if not provided
	mA = float(np.mean(A))
	mB = float(np.mean(B))
	if mu0_A is None:
		mu0_A = mA
	if mu0_B is None:
		mu0_B = mB

	# Set beta0 from pooled variance if not provided
	if beta0 is None:
		vA = float(np.var(A, ddof=1))
		vB = float(np.var(B, ddof=1))
		v_pool = max(1e-12, 0.5 * (vA + vB))
		beta0 = v_pool * (float(alpha0) - 1.0)
	beta0 = float(beta0)

	def posterior_params_normal_ig(y, mu0, kappa0, alpha0, beta0):
		y = np.asarray(y, dtype=float)
		n = y.size
		ybar = float(np.mean(y))
		ssd = float(np.sum((y - ybar) ** 2))

		kappa_n = float(kappa0) + n
		mu_n = (float(kappa0) * float(mu0) + n * ybar) / kappa_n
		alpha_n = float(alpha0) + 0.5 * n
		beta_n = float(beta0) + 0.5 * ssd + (float(kappa0) * n * (ybar - float(mu0)) ** 2) / (2.0 * kappa_n)
		return mu_n, kappa_n, alpha_n, beta_n

	mu_nA, kappa_nA, alpha_nA, beta_nA = posterior_params_normal_ig(A, mu0_A, kappa0, alpha0, beta0)
	mu_nB, kappa_nB, alpha_nB, beta_nB = posterior_params_normal_ig(B, mu0_B, kappa0, alpha0, beta0)

	def sample_sigma2(alpha: float, beta: float, size: int) -> np.ndarray:
		gam = rng.gamma(shape=float(alpha), scale=1.0 / float(beta), size=int(size))
		return 1.0 / gam

	s2A = sample_sigma2(alpha_nA, beta_nA, int(n_draws))
	s2B = sample_sigma2(alpha_nB, beta_nB, int(n_draws))

	muA = rng.normal(loc=mu_nA, scale=np.sqrt(s2A / kappa_nA), size=int(n_draws))
	muB = rng.normal(loc=mu_nB, scale=np.sqrt(s2B / kappa_nB), size=int(n_draws))

	# Mean-level effect
	mean_d = muB - muA

	if better == "lower":
		dir_text = "P(E[B]-E[A] < 0)"
		prob_dir = float(np.mean(mean_d < 0.0))
		event_text = f"E[B]-E[A] <= {-delta:.6g}"
		prob_better_by_delta = float(np.mean(mean_d <= -delta))
		regret_choose_B = np.maximum(0.0, mean_d)
		regret_choose_A = np.maximum(0.0, -mean_d)
		better_label = "Lower is better (cost)"
		pct = (muA - muB) / (np.abs(muA) + 1e-12)

		z0 = (muA - muB) / np.sqrt(s2A + s2B)
		zD = (muA - muB - delta) / np.sqrt(s2A + s2B)
		sup0_name = "P(B < A) (episode-level predictive)"
		supD_name = f"P(B <= A - {delta:.6g}) (episode-level predictive)"

	elif better == "higher":
		dir_text = "P(E[B]-E[A] > 0)"
		prob_dir = float(np.mean(mean_d > 0.0))
		event_text = f"E[B]-E[A] >= {delta:.6g}"
		prob_better_by_delta = float(np.mean(mean_d >= delta))
		regret_choose_B = np.maximum(0.0, -mean_d)
		regret_choose_A = np.maximum(0.0, mean_d)
		better_label = "Higher is better (reward)"
		pct = (muB - muA) / (np.abs(muA) + 1e-12)

		z0 = (muB - muA) / np.sqrt(s2A + s2B)
		zD = (muB - muA - delta) / np.sqrt(s2A + s2B)
		sup0_name = "P(B > A) (episode-level predictive)"
		supD_name = f"P(B >= A + {delta:.6g}) (episode-level predictive)"
	else:
		raise ValueError("better must be 'lower' or 'higher'.")

	prob_in_rope = float(np.mean(np.abs(mean_d) <= rope)) if rope > 0 else None

	p_sup0 = _norm_cdf_vec(z0)
	p_supD = _norm_cdf_vec(zD)

	# Decision risk
	exp_regret_B = float(np.mean(regret_choose_B))
	exp_regret_A = float(np.mean(regret_choose_A))
	wrong_decision_prob_proxy = float(1.0 - prob_better_by_delta)

	# Verdict
	if prob_better_by_delta >= 0.95:
		strength = "Strong"
	elif prob_better_by_delta >= 0.80:
		strength = "Moderate"
	else:
		strength = "Weak/insufficient"
	verdict = f"{strength} evidence: P({event_text}) = {prob_better_by_delta:.3f}"

	posterior = {
		"method": "normal_ig_conjugate (independent arms)",
		"n_draws": int(n_draws),
		"mean_level_event": event_text,
		"mean_effect_(E[B]-E[A])": summarize(mean_d),
		dir_text: prob_dir,
		"prob_B_better_by_delta_(mean_level)": prob_better_by_delta,
		"rope": rope if rope > 0 else None,
		"prob_practically_equivalent_(|E[B]-E[A]|<=rope)": prob_in_rope,
	}

	totals_post = {
		"E[A]": summarize(muA),
		"E[B]": summarize(muB),
	}
	if percent_improvement:
		totals_post["mean_percent_improvement_(E-level)"] = summarize(pct)

	if ratio_and_uplift:
		ratio = muB / (muA + 1e-12)
		uplift = ratio - 1.0
		totals_post["ratio_of_means_(E[B]/E[A])"] = summarize(ratio)
		totals_post["uplift_percent_(E[B]/E[A]-1)"] = summarize(uplift)

	superiority_post = {
		sup0_name: summarize(p_sup0),
		supD_name: summarize(p_supD),
		"note": "Episode-level predictive superiority under Normal model; integrates uncertainty in (mu, sigma^2).",
	}

	# One-page report
	md = posterior["mean_effect_(E[B]-E[A])"]
	lines: list[str] = []
	lines.append("Bayesian A/B Test (parametric Normal, independent samples)")
	lines.append("=" * 68)
	lines.append(f"n_A={nA}, n_B={nB}   ({better_label})")
	lines.append(f"Prior per arm: Normal–InvGamma with kappa0={kappa0:g}, alpha0={alpha0:g}, beta0={beta0:.6g}")
	lines.append("")
	lines.append("A) Effect size & uncertainty (mean-level)")
	lines.append(
		f"  E[B]-E[A]: mean={md['mean']:.6g}  {cred_level*100:.1f}% CrI [{md['ci'][0]:.6g}, {md['ci'][1]:.6g}]"
	)
	lines.append(f"  {dir_text}: {prob_dir:.3f}")
	lines.append(f"  P({event_text}): {prob_better_by_delta:.3f}")
	lines.append("")
	lines.append("B) Practical significance")
	if rope > 0:
		lines.append(f"  ROPE ±{rope:g}: P(|E[B]-E[A]|<=rope) = {float(prob_in_rope):.3f}")
	else:
		lines.append("  ROPE: (not used)")
	lines.append("")
	lines.append("C) Superiority / robustness (episode-level predictive)")
	for k in (sup0_name, supD_name):
		s = superiority_post[k]
		ci = s["ci"]
		lines.append(f"  {k}: mean={s['mean']:.3f}  CrI=[{ci[0]:.3f}, {ci[1]:.3f}]")
	lines.append("")
	lines.append("D) Decision quality / risk")
	lines.append(f"  Expected regret if choose B: {exp_regret_B:.6g}")
	lines.append(f"  Expected regret if choose A: {exp_regret_A:.6g}")
	lines.append(f"  Wrong-decision probability proxy: {wrong_decision_prob_proxy:.3f} (= 1 - P({event_text}))")
	lines.append("")
	lines.append("E) Sensitivity & reproducibility")
	lines.append(f"  n_draws={n_draws}, seed={random_state}")
	lines.append("")
	lines.append("Verdict:")
	lines.append(f"  {verdict}")
	report = "\n".join(lines)

	result: dict[str, Any] = {
		"inputs": {
			"better": better,
			"delta": float(delta),
			"cred_level": float(cred_level),
			"rope": rope if rope > 0 else None,
			"method": "normal_ig_conjugate (independent arms)",
			"n_draws": int(n_draws),
			"random_state": int(random_state),
			"prior": {
				"mu0_A": float(mu0_A),
				"mu0_B": float(mu0_B),
				"kappa0": float(kappa0),
				"alpha0": float(alpha0),
				"beta0": float(beta0),
			},
		},
		"data_summary": {
			"n_A": int(nA),
			"n_B": int(nB),
			"sample_mean_A": float(np.mean(A)),
			"sample_mean_B": float(np.mean(B)),
			"sample_sd_A": float(np.std(A, ddof=1)),
			"sample_sd_B": float(np.std(B, ddof=1)),
		},
		"posterior": posterior,
		"posterior_totals": totals_post,
		"posterior_episode_superiority_predictive": superiority_post,
		"decision_risk": {
			"expected_regret_choose_B": exp_regret_B,
			"expected_regret_choose_A": exp_regret_A,
			"wrong_decision_prob_proxy": wrong_decision_prob_proxy,
		},
		"verdict": verdict,
		"report": report,
	}

	if return_draws:
		result["draws"] = {
			"muA": muA,
			"muB": muB,
			"s2A": s2A,
			"s2B": s2B,
			"mean_effect": mean_d,
			"p_sup0": p_sup0,
			"p_supD": p_supD,
			"regret_choose_A": regret_choose_A,
			"regret_choose_B": regret_choose_B,
		}
		if ratio_and_uplift:
			result["draws"]["ratio_of_means"] = ratio
			result["draws"]["uplift"] = uplift

	return result


def print_bayesian_ab_report_normal(
	totals_A,
	totals_B,
	*,
	delta=0.0,
	better="lower",
	cred_level=0.95,
	rope=0.0,
	n_draws=200_000,
	random_state=0,
	percent_improvement=True,
	ratio_and_uplift=True,
	pass_threshold=0.95,
	mode="complete_AB_report",  # {"complete_AB_report","full","compact","decision_only"}
	include_assumptions=True,
	assumptions_text=None,
	sensitivity_note="(Optional) Sensitivity: rerun with different priors (kappa0/alpha0/beta0) and n_draws.",
	digits=4,
	# Priors
	mu0_A=None,
	mu0_B=None,
	kappa0=1e-3,
	alpha0=2.0,
	beta0=None,
	return_result=True,
):
	"""Print A–E complete report for the parametric Normal Bayesian A/B test."""

	res = bayesian_ab_test_normal(
		totals_A,
		totals_B,
		delta=delta,
		better=better,
		cred_level=cred_level,
		rope=rope,
		n_draws=n_draws,
		random_state=random_state,
		percent_improvement=percent_improvement,
		ratio_and_uplift=ratio_and_uplift,
		mu0_A=mu0_A,
		mu0_B=mu0_B,
		kappa0=kappa0,
		alpha0=alpha0,
		beta0=beta0,
		return_draws=False,
	)

	post = res["posterior"]
	totals_post = res["posterior_totals"]
	event_txt = post["mean_level_event"]
	p = post["prob_B_better_by_delta_(mean_level)"]
	status = "PASS ✅" if p >= pass_threshold else "NO PASS —"
	decision_line = f"Decision @ {pass_threshold:.2f}: {status}   P({event_txt}) = {p:.3f}"

	if assumptions_text is None and include_assumptions:
		rope_str = f"ROPE ±{rope:g}" if (rope is not None and float(rope) > 0) else "no ROPE"
		assumptions_text = (
			"Assumptions / setup: A and B are treated as independent samples and each arm is modeled as Normal with unknown mean and variance. "
			f"We compare the mean effect Δ = E[B] − E[A] using a conjugate Normal–Inverse-Gamma prior, with practical threshold δ={delta:g} and {rope_str}."
		)

	mode_l = str(mode).lower().strip()

	if mode_l == "decision_only":
		print(decision_line)
		return res if return_result else None

	if mode_l == "compact":
		md = post["mean_effect_(E[B]-E[A])"]
		ci = md["ci"]
		print(decision_line)
		print(f"E[B]-E[A]: mean={md['mean']:.{digits}g}  CrI=[{ci[0]:.{digits}g}, {ci[1]:.{digits}g}]")
		for k in ("P(E[B]-E[A] < 0)", "P(E[B]-E[A] > 0)"):
			if k in post:
				print(f"{k}: {post[k]:.3f}")
		if ratio_and_uplift and "ratio_of_means_(E[B]/E[A])" in totals_post:
			r = totals_post["ratio_of_means_(E[B]/E[A])"]
			u = totals_post["uplift_percent_(E[B]/E[A]-1)"]
			print(f"E[B]/E[A]: mean={r['mean']:.3g}  CrI=[{r['ci'][0]:.3g}, {r['ci'][1]:.3g}]")
			print(f"uplift: mean={100*u['mean']:.3g}%  CrI=[{100*u['ci'][0]:.3g}%, {100*u['ci'][1]:.3g}%]")
		return res if return_result else None

	if mode_l == "full":
		print(decision_line)
		print("-" * len(decision_line))
		print(res["report"])
		return res if return_result else None

	if mode_l != "complete_ab_report":
		raise ValueError("mode must be one of: {'complete_AB_report','full','compact','decision_only'}")

	# complete_AB_report
	md = post["mean_effect_(E[B]-E[A])"]
	ci = md["ci"]
	risk = res["decision_risk"]
	sup = res["posterior_episode_superiority_predictive"]
	prior = res["inputs"]["prior"]

	print("Bayesian A/B Report (COMPLETE, Normal model, independent samples)")
	print("=" * 76)
	print(f"n_A={res['data_summary']['n_A']}, n_B={res['data_summary']['n_B']}")
	print(f"Method: {res['inputs']['method']}")
	print(f"Prior: kappa0={prior['kappa0']}, alpha0={prior['alpha0']}, beta0={prior['beta0']:.6g}")
	print(f"Monte Carlo: n_draws={n_draws}, seed={random_state}")
	print(f"Decision rule: Ship B if P({event_txt}) ≥ {pass_threshold:.2f}.")
	print("")
	if include_assumptions and assumptions_text:
		print(assumptions_text)
		print("")

	# A
	print("A) Effect size & uncertainty (mean-level)")
	print(f"  E[B]-E[A]: mean={md['mean']:.{digits}g}  {100*cred_level:.1f}% CrI=[{ci[0]:.{digits}g}, {ci[1]:.{digits}g}]")
	for k in ("P(E[B]-E[A] < 0)", "P(E[B]-E[A] > 0)"):
		if k in post:
			print(f"  {k}: {post[k]:.3f}")
	print(f"  P({event_txt}): {p:.3f}")
	print("")

	# B
	print("B) Practical significance")
	if post.get("rope") is not None:
		print(f"  ROPE ±{post['rope']:g}: P(|E[B]-E[A]|<=rope) = {post['prob_practically_equivalent_(|E[B]-E[A]|<=rope)']:.3f}")
	else:
		print("  ROPE: (not used)")
	print("")

	# (optional) mean totals, ratio, uplift
	print("Additional mean-level summaries")
	EA = totals_post["E[A]"]
	EB = totals_post["E[B]"]
	print(f"  E[A]: mean={EA['mean']:.{digits}g}  CrI=[{EA['ci'][0]:.{digits}g}, {EA['ci'][1]:.{digits}g}]")
	print(f"  E[B]: mean={EB['mean']:.{digits}g}  CrI=[{EB['ci'][0]:.{digits}g}, {EB['ci'][1]:.{digits}g}]")
	if percent_improvement and "mean_percent_improvement_(E-level)" in totals_post:
		pi = totals_post["mean_percent_improvement_(E-level)"]
		print(f"  % improvement: mean={100*pi['mean']:.{digits}g}%  CrI=[{100*pi['ci'][0]:.{digits}g}%, {100*pi['ci'][1]:.{digits}g}%]")
	if ratio_and_uplift and "ratio_of_means_(E[B]/E[A])" in totals_post:
		r = totals_post["ratio_of_means_(E[B]/E[A])"]
		u = totals_post["uplift_percent_(E[B]/E[A]-1)"]
		print(f"  E[B]/E[A]: mean={r['mean']:.{digits}g}  CrI=[{r['ci'][0]:.{digits}g}, {r['ci'][1]:.{digits}g}]")
		print(f"  uplift: mean={100*u['mean']:.{digits}g}%  CrI=[{100*u['ci'][0]:.{digits}g}%, {100*u['ci'][1]:.{digits}g}%]")
	print("")

	# C
	print("C) Superiority / robustness (episode-level predictive)")
	for k in (k for k in sup.keys() if str(k).startswith("P(")):
		s = sup[k]
		ci_k = s["ci"]
		print(f"  {k}: mean={s['mean']:.3f}  CrI=[{ci_k[0]:.3f}, {ci_k[1]:.3f}]")
	print("")

	# D
	print("D) Decision quality / risk")
	print(f"  Expected regret if choose B: {risk['expected_regret_choose_B']:.{digits}g}")
	print(f"  Expected regret if choose A: {risk['expected_regret_choose_A']:.{digits}g}")
	print(f"  Wrong-decision probability proxy: {risk['wrong_decision_prob_proxy']:.3f} (= 1 - P({event_txt}))")
	print("")

	# E
	print("E) Sensitivity & reproducibility")
	print(f"  n_draws={n_draws}, seed={random_state}")
	print(f"  {sensitivity_note}")
	print("")
	print("Verdict:")
	print(f"  {res['verdict']}")
	print("")
	print(decision_line)

	return res if return_result else None
