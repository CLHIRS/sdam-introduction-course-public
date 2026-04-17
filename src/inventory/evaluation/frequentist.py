from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PairedDeltaReport:
    baseline: str
    other: str
    mean_delta: float
    std_delta: float
    win_rate: float


Objective = Literal["cost", "revenue"]  # cost: lower better; revenue: higher better
DeltaConvention = Literal["treatment-control", "control-treatment"]
Alternative = Literal["less", "greater", "two-sided"]


@dataclass(frozen=True)
class DeltaTestDirection:
    objective: Objective
    delta_convention: DeltaConvention
    delta_definition: str
    treatment_better_condition: str
    one_sided_alternative: Literal["less", "greater"]
    hypotheses_one_sided_vs_0: Tuple[str, str]  # (H0, H1)


def infer_direction_for_treatment_better(
    *,
    objective: Objective,
    delta_convention: DeltaConvention = "treatment-control",
) -> DeltaTestDirection:
    """Infer correct one-sided alternative for the confirmatory claim "B better than A".

    Control = A, Treatment = B.

    objective:
      - "cost": lower is better
      - "revenue": higher is better

    delta_convention:
      - "treatment-control": Δ = B - A
      - "control-treatment": Δ = A - B
    """

    if objective not in ("cost", "revenue"):
        raise ValueError("objective must be 'cost' or 'revenue'.")
    if delta_convention not in ("treatment-control", "control-treatment"):
        raise ValueError("delta_convention must be 'treatment-control' or 'control-treatment'.")

    if delta_convention == "treatment-control":
        delta_def = "Δ = Treatment - Control (B - A)"
        if objective == "cost":
            cond = "Δ < 0"
            alt: Literal["less", "greater"] = "less"
        else:
            cond = "Δ > 0"
            alt = "greater"
    else:
        delta_def = "Δ = Control - Treatment (A - B)"
        if objective == "cost":
            cond = "Δ > 0"
            alt = "greater"
        else:
            cond = "Δ < 0"
            alt = "less"

    h0 = "H0: E[Δ] = 0  (no average difference)"
    h1 = f"H1: E[Δ] {'<' if alt=='less' else '>'} 0  (B better than A, because B better ⇔ {cond})"

    return DeltaTestDirection(
        objective=objective,
        delta_convention=delta_convention,
        delta_definition=delta_def,
        treatment_better_condition=f"B better than A ⇔ {cond}",
        one_sided_alternative=alt,
        hypotheses_one_sided_vs_0=(h0, h1),
    )


def _format_p(p: float, decimals: int = 4, sci_floor: float = 1e-4) -> str:
    """Format p-values as fixed decimals, with a scientific floor for very small p."""

    if p is None or (isinstance(p, float) and (np.isnan(p) or np.isinf(p))):
        return "NA"
    p = float(p)
    base = f"{p:.{decimals}f}"
    if p < sci_floor:
        exp = int(round(-np.log10(sci_floor)))
        return f"{base} (<1e-{exp})"
    return base


def paired_delta(a: np.ndarray, b: np.ndarray, *, higher_is_better: bool) -> Tuple[float, float, float]:
    """Paired delta statistics for strict-CRN totals.

    Args:
      a: totals for baseline
      b: totals for other
      higher_is_better: interpret "win" as b > a if True, else b < a.

    Returns:
      (mean_delta, std_delta, win_rate)
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired_delta requires equal shapes; got {a.shape} vs {b.shape}")
    d = b - a
    mean = float(np.mean(d)) if len(d) else float("nan")
    std = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    if len(d) == 0:
        win = float("nan")
    else:
        win = float(np.mean(d > 0.0)) if higher_is_better else float(np.mean(d < 0.0))
    return mean, std, win


def paired_deltas_against_baseline(
    totals_by_policy: Dict[str, np.ndarray],
    *,
    baseline_name: str,
    higher_is_better: bool,
) -> Dict[str, PairedDeltaReport]:
    base = totals_by_policy[baseline_name]
    out: Dict[str, PairedDeltaReport] = {}
    for name, tot in totals_by_policy.items():
        if name == baseline_name:
            continue
        m, s, w = paired_delta(base, tot, higher_is_better=higher_is_better)
        out[name] = PairedDeltaReport(
            baseline=baseline_name,
            other=name,
            mean_delta=float(m),
            std_delta=float(s),
            win_rate=float(w),
        )
    return out


def frequentist_analysis_AB(
    deltas: Sequence[float],
    *,
    delta_convention: DeltaConvention = "treatment-control",
    objective: Objective = "cost",
    claim: Literal["treatment_better", "two_sided"] = "treatment_better",
    alpha: float = 0.05,
    normality_alpha: float = 0.05,
    n_perm: int = 20000,
    random_state: Optional[int] = 0,
    recommend: Literal["auto", "ttest", "permutation", "wilcoxon", "sign"] = "auto",
    practical_delta: float = 0.0,
    n_boot: int = 20000,
) -> Dict[str, Any]:
    """One-stop frequentist analysis for paired CRN deltas (Control=A vs Treatment=B)."""

    try:
        from scipy import stats
    except Exception as e:  # pragma: no cover
        raise ImportError("frequentist_analysis_AB requires scipy (scipy.stats).") from e

    x = np.asarray(deltas, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 2:
        raise ValueError(f"Need at least 2 paired deltas, got n={n}.")

    rng = np.random.default_rng(random_state)
    dir_info = infer_direction_for_treatment_better(objective=objective, delta_convention=delta_convention)

    if claim == "two_sided":
        alternative: Alternative = "two-sided"
    else:
        alternative = dir_info.one_sided_alternative

    b_better_is_delta_negative = dir_info.one_sided_alternative == "less"

    mean = float(np.mean(x))
    median = float(np.median(x))
    std = float(np.std(x, ddof=1))
    se = float(std / np.sqrt(n))
    minv = float(np.min(x))
    maxv = float(np.max(x))
    dz = float(mean / std) if std > 0 else (np.inf if mean != 0 else 0.0)

    if b_better_is_delta_negative:
        n_win = int(np.sum(x < 0))
        n_loss = int(np.sum(x > 0))
        win_rate = float(n_win / n)
        win_rule_str = "B better ⇔ Δ < 0"
    else:
        n_win = int(np.sum(x > 0))
        n_loss = int(np.sum(x < 0))
        win_rate = float(n_win / n)
        win_rule_str = "B better ⇔ Δ > 0"
    n_tie = int(np.sum(x == 0))
    tie_rate = float(n_tie / n)

    try:
        shapiro_stat, shapiro_p = stats.shapiro(x)
        shapiro_stat = float(shapiro_stat)
        shapiro_p = float(shapiro_p)
        normality_reject = bool(shapiro_p < normality_alpha)
    except Exception:
        shapiro_stat, shapiro_p, normality_reject = np.nan, np.nan, False

    t_res = stats.ttest_1samp(x, popmean=0.0, alternative=alternative)
    t_stat = float(t_res.statistic)
    t_p = float(t_res.pvalue)
    df = n - 1
    tcrit = stats.t.ppf(1.0 - alpha / 2.0, df)
    ci_mean_t = (float(mean - tcrit * se), float(mean + tcrit * se))

    wilcox_out: Dict[str, Any] = {"ok": False}
    try:
        w_res = stats.wilcoxon(x, alternative=alternative, zero_method="wilcox", correction=False)
        wilcox_out = {
            "ok": True,
            "statistic": float(w_res.statistic),
            "pvalue": float(w_res.pvalue),
            "note": "Drops zero differences; assumes symmetric deltas.",
        }
    except Exception as e:
        wilcox_out = {"ok": False, "error": str(e)}

    n_pos = int(np.sum(x > 0))
    n_neg = int(np.sum(x < 0))
    n_eff = n_pos + n_neg
    if n_eff == 0:
        sign_p = 1.0
        sign_note = "All deltas are zero; no evidence of difference."
    else:
        if alternative == "greater":
            sign_p = float(stats.binom.sf(n_pos - 1, n_eff, 0.5))
        elif alternative == "less":
            sign_p = float(stats.binom.sf(n_neg - 1, n_eff, 0.5))
        else:
            p_le = float(stats.binom.cdf(n_pos, n_eff, 0.5))
            p_ge = float(stats.binom.sf(n_pos - 1, n_eff, 0.5))
            sign_p = min(1.0, 2.0 * min(p_le, p_ge))
        sign_note = "Exact binomial sign test (zeros ignored)."

    sign_out = {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": int(np.sum(x == 0)),
        "n_effective": int(n_eff),
        "pvalue": float(sign_p),
        "note": sign_note,
    }

    obs = mean
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n), replace=True)
    perm_means = (signs * x).mean(axis=1)

    if alternative == "less":
        perm_p = float((np.sum(perm_means <= obs) + 1) / (n_perm + 1))
    elif alternative == "greater":
        perm_p = float((np.sum(perm_means >= obs) + 1) / (n_perm + 1))
    else:
        perm_p = float((np.sum(np.abs(perm_means) >= abs(obs)) + 1) / (n_perm + 1))

    perm_out = {
        "statistic": float(obs),
        "pvalue": float(perm_p),
        "n_perm": int(n_perm),
        "note": "Monte Carlo sign-flip randomization test on mean(Δ).",
    }

    practical_delta = float(practical_delta)
    idx = rng.integers(0, n, size=(n_boot, n))
    xb = x[idx]
    boot_mean = xb.mean(axis=1)
    boot_median = np.median(xb, axis=1)
    boot_winrate = (xb < 0).mean(axis=1) if b_better_is_delta_negative else (xb > 0).mean(axis=1)

    def _pct_ci(samples: np.ndarray) -> tuple[float, float]:
        lo = float(np.quantile(samples, alpha / 2.0))
        hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
        return lo, hi

    ci_mean_boot = _pct_ci(boot_mean)
    ci_median_boot = _pct_ci(boot_median)
    ci_winrate_boot = _pct_ci(boot_winrate)

    p_improve = float(np.mean(boot_mean < 0.0)) if b_better_is_delta_negative else float(np.mean(boot_mean > 0.0))

    if practical_delta > 0:
        p_practical = float(np.mean(boot_mean <= -practical_delta)) if b_better_is_delta_negative else float(np.mean(boot_mean >= practical_delta))
        practical_definition = (
            f"B practically better by ≥ {practical_delta:g} means E[Δ] ≤ -{practical_delta:g}."
            if b_better_is_delta_negative
            else f"B practically better by ≥ {practical_delta:g} means E[Δ] ≥ +{practical_delta:g}."
        )
    else:
        p_practical = np.nan
        practical_definition = "No practical margin specified."

    bootstrap_out = {
        "n_boot": int(n_boot),
        "ci_level": float(1.0 - alpha),
        "mean_delta": {"estimate": mean, "ci": ci_mean_boot},
        "median_delta": {"estimate": median, "ci": ci_median_boot},
        "win_rate_B_beats_A": {"estimate": win_rate, "ci": ci_winrate_boot},
        "prob_mean_improvement_B_beats_A": p_improve,
        "practical_delta": practical_delta,
        "prob_practically_meaningful_B_beats_A": float(p_practical) if np.isfinite(p_practical) else p_practical,
        "practical_definition": practical_definition,
    }

    if recommend == "ttest":
        rec_test, rec_p = "paired_t_test_on_deltas", t_p
    elif recommend == "permutation":
        rec_test, rec_p = "sign_flip_permutation_test", perm_p
    elif recommend == "wilcoxon":
        rec_test = "wilcoxon_signed_rank"
        rec_p = wilcox_out["pvalue"] if wilcox_out.get("ok") else np.nan
    elif recommend == "sign":
        rec_test, rec_p = "sign_test", sign_out["pvalue"]
    else:
        if (claim == "two_sided") or ((not normality_reject) and np.isfinite(shapiro_p)):
            rec_test, rec_p = "paired_t_test_on_deltas", t_p
        else:
            rec_test, rec_p = "sign_flip_permutation_test", perm_p

    reject = bool(rec_p < alpha) if np.isfinite(rec_p) else False

    if claim == "two_sided":
        h0 = "H0: E[Δ] = 0  (no average difference)"
        h1 = "H1: E[Δ] ≠ 0  (difference exists)"
        claim_str = "two-sided difference"
    else:
        h0, h1 = dir_info.hypotheses_one_sided_vs_0
        claim_str = "B better than A (one-sided, pre-specified)"

    decision = (
        "Reject H0 (working conclusion: evidence supports H1)"
        if reject
        else "Fail to reject H0 (working conclusion: insufficient evidence for H1; does NOT prove H0)"
    )

    return {
        "setup": {
            "control": "A",
            "treatment": "B",
            "objective": objective,
            "delta_convention": delta_convention,
            "delta_definition": dir_info.delta_definition,
            "b_better_condition": dir_info.treatment_better_condition,
            "claim": claim_str,
            "alpha": float(alpha),
            "alternative": alternative,
            "hypotheses": {"H0": h0, "H1": h1},
            "win_rate_rule": win_rule_str,
        },
        "descriptives": {
            "n": n,
            "mean": mean,
            "median": median,
            "std": std,
            "se": se,
            "min": minv,
            "max": maxv,
            "cohen_dz": dz,
            "win_rate_B_beats_A": win_rate,
            "tie_rate": tie_rate,
            "n_win": n_win,
            "n_loss": n_loss,
            "n_tie": n_tie,
        },
        "normality": {
            "test": "Shapiro-Wilk on deltas",
            "alpha": float(normality_alpha),
            "statistic": shapiro_stat,
            "pvalue": shapiro_p,
            "reject_normality": bool(normality_reject),
        },
        "tests": {
            "paired_t_test_on_deltas": {
                "statistic": t_stat,
                "df": df,
                "pvalue": t_p,
                "ci_mean_two_sided": ci_mean_t,
            },
            "wilcoxon_signed_rank": wilcox_out,
            "sign_test": sign_out,
            "sign_flip_permutation_test": perm_out,
        },
        "bootstrap": bootstrap_out,
        "recommendation": {
            "recommended_test": rec_test,
            "pvalue": float(rec_p) if np.isfinite(rec_p) else rec_p,
            "reject_H0": reject,
            "decision": decision,
        },
    }


def format_frequentist_report_AB(res: Dict[str, Any]) -> str:
    setup = res["setup"]
    desc = res["descriptives"]
    norm = res["normality"]
    tests = res["tests"]
    boot = res["bootstrap"]
    rec = res["recommendation"]

    def fnum(v: Any, digits: int = 4) -> str:
        if v is None:
            return "NA"
        if isinstance(v, (float, np.floating)):
            if np.isnan(v) or np.isinf(v):
                return "NA"
            a = float(v)
            return f"{a:.{digits}g}" if abs(a) >= 1e4 else f"{a:.{digits}f}"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return str(v)

    unit = "cost units" if setup["objective"] == "cost" else "revenue units"

    t = tests["paired_t_test_on_deltas"]
    w = tests["wilcoxon_signed_rank"]
    s = tests["sign_test"]
    p = tests["sign_flip_permutation_test"]

    lines: list[str] = []
    lines.append("=== Frequentist CRN Paired-Delta Analysis (Treatment=B vs Control=A) ===")
    lines.append(f"Objective: {setup['objective']} | Delta convention: {setup['delta_convention']}")
    lines.append(f"Delta definition: {setup['delta_definition']}")
    lines.append(f"Direction: {setup['b_better_condition']} | Win-rate rule: {setup['win_rate_rule']}")
    lines.append(f"Claim: {setup['claim']}")
    lines.append(f"Alternative used: {setup['alternative']} | alpha={fnum(setup['alpha'],3)}")
    lines.append("")
    lines.append("--- Hypotheses ---")
    lines.append(setup["hypotheses"]["H0"])
    lines.append(setup["hypotheses"]["H1"])
    lines.append("")

    lines.append("--- Descriptives (deltas) ---")
    lines.append(
        f"n={desc['n']} | mean={fnum(desc['mean'])} | median={fnum(desc['median'])} | std={fnum(desc['std'])} | dz={fnum(desc['cohen_dz'])}"
    )
    lines.append(f"min={fnum(desc['min'])} | max={fnum(desc['max'])}")
    lines.append(
        f"win rate (B beats A)={fnum(desc['win_rate_B_beats_A'])} (wins={desc['n_win']}, losses={desc['n_loss']}, ties={desc['n_tie']})"
    )
    lines.append("")

    lines.append("--- Normality check (Shapiro on deltas) ---")
    lines.append(
        f"stat={fnum(norm['statistic'])} | p={_format_p(float(norm['pvalue']))} | reject@{fnum(norm['alpha'],3)}={norm['reject_normality']}"
    )
    lines.append("")

    lines.append("--- Hypothesis tests ---")
    ci_t = t["ci_mean_two_sided"]
    lines.append(
        f"Paired t-test (Δ vs 0): t={fnum(t['statistic'])} (df={t['df']}) | p={_format_p(float(t['pvalue']))} "
        f"| CI(mean)=[{fnum(ci_t[0])}, {fnum(ci_t[1])}]"
    )
    if w.get("ok"):
        lines.append(f"Wilcoxon: W={fnum(w['statistic'])} | p={_format_p(float(w['pvalue']))}")
    else:
        lines.append(f"Wilcoxon: NA ({w.get('error','not available')})")
    lines.append(f"Sign test: p={_format_p(float(s['pvalue']))} | n_eff={s['n_effective']} (zeros ignored)")
    lines.append(f"Permutation (sign-flip mean): p={_format_p(float(p['pvalue']))} (n_perm={p['n_perm']})")
    lines.append("")

    lines.append("--- Paired bootstrap uncertainty ---")
    ci_mean = boot["mean_delta"]["ci"]
    ci_med = boot["median_delta"]["ci"]
    ci_wr = boot["win_rate_B_beats_A"]["ci"]
    lines.append(f"CI(mean)=[{fnum(ci_mean[0])}, {fnum(ci_mean[1])}]")
    lines.append(f"CI(median)=[{fnum(ci_med[0])}, {fnum(ci_med[1])}]")
    lines.append(f"CI(win rate)=[{fnum(ci_wr[0])}, {fnum(ci_wr[1])}]")
    lines.append(f"P(mean improvement: B beats A)={fnum(boot['prob_mean_improvement_B_beats_A'])}")
    if boot["practical_delta"] and boot["practical_delta"] > 0:
        lines.append(boot["practical_definition"])
        lines.append(f"P(practically meaningful: B beats A)={fnum(boot['prob_practically_meaningful_B_beats_A'])}")
    lines.append("")

    lines.append("--- Recommendation ---")
    lines.append(f"Recommended test: {rec['recommended_test']} | p={_format_p(float(rec['pvalue']))}")
    lines.append(f"Decision: {rec['decision']} | reject_H0={rec['reject_H0']}")
    lines.append(
        f"Effect summary: mean Δ = {fnum(desc['mean'])} {unit} | "
        f"95% CI(mean) = [{fnum(ci_mean[0])}, {fnum(ci_mean[1])}] {unit} | "
        f"win rate (B beats A) = {fnum(desc['win_rate_B_beats_A'])}"
    )
    return "\n".join(lines)


def print_frequentist_report_AB(
    deltas: Sequence[float],
    *,
    delta_convention: DeltaConvention = "treatment-control",
    objective: Objective = "cost",
    claim: Literal["treatment_better", "two_sided"] = "treatment_better",
    alpha: float = 0.05,
    normality_alpha: float = 0.05,
    n_perm: int = 20000,
    n_boot: int = 20000,
    random_state: Optional[int] = 0,
    recommend: Literal["auto", "ttest", "permutation", "wilcoxon", "sign"] = "auto",
    practical_delta: float = 0.0,
) -> Dict[str, Any]:
    res = frequentist_analysis_AB(
        deltas,
        delta_convention=delta_convention,
        objective=objective,
        claim=claim,
        alpha=alpha,
        normality_alpha=normality_alpha,
        n_perm=n_perm,
        n_boot=n_boot,
        random_state=random_state,
        recommend=recommend,
        practical_delta=practical_delta,
    )
    print(format_frequentist_report_AB(res))
    return res


__all__ = [
    "PairedDeltaReport",
    "paired_delta",
    "paired_deltas_against_baseline",
    "frequentist_analysis_AB",
    "format_frequentist_report_AB",
    "print_frequentist_report_AB",
]
