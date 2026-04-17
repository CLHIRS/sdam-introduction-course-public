from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


def print_totals_summary(
    totals_by_policy: Mapping[str, Any],
    ddof: int = 1,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
    float_fmt: str = ".2f",
) -> Dict[str, Dict[str, float]]:
    """Print a compact summary of totals for each policy.

    Parameters
    ----------
    totals_by_policy : dict[str, array_like]
        Mapping policy name -> 1D array of episode total costs.
    ddof : int, default 1
        Delta degrees of freedom for the standard deviation (sample std uses 1).
    quantiles : tuple[float, ...], default (0.25, 0.5, 0.75)
        Quantiles to report (in [0, 1]).
    float_fmt : str, default ".2f"
        Format specifier for floats (without leading ':').

    Returns
    -------
    summary : dict[str, dict[str, float]]
        Nested dict with summary stats per policy.
    """

    if not isinstance(totals_by_policy, Mapping) or len(totals_by_policy) == 0:
        raise ValueError("totals_by_policy must be a non-empty mapping name -> totals array")

    names = list(totals_by_policy.keys())
    rows: list[tuple[str, Dict[str, float]]] = []
    summary: Dict[str, Dict[str, float]] = {}

    for name in names:
        arr = np.asarray(totals_by_policy[name], dtype=float).ravel()
        if arr.ndim != 1:
            raise ValueError(f"Totals for policy '{name}' must be 1D after ravel(); got shape {arr.shape}")
        if arr.size == 0:
            raise ValueError(f"Totals for policy '{name}' is empty")

        qs = np.quantile(arr, quantiles)
        row: Dict[str, float] = {
            "n": float(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=ddof)) if arr.size > ddof else float("nan"),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
        for q, v in zip(quantiles, qs):
            row[f"q{int(round(100 * q))}"] = float(v)
        summary[name] = row
        rows.append((name, row))

    q_cols = [f"q{int(round(100 * q))}" for q in quantiles]
    cols = ["n", "mean", "std", *q_cols, "min", "max"]
    header = ["policy", *cols]
    table: list[list[Any]] = []
    for name, row in rows:
        table.append(
            [
                name,
                int(row["n"]),
                format(row["mean"], float_fmt),
                format(row["std"], float_fmt) if np.isfinite(row["std"]) else "nan",
                *[format(row[c], float_fmt) for c in q_cols],
                format(row["min"], float_fmt),
                format(row["max"], float_fmt),
            ]
        )

    widths = [len(h) for h in header]
    for r in table:
        for j, val in enumerate(r):
            widths[j] = max(widths[j], len(str(val)))

    def _fmt_row(r: Sequence[Any]) -> str:
        return "  ".join(str(val).rjust(widths[j]) for j, val in enumerate(r))

    print(_fmt_row(header))
    print("  ".join("-" * w for w in widths))
    for r in table:
        print(_fmt_row(r))

    return summary


def crn_delta_totals_from_totals_by_policy(
    totals_by_policy: Mapping[str, Any],
    base_policy: str,
    other_policy: str,
    *,
    plot: bool = True,
    figsize: tuple[float, float] = (10, 3),
    title: Optional[str] = None,
    zero_line: bool = True,
    return_fig: bool = False,
):
    """Compute pairwise CRN deltas of episode TOTAL costs between two policies.

    Delta definition (episode-by-episode):
        delta[i] = totals_other[i] - totals_base[i]

    Parameters
    ----------
    totals_by_policy : dict[str, array_like]
        Mapping policy name -> 1D array of episode total costs (aligned by episode index).
    base_policy : str
        Name of the baseline policy (subtracted).
    other_policy : str
        Name of the comparison policy.
    plot : bool, default True
        If True, plots delta as a line chart.
    figsize : tuple, default (10, 3)
        Figure size for the plot (if plot=True).
    title : str | None
        Plot title override.
    zero_line : bool, default True
        If True, draws a horizontal y=0 reference line.
    return_fig : bool, default False
        If True and plot=True, returns (deltas, fig, ax). Otherwise returns deltas.

    Returns
    -------
    deltas : np.ndarray, shape (n_episodes,)
        Episode-by-episode total-cost deltas (other - base).

    Notes
    -----
    This is intended for paired totals where episodes are aligned by CRN seed / episode index.
    """

    if not isinstance(totals_by_policy, Mapping) or len(totals_by_policy) == 0:
        raise ValueError("totals_by_policy must be a non-empty mapping name -> totals array")
    if base_policy not in totals_by_policy:
        raise KeyError(f"base_policy '{base_policy}' not found. Available: {list(totals_by_policy.keys())}")
    if other_policy not in totals_by_policy:
        raise KeyError(f"other_policy '{other_policy}' not found. Available: {list(totals_by_policy.keys())}")

    totals_base = np.asarray(totals_by_policy[base_policy], dtype=float).ravel()
    totals_other = np.asarray(totals_by_policy[other_policy], dtype=float).ravel()
    if totals_base.ndim != 1 or totals_other.ndim != 1:
        raise ValueError("Both totals arrays must be 1D after ravel()")
    if totals_base.shape[0] != totals_other.shape[0]:
        raise ValueError(
            f"Episode alignment mismatch: '{base_policy}' has {totals_base.shape[0]} episodes, "
            f"'{other_policy}' has {totals_other.shape[0]} episodes"
        )

    deltas = totals_other - totals_base

    if plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:  # pragma: no cover
            raise ImportError("Plotting requires matplotlib.") from e

        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        ax.plot(deltas, lw=1.5)
        if zero_line:
            ax.axhline(0.0, color="k", lw=1, alpha=0.5)
        ax.set_title(title or f"CRN delta totals: {other_policy} − {base_policy}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Δ total cost")
        if return_fig:
            return deltas, fig, ax

    return deltas


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """Approximate inverse CDF of standard normal (Acklam approximation)."""

    p = np.asarray(p, dtype=float)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("p must be in (0, 1)")

    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    d = np.array([7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00])

    plow = 0.02425
    phigh = 1 - plow

    x = np.empty_like(p)

    mask_l = p < plow
    if np.any(mask_l):
        q = np.sqrt(-2 * np.log(p[mask_l]))
        x[mask_l] = (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
        x[mask_l] = -x[mask_l]

    mask_c = (p >= plow) & (p <= phigh)
    if np.any(mask_c):
        q = p[mask_c] - 0.5
        r = q * q
        x[mask_c] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )

    mask_u = p > phigh
    if np.any(mask_u):
        q = np.sqrt(-2 * np.log(1 - p[mask_u]))
        x[mask_u] = (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    return x


def normality_diagnostics_for_deltas(
    deltas: Sequence[float],
    *,
    alpha: float = 0.05,
    bins: str = "auto",
    figsize: tuple[float, float] = (12, 4),
    title_prefix: str = "CRN delta normality",
) -> Dict[str, float]:
    """Normality diagnostics for paired CRN deltas (pure NumPy/Matplotlib).

    Prints: n, mean, std, skew, excess kurtosis, Jarque–Bera statistic and p-value.
    Plots: histogram with fitted Normal PDF, and a Normal Q–Q plot.

    Notes
    -----
    - Jarque–Bera is asymptotic; with large n it will flag tiny deviations.
    - For paired t-tests on the mean delta, large n often makes inference robust.
    """

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise ImportError("normality_diagnostics_for_deltas requires matplotlib.") from e

    x = np.asarray(deltas, dtype=float).ravel()
    if x.ndim != 1 or x.size == 0:
        raise ValueError("deltas must be a non-empty 1D array")
    if not np.all(np.isfinite(x)):
        raise ValueError("deltas contains NaN/Inf")

    n = int(x.size)
    mu = float(x.mean())
    sigma = float(x.std(ddof=1))

    xc = x - mu
    m2 = float(np.mean(xc**2))
    if m2 <= 0:
        raise ValueError("Variance is zero; normality tests are not meaningful")
    m3 = float(np.mean(xc**3))
    m4 = float(np.mean(xc**4))
    skew = m3 / (m2**1.5)
    kurt = m4 / (m2**2)
    ex_kurt = kurt - 3.0

    jb = n / 6.0 * (skew**2 + 0.25 * ex_kurt**2)
    jb_p = math.exp(-jb / 2.0)

    print("=== CRN delta normality diagnostics ===")
    print(f"n={n}, mean={mu:.4f}, std={sigma:.4f}")
    print(f"skew={skew:.4f}, excess_kurtosis={ex_kurt:.4f}")
    print(f"Jarque–Bera: JB={jb:.4f}, p≈{jb_p:.4g} (ChiSq df=2 approx)")
    print(f"Decision at α={alpha}: {'reject normality' if jb_p < alpha else 'fail to reject normality'}")
    print("Note: with large n, even mild deviations can be 'significant'.")

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    ax_h, ax_q = axes

    ax_h.hist(x, bins=bins, density=True, alpha=0.6, edgecolor="black")
    xs = np.linspace(float(x.min()), float(x.max()), 300)
    pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
    ax_h.plot(xs, pdf, "r-", lw=2)
    ax_h.set_title(f"{title_prefix}: histogram + fitted Normal")
    ax_h.set_xlabel("Δ total cost")
    ax_h.set_ylabel("Density")
    ax_h.grid(True, alpha=0.25)

    ps = (np.arange(1, n + 1) - 0.5) / n
    theo = mu + sigma * _norm_ppf(ps)
    samp = np.sort(x)
    ax_q.scatter(theo, samp, s=10, alpha=0.6)
    lo = min(float(theo.min()), float(samp.min()))
    hi = max(float(theo.max()), float(samp.max()))
    ax_q.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax_q.set_title(f"{title_prefix}: Normal Q–Q plot")
    ax_q.set_xlabel("Theoretical quantiles")
    ax_q.set_ylabel("Sample quantiles")
    ax_q.grid(True, alpha=0.25)

    _ = fig

    return {
        "n": float(n),
        "mean": mu,
        "std": sigma,
        "skew": float(skew),
        "excess_kurtosis": float(ex_kurt),
        "jb": float(jb),
        "jb_p": float(jb_p),
    }


def normality_diagnostics(
    data: Sequence[float],
    alpha: float = 0.05,
    verbose: bool = True,
    figsize: tuple[float, float] = (12, 4),
    bins: str = "auto",
) -> Dict[str, Any]:
    """Run a compact normality diagnostic on a 1D sample (side-by-side plots).

    - Computes basic summary statistics (n, mean, std).
    - Applies Shapiro–Wilk and D’Agostino–Pearson tests.
    - Produces a 1x2 grid: histogram + fitted Normal PDF, and a Normal Q–Q plot.
    """

    try:
        from scipy import stats
    except Exception as e:  # pragma: no cover
        raise ImportError("normality_diagnostics requires scipy (scipy.stats).") from e

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise ImportError("normality_diagnostics requires matplotlib.") from e

    arr = np.asarray(data, dtype=float).ravel()
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("data must be a non-empty 1D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("data contains NaN/Inf")

    print("=== Normality Diagnostics ===")
    print(f"Statistics: n={len(arr)}, mean={arr.mean():.2f}, std={arr.std(ddof=1):.2f}")

    shapiro_stat, shapiro_p = stats.shapiro(arr)
    print(f"Shapiro–Wilk:       stat={shapiro_stat:.4f}, p={shapiro_p:.4g}")

    k2_stat, k2_p = stats.normaltest(arr)
    print(f"D’Agostino–Pearson: stat={k2_stat:.4f}, p={k2_p:.4g}")

    mu, sigma = float(arr.mean()), float(arr.std(ddof=1))
    fig, (ax_h, ax_q) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    ax_h.hist(arr, bins=bins, density=True, alpha=0.6, edgecolor="black", label="Empirical")
    x_vals = np.linspace(float(arr.min()), float(arr.max()), 200)
    ax_h.plot(x_vals, stats.norm.pdf(x_vals, mu, sigma), "r-", lw=2, label="Fitted Normal")
    ax_h.set_xlabel("Δ total cost")
    ax_h.set_ylabel("Density")
    ax_h.set_title("Histogram + fitted Normal")
    ax_h.legend()
    ax_h.grid(True, alpha=0.3)

    stats.probplot(arr, dist="norm", plot=ax_q)
    ax_q.set_title("Normal Q–Q plot")
    ax_q.grid(True, alpha=0.3)

    if verbose:
        print(f"=== Decision at α = {alpha} ===")
        if shapiro_p > alpha and k2_p > alpha:
            print("- Both tests fail to reject normality.")
        else:
            print("- At least one test rejects strict normality.")
            print("- With n ≈ 1000, even tiny deviations from normality become significant.")
            print("- For inference on the mean delta, t-tests/CLT reasoning is often still fine.")

    return {
        "shapiro_stat": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
        "k2_stat": float(k2_stat),
        "k2_p": float(k2_p),
        "mean": float(mu),
        "std": float(sigma),
        "fig": fig,
    }


__all__ = [
    "print_totals_summary",
    "crn_delta_totals_from_totals_by_policy",
    "normality_diagnostics_for_deltas",
    "normality_diagnostics",
]
