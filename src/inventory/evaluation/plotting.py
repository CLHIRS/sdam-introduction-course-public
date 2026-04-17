from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


def _require_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plotting. Install it (e.g., `pip install matplotlib`) "
            "or run in an environment that already has it."
        ) from e


ResultsTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def plot_results_grid(
    results: Sequence[ResultsTuple],
    *,
    state_index: int = 0,
    action_index: int = 0,
    exog_index: int = 0,
    figsize: Tuple[float, float] = (12, 14),
    alpha: float = 0.5,
    linewidth: float = 1.0,
    show: bool = True,
    title: str = "Rollouts overview (one line per episode)",
):
    """Plot a 5x1 grid of episode rollouts (one line per episode), stacked vertically.

    Order (top -> bottom):
      1) State
      2) Action
      3) Exogenous
      4) Stage costs
      5) Step seeds

    Args:
      results: list of (traj, costs, actions, step_seeds, Ws)
        - traj: (T+1, dS)
        - costs: (T,)
        - actions: (T, dX)
        - step_seeds: (T,)
        - Ws: (T, dW)
      state_index: which state component to plot
      action_index: which action component to plot
      exog_index: which exogenous component to plot (demand often Ws[:,0])

    Returns:
      (fig, axes)
    """

    plt = _require_matplotlib_pyplot()

    if results is None or len(results) == 0:
        raise ValueError("results is empty; run simulate_n_episodes(...) first.")

    traj0, costs0, actions0, step_seeds0, ws0 = results[0]
    traj0 = np.asarray(traj0)
    costs0 = np.asarray(costs0)
    actions0 = np.asarray(actions0)
    step_seeds0 = np.asarray(step_seeds0)
    ws0 = np.asarray(ws0)

    if traj0.ndim != 2:
        raise ValueError(f"traj must be 2D (T+1, dS). Got {traj0.shape}.")
    if costs0.ndim != 1:
        raise ValueError(f"costs must be 1D (T,). Got {costs0.shape}.")
    if actions0.ndim != 2:
        raise ValueError(f"actions must be 2D (T, dX). Got {actions0.shape}.")
    if step_seeds0.ndim != 1:
        raise ValueError(f"step_seeds must be 1D (T,). Got {step_seeds0.shape}.")
    if ws0.ndim != 2:
        raise ValueError(f"Ws must be 2D (T, dW). Got {ws0.shape}.")

    d_s = int(traj0.shape[1])
    d_x = int(actions0.shape[1])
    d_w = int(ws0.shape[1])

    if not (0 <= state_index < d_s):
        raise ValueError(f"state_index={state_index} out of range for dS={d_s}.")
    if not (0 <= action_index < d_x):
        raise ValueError(f"action_index={action_index} out of range for dX={d_x}.")
    if not (0 <= exog_index < d_w):
        raise ValueError(f"exog_index={exog_index} out of range for dW={d_w}.")

    fig, axes = plt.subplots(5, 1, figsize=figsize, constrained_layout=True, squeeze=False)
    axes = axes.reshape(-1)
    ax_s, ax_x, ax_w, ax_c, ax_seed = axes

    for (traj, costs, actions, step_seeds, ws) in results:
        traj = np.asarray(traj)
        costs = np.asarray(costs)
        actions = np.asarray(actions)
        step_seeds = np.asarray(step_seeds)
        ws = np.asarray(ws)

        t_ep = min(
            int(costs.shape[0]),
            int(step_seeds.shape[0]),
            int(actions.shape[0]),
            int(ws.shape[0]),
            int(traj.shape[0]) - 1,
        )
        if t_ep <= 0:
            continue

        ax_s.plot(np.arange(t_ep + 1), traj[: t_ep + 1, state_index], alpha=alpha, linewidth=linewidth)
        ax_x.plot(np.arange(t_ep), actions[:t_ep, action_index], alpha=alpha, linewidth=linewidth)
        ax_w.plot(np.arange(t_ep), ws[:t_ep, exog_index], alpha=alpha, linewidth=linewidth)
        ax_c.plot(np.arange(t_ep), costs[:t_ep], alpha=alpha, linewidth=linewidth)
        ax_seed.plot(np.arange(t_ep), step_seeds[:t_ep], alpha=alpha, linewidth=linewidth)

    ax_s.set_title(f"State S[:, {state_index}] (dS={d_s})")
    ax_s.set_xlabel("t")
    ax_s.set_ylabel("state")
    ax_s.grid(True, alpha=0.3)

    ax_x.set_title(f"Action X[:, {action_index}] (dX={d_x})")
    ax_x.set_xlabel("t")
    ax_x.set_ylabel("action")
    ax_x.grid(True, alpha=0.3)

    ax_w.set_title(f"Exogenous W[:, {exog_index}] (dW={d_w})")
    ax_w.set_xlabel("t")
    ax_w.set_ylabel("exog")
    ax_w.grid(True, alpha=0.3)

    ax_c.set_title("Stage costs C_t")
    ax_c.set_xlabel("t")
    ax_c.set_ylabel("cost")
    ax_c.grid(True, alpha=0.3)

    ax_seed.set_title("Step seeds (CRN object)")
    ax_seed.set_xlabel("t")
    ax_seed.set_ylabel("seed")
    ax_seed.grid(True, alpha=0.3)

    fig.suptitle(title)
    if show:
        plt.show()
    return fig, axes


def plot_multiple_results_grid(
    results_by_policy: Union[Mapping[str, Sequence[ResultsTuple]], Iterable[Tuple[str, Sequence[ResultsTuple]]]],
    *,
    state_index: int = 0,
    action_index: int = 0,
    exog_index: int = 0,
    figsize: Tuple[float, float] = (12, 14),
    alpha: float = 0.25,
    linewidth: float = 1.0,
    show: bool = True,
    title: str = "Rollouts comparison (color = policy; one line per episode)",
):
    """Plot the same 5x1 grid as `plot_results_grid`, but for multiple policies.

    Each policy gets one color; each episode is a semi-transparent line.

    Args:
      results_by_policy: mapping or iterable of (policy_name, results)

    Returns:
      (fig, axes)
    """

    plt = _require_matplotlib_pyplot()

    if results_by_policy is None:
        raise ValueError("results_by_policy is empty.")

    if hasattr(results_by_policy, "items"):
        items = list(results_by_policy.items())  # type: ignore[attr-defined]
    else:
        items = list(results_by_policy)

    if len(items) == 0:
        raise ValueError("results_by_policy is empty.")

    first_results: Optional[Sequence[ResultsTuple]] = None
    for _, res in items:
        if res is not None and len(res) > 0:
            first_results = res
            break
    if first_results is None:
        raise ValueError("All results lists are empty.")

    traj0, _costs0, actions0, _step_seeds0, ws0 = first_results[0]
    traj0 = np.asarray(traj0)
    actions0 = np.asarray(actions0)
    ws0 = np.asarray(ws0)

    if traj0.ndim != 2:
        raise ValueError(f"traj must be 2D (T+1, dS). Got {traj0.shape}.")
    if actions0.ndim != 2:
        raise ValueError(f"actions must be 2D (T, dX). Got {actions0.shape}.")
    if ws0.ndim != 2:
        raise ValueError(f"Ws must be 2D (T, dW). Got {ws0.shape}.")

    d_s = int(traj0.shape[1])
    d_x = int(actions0.shape[1])
    d_w = int(ws0.shape[1])

    if not (0 <= state_index < d_s):
        raise ValueError(f"state_index={state_index} out of range for dS={d_s}.")
    if not (0 <= action_index < d_x):
        raise ValueError(f"action_index={action_index} out of range for dX={d_x}.")
    if not (0 <= exog_index < d_w):
        raise ValueError(f"exog_index={exog_index} out of range for dW={d_w}.")

    fig, axes = plt.subplots(5, 1, figsize=figsize, constrained_layout=True, squeeze=False)
    axes = axes.reshape(-1)
    ax_s, ax_x, ax_w, ax_c, ax_seed = axes

    cmap = plt.get_cmap("tab10")
    handles = []

    for k, (policy_name, results) in enumerate(items):
        if results is None or len(results) == 0:
            continue
        color = cmap(k % 10)

        # Proxy handle so legend is clean (one entry per policy)
        (h,) = ax_s.plot([], [], color=color, linewidth=2.0, label=str(policy_name))
        handles.append(h)

        for (traj, costs, actions, step_seeds, ws) in results:
            traj = np.asarray(traj)
            costs = np.asarray(costs)
            actions = np.asarray(actions)
            step_seeds = np.asarray(step_seeds)
            ws = np.asarray(ws)

            t_ep = min(
                int(costs.shape[0]),
                int(step_seeds.shape[0]),
                int(actions.shape[0]),
                int(ws.shape[0]),
                int(traj.shape[0]) - 1,
            )
            if t_ep <= 0:
                continue

            ax_s.plot(
                np.arange(t_ep + 1),
                traj[: t_ep + 1, state_index],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )
            ax_x.plot(np.arange(t_ep), actions[:t_ep, action_index], color=color, alpha=alpha, linewidth=linewidth)
            ax_w.plot(np.arange(t_ep), ws[:t_ep, exog_index], color=color, alpha=alpha, linewidth=linewidth)
            ax_c.plot(np.arange(t_ep), costs[:t_ep], color=color, alpha=alpha, linewidth=linewidth)
            ax_seed.plot(np.arange(t_ep), step_seeds[:t_ep], color=color, alpha=alpha, linewidth=linewidth)

    ax_s.set_title(f"State S[:, {state_index}] (dS={d_s})")
    ax_s.set_xlabel("t")
    ax_s.set_ylabel("state")
    ax_s.grid(True, alpha=0.3)

    ax_x.set_title(f"Action X[:, {action_index}] (dX={d_x})")
    ax_x.set_xlabel("t")
    ax_x.set_ylabel("action")
    ax_x.grid(True, alpha=0.3)

    ax_w.set_title(f"Exogenous W[:, {exog_index}] (dW={d_w})")
    ax_w.set_xlabel("t")
    ax_w.set_ylabel("exog")
    ax_w.grid(True, alpha=0.3)

    ax_c.set_title("Stage costs C_t")
    ax_c.set_xlabel("t")
    ax_c.set_ylabel("cost")
    ax_c.grid(True, alpha=0.3)

    ax_seed.set_title("Step seeds (CRN object)")
    ax_seed.set_xlabel("t")
    ax_seed.set_ylabel("seed")
    ax_seed.grid(True, alpha=0.3)

    if handles:
        ax_s.legend(handles=handles)

    fig.suptitle(title)
    if show:
        plt.show()
    return fig, axes


def show_forecast_paths(
    f_const,
    f_ml,
    S0,
    times: Sequence[int],
    H: int,
    *,
    show: bool = True,
    verbose: bool = True,
    title: str = "Forecast mean paths μ(t+1:t+H): Constant vs ML",
):
    """Plot and (optionally) print forecast mean paths from two forecasters.

    Canonical notebook source: Lecture 12_b.

    Requirements:
    - `f_const.forecast_mean_path(S0, t, H)` returns shape (H,)
    - `f_ml.forecast_mean_path(S0, t, H)` returns shape (H,)

    Args:
      f_const: baseline/constant forecaster-like object
      f_ml: ML forecaster-like object
      S0: state used for forecasting context
      times: iterable of times t at which to compute paths
      H: horizon length
      show: whether to call plt.show()
      verbose: whether to print the numeric paths

    Returns:
      (fig, ax)
    """

    plt = _require_matplotlib_pyplot()

    times = list(times)
    H = int(H)

    fig, ax = plt.subplots(constrained_layout=True)

    for t in times:
        t = int(t)
        mu_c = np.asarray(f_const.forecast_mean_path(S0, t, H), dtype=float).reshape(-1)
        mu_m = np.asarray(f_ml.forecast_mean_path(S0, t, H), dtype=float).reshape(-1)

        if mu_c.shape[0] != H or mu_m.shape[0] != H:
            raise ValueError(f"forecast_mean_path must return shape (H,). Got {mu_c.shape} and {mu_m.shape} with H={H}.")

        if verbose:
            print(f"\n--- t={t} ---")
            print("Const forecaster mu:", np.round(mu_c, 1))
            print("ML forecaster    mu:", np.round(mu_m, 1))

        xs = np.arange(t + 1, t + H + 1)
        ax.plot(xs, mu_c, marker="o", linestyle="--", label=f"Const @t={t}")
        ax.plot(xs, mu_m, marker="x", linestyle="-", label=f"ML @t={t}")

    ax.set_title(title)
    ax.set_xlabel("future time index")
    ax.set_ylabel("forecast mean demand")
    ax.legend()

    if show:
        plt.show()
    return fig, ax


def plot_totals_hist_and_box(
    totals_by_policy: Mapping[str, np.ndarray],
    bins: Union[int, Sequence[float]] = 30,
    figsize: Tuple[float, float] = (12, 4),
    show: bool = True,
):
    """Plot a 1x2 grid: histograms (left) and boxplot (right).

    Parameters
    ----------
    totals_by_policy : dict[str, array_like]
        Mapping policy name -> 1D array of episode total costs.
    bins : int | sequence, default 30
        Binning passed to matplotlib hist().
    figsize : tuple, default (12, 4)
        Figure size in inches.

    Returns
    -------
    fig, axes : matplotlib Figure and AxesArray
    """

    plt = _require_matplotlib_pyplot()

    if not isinstance(totals_by_policy, Mapping) or len(totals_by_policy) == 0:
        raise ValueError("totals_by_policy must be a non-empty mapping name -> totals array")

    names = list(totals_by_policy.keys())
    totals_list = []
    for name in names:
        arr = np.asarray(totals_by_policy[name], dtype=float).ravel()
        if arr.ndim != 1:
            raise ValueError(f"Totals for policy '{name}' must be 1D after ravel(); got shape {arr.shape}")
        if arr.size == 0:
            raise ValueError(f"Totals for policy '{name}' is empty")
        totals_list.append(arr)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    ax_hist, ax_box = axes

    for name, arr in zip(names, totals_list):
        ax_hist.hist(arr, bins=bins, alpha=0.45, label=name)
    ax_hist.set_title("Total cost histograms")
    ax_hist.set_xlabel("Total cost")
    ax_hist.set_ylabel("Count")
    ax_hist.legend()

    ax_box.boxplot(totals_list, tick_labels=names)
    ax_box.set_title("Total cost boxplot")
    ax_box.set_ylabel("Total cost")
    ax_box.tick_params(axis="x", rotation=20)

    if show:
        plt.show()
    return fig, axes


@dataclass(frozen=True)
class PairedDeltaPlotSummary:
    baseline: str
    higher_is_better: bool
    mean_delta_by_policy: Dict[str, float]
    win_rate_by_policy: Dict[str, float]


def plot_paired_deltas_vs_baseline(
    totals_by_policy: Mapping[str, np.ndarray],
    *,
    baseline_name: str,
    higher_is_better: bool,
    figsize: Tuple[float, float] = (12, 4),
    alpha_points: float = 0.25,
    show: bool = True,
):
    """Plot paired per-episode deltas vs baseline under strict CRN.

    This is designed to pair with `inventory.evaluation.frequentist.paired_deltas_against_baseline`.

    Delta definition:
      d = totals[policy] - totals[baseline]

    Interpretation:
      - If `higher_is_better=False` (typical cost minimization), then a *win* is d < 0.
      - If `higher_is_better=True` (e.g. reward), then a *win* is d > 0.

    Returns:
      (fig, ax, summary)
    """

    plt = _require_matplotlib_pyplot()

    if baseline_name not in totals_by_policy:
        raise KeyError(f"baseline_name '{baseline_name}' not in totals_by_policy")

    base = np.asarray(totals_by_policy[baseline_name], dtype=float).ravel()
    if base.ndim != 1 or base.size == 0:
        raise ValueError("baseline totals must be non-empty 1D")

    names = [k for k in totals_by_policy.keys() if k != baseline_name]
    if len(names) == 0:
        raise ValueError("Need at least one non-baseline policy")

    deltas = []
    mean_delta_by_policy: Dict[str, float] = {}
    win_rate_by_policy: Dict[str, float] = {}

    for name in names:
        arr = np.asarray(totals_by_policy[name], dtype=float).ravel()
        if arr.shape != base.shape:
            raise ValueError(f"Paired deltas require equal shapes; baseline {base.shape} vs {name} {arr.shape}")
        d = arr - base
        deltas.append(d)
        mean_delta_by_policy[name] = float(np.mean(d))
        if higher_is_better:
            win_rate_by_policy[name] = float(np.mean(d > 0.0))
        else:
            win_rate_by_policy[name] = float(np.mean(d < 0.0))

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    # points (strip)
    rng = np.random.default_rng(0)
    for i, (name, d) in enumerate(zip(names, deltas)):
        x0 = float(i)
        jitter = rng.normal(loc=0.0, scale=0.06, size=d.shape[0])
        ax.scatter(x0 + jitter, d, s=10, alpha=float(alpha_points))

        # mean marker
        ax.plot([x0 - 0.2, x0 + 0.2], [mean_delta_by_policy[name], mean_delta_by_policy[name]], linewidth=2.5)

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_xticks(list(range(len(names))), labels=names, rotation=20, ha="right")
    ax.set_title(f"Paired deltas vs baseline: {baseline_name} (d = policy - baseline)")
    ax.set_ylabel("Delta")
    ax.grid(True, axis="y", alpha=0.2)

    # annotate win-rates
    y_min, y_max = ax.get_ylim()
    y_text = y_max - 0.05 * (y_max - y_min)
    for i, name in enumerate(names):
        wr = win_rate_by_policy[name]
        ax.text(i, y_text, f"win={wr:.2f}", ha="center", va="top", fontsize=9)

    if show:
        plt.show()

    summary = PairedDeltaPlotSummary(
        baseline=str(baseline_name),
        higher_is_better=bool(higher_is_better),
        mean_delta_by_policy=mean_delta_by_policy,
        win_rate_by_policy=win_rate_by_policy,
    )
    return fig, ax, summary


def plot_reference_episode_rollouts(
    rollouts: Mapping[str, Mapping[str, np.ndarray]],
    *,
    demand: Optional[np.ndarray] = None,
    title_suffix: str = "(shared reference episode)",
    marker: Optional[str] = None,
    show: bool = True,
):
    """Plot a single shared episode rollout per policy (traj/actions/costs), plus optional demand.

    Expected input shape (by convention used in some lectures):
      rollouts[name] = {
        "traj": (T+1,) or (T+1, dS),
        "actions": (T,) or (T, dX),
        "demands": (T,) or (T, dW),
        "costs": (T,)
      }

    This is intentionally lightweight and tolerant: if arrays are 2D, it plots column 0.

    Returns:
      dict with the matplotlib Figure objects.
    """

    plt = _require_matplotlib_pyplot()

    names = list(rollouts.keys())
    if len(names) == 0:
        raise ValueError("rollouts is empty")

    def _as_1d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return x
        if x.ndim == 2:
            return x[:, 0]
        raise ValueError(f"expected 1D or 2D array, got shape {x.shape}")

    costs0 = _as_1d(np.asarray(rollouts[names[0]]["costs"]))
    t = np.arange(costs0.shape[0])
    tp1 = np.arange(costs0.shape[0] + 1)

    figs: Dict[str, object] = {}

    # Inventory
    fig_inv = plt.figure(constrained_layout=True)
    for name in names:
        traj = _as_1d(np.asarray(rollouts[name]["traj"]))
        plt.plot(tp1, traj, label=name, marker=marker)
    if demand is not None:
        d = _as_1d(np.asarray(demand))
        plt.plot(t + 1, d[: t.shape[0]], linestyle="--", label="Demand (W)", marker=marker)
    plt.xlabel("t")
    plt.ylabel("State (S)")
    plt.title(f"State trajectories {title_suffix}")
    plt.legend()
    figs["state"] = fig_inv

    # Actions
    fig_act = plt.figure(constrained_layout=True)
    for name in names:
        act = _as_1d(np.asarray(rollouts[name]["actions"]))
        plt.plot(t, act[: t.shape[0]], label=name, marker=marker)
    plt.xlabel("t")
    plt.ylabel("Action (X)")
    plt.title(f"Actions {title_suffix}")
    plt.legend()
    figs["actions"] = fig_act

    # Exogenous (W) if available
    # - Prefer explicit `demand=` if provided.
    # - Otherwise, if rollouts include a `ws` array, plot its first component.
    if demand is None:
        has_ws = any(("ws" in rollouts.get(name, {})) for name in names)
        if has_ws:
            fig_w = plt.figure(constrained_layout=True)
            for name in names:
                if "ws" not in rollouts.get(name, {}):
                    continue
                ws = np.asarray(rollouts[name]["ws"], dtype=float)
                w1 = _as_1d(ws)
                plt.plot(t, w1[: t.shape[0]], linestyle="--", alpha=0.8, label=str(name), marker=marker)
            plt.xlabel("t")
            plt.ylabel("Exogenous (W[:,0])")
            plt.title(f"Exogenous paths {title_suffix}")
            plt.legend()
            figs["exog"] = fig_w

    # Costs
    fig_cost = plt.figure(constrained_layout=True)
    for name in names:
        c = _as_1d(np.asarray(rollouts[name]["costs"]))
        plt.plot(t, c[: t.shape[0]], label=name, marker=marker)
    plt.xlabel("t")
    plt.ylabel("Cost (C)")
    plt.title(f"Per-step costs {title_suffix}")
    plt.legend()
    figs["costs"] = fig_cost

    # Cumulative costs
    fig_cum = plt.figure(constrained_layout=True)
    for name in names:
        c = _as_1d(np.asarray(rollouts[name]["costs"]))
        plt.plot(t, np.cumsum(c[: t.shape[0]]), label=name, marker=marker)
    plt.xlabel("t")
    plt.ylabel("Cumulative cost")
    plt.title(f"Cumulative costs {title_suffix}")
    plt.legend()
    figs["cum_costs"] = fig_cum

    if show:
        plt.show()

    return figs


def plot_reference_episode_rollouts_grid(
    rollouts: Mapping[str, Mapping[str, np.ndarray]],
    *,
    demand: Optional[np.ndarray] = None,
    title_suffix: str = "(shared reference episode)",
    figsize: Tuple[float, float] = (12, 14),
    marker: Optional[str] = None,
    show: bool = True,
):
    """Plot a single shared reference episode per policy in one 5x1 grid.

    Rows (top -> bottom):
      1) State trajectory (S)
    2) Actions (X)
    3) Exogenous path (W[:,0]) (from `demand=` or rollouts[name]['ws'])
      4) Per-step costs (C)
      5) Cumulative costs (cumsum(C))

    Expected rollouts mapping:
      rollouts[name] = {
        "traj": (T+1,) or (T+1, dS)
        "actions": (T,) or (T, dX)
        "costs": (T,)
        "ws": (T, dW)  (optional)
      }

    Notes:
      - If arrays are 2D, column 0 is plotted.
      - If no exogenous data is available (no `demand` and no `ws`), the exogenous row is left blank.

    Returns:
      (fig, axes)
    """

    plt = _require_matplotlib_pyplot()

    names = list(rollouts.keys())
    if len(names) == 0:
        raise ValueError("rollouts is empty")

    def _as_1d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return x
        if x.ndim == 2:
            return x[:, 0]
        raise ValueError(f"expected 1D or 2D array, got shape {x.shape}")

    costs0 = _as_1d(np.asarray(rollouts[names[0]]["costs"]))
    t = np.arange(costs0.shape[0])
    tp1 = np.arange(costs0.shape[0] + 1)

    fig, axes = plt.subplots(5, 1, figsize=figsize, constrained_layout=True, squeeze=False)
    axes = axes.reshape(-1)
    ax_s, ax_x, ax_w, ax_c, ax_cum = axes

    # ---- State ----
    for name in names:
        traj = _as_1d(np.asarray(rollouts[name]["traj"]))
        ax_s.plot(tp1, traj, label=str(name), marker=marker)
    ax_s.set_title(f"State trajectories {title_suffix}")
    ax_s.set_xlabel("t")
    ax_s.set_ylabel("State (S)")
    ax_s.grid(True, alpha=0.2)
    ax_s.legend()

    # ---- Actions ----
    for name in names:
        act = _as_1d(np.asarray(rollouts[name]["actions"]))
        ax_x.plot(t, act[: t.shape[0]], label=str(name), marker=marker)
    ax_x.set_title(f"Actions {title_suffix}")
    ax_x.set_xlabel("t")
    ax_x.set_ylabel("Action (X)")
    ax_x.grid(True, alpha=0.2)
    ax_x.legend()

    # ---- Exogenous ----
    if demand is not None:
        d = _as_1d(np.asarray(demand))
        ax_w.plot(t + 1, d[: t.shape[0]], linestyle="--", label="Demand (W)", marker=marker)
        ax_w.legend()
    else:
        has_ws = any(("ws" in rollouts.get(name, {})) for name in names)
        if has_ws:
            for name in names:
                if "ws" not in rollouts.get(name, {}):
                    continue
                ws = np.asarray(rollouts[name]["ws"], dtype=float)
                w1 = _as_1d(ws)
                ax_w.plot(t, w1[: t.shape[0]], linestyle="--", alpha=0.8, label=str(name), marker=marker)
            ax_w.legend()
        else:
            ax_w.text(0.5, 0.5, "(no exogenous path available)", transform=ax_w.transAxes, ha="center", va="center")

    ax_w.set_title(f"Exogenous paths {title_suffix}")
    ax_w.set_xlabel("t")
    ax_w.set_ylabel("Exogenous (W[:,0])")
    ax_w.grid(True, alpha=0.2)

    # ---- Per-step costs ----
    for name in names:
        c = _as_1d(np.asarray(rollouts[name]["costs"]))
        ax_c.plot(t, c[: t.shape[0]], label=str(name), marker=marker)
    ax_c.set_title(f"Per-step costs {title_suffix}")
    ax_c.set_xlabel("t")
    ax_c.set_ylabel("Cost (C)")
    ax_c.grid(True, alpha=0.2)
    ax_c.legend()

    # ---- Cumulative costs ----
    for name in names:
        c = _as_1d(np.asarray(rollouts[name]["costs"]))
        ax_cum.plot(t, np.cumsum(c[: t.shape[0]]), label=str(name), marker=marker)
    ax_cum.set_title(f"Cumulative costs {title_suffix}")
    ax_cum.set_xlabel("t")
    ax_cum.set_ylabel("Cumulative cost")
    ax_cum.grid(True, alpha=0.2)
    ax_cum.legend()

    if show:
        plt.show()

    return fig, axes


def plot_rollouts_overlay_grid(
    rollouts_by_policy: Mapping[str, Sequence[Mapping[str, np.ndarray]]],
    *,
    state_index: int = 0,
    action_index: int = 0,
    exog_index: int = 0,
    title_suffix: str = "(CRN-MC overlay)",
    figsize: Tuple[float, float] = (12, 14),
    alpha_episode: float = 0.15,
    linewidth_episode: float = 1.0,
    linewidth_mean: float = 2.8,
    show: bool = True,
):
    """Overlay many episodes per policy in the same 5x1 grid, plus a bold mean line.

    Expected input shape:
      rollouts_by_policy[name] = [
        {"traj": (T+1,dS), "actions": (T,dX), "costs": (T,), "ws": (T,dW)},
        ...
      ]

    Rows (top -> bottom):
      1) State trajectory (S[:, state_index])
      2) Actions (X[:, action_index])
      3) Exogenous path (W[:, exog_index])
      4) Per-step costs (C)
      5) Cumulative costs (cumsum(C))

    Notes:
      - Uses the shortest episode length per policy (common truncation).
      - One legend entry per policy.

    Returns:
      (fig, axes)
    """

    plt = _require_matplotlib_pyplot()

    if not isinstance(rollouts_by_policy, Mapping) or len(rollouts_by_policy) == 0:
        raise ValueError("rollouts_by_policy must be a non-empty mapping")

    def _as_2d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return x.reshape(-1, 1)
        if x.ndim == 2:
            return x
        raise ValueError(f"expected 1D or 2D array, got shape {x.shape}")

    fig, axes = plt.subplots(5, 1, figsize=figsize, constrained_layout=True, squeeze=False)
    axes = axes.reshape(-1)
    ax_s, ax_x, ax_w, ax_c, ax_cum = axes

    cmap = plt.get_cmap("tab10")
    handles = []

    items = list(rollouts_by_policy.items())
    for k, (policy_name, eps) in enumerate(items):
        if eps is None or len(eps) == 0:
            continue

        color = cmap(k % 10)
        (h,) = ax_s.plot([], [], color=color, linewidth=linewidth_mean, label=str(policy_name))
        handles.append(h)

        # determine common truncation length for this policy
        t_common: Optional[int] = None
        for ep in eps:
            traj = _as_2d(np.asarray(ep["traj"]))
            actions = _as_2d(np.asarray(ep["actions"]))
            ws = _as_2d(np.asarray(ep["ws"]))
            costs = np.asarray(ep["costs"], dtype=float).reshape(-1)
            t_ep = min(int(costs.shape[0]), int(actions.shape[0]), int(ws.shape[0]), int(traj.shape[0]) - 1)
            if t_ep <= 0:
                continue
            t_common = t_ep if t_common is None else min(t_common, t_ep)
        if t_common is None or t_common <= 0:
            continue

        # overlay episodes (faint)
        for ep in eps:
            traj = _as_2d(np.asarray(ep["traj"]))
            actions = _as_2d(np.asarray(ep["actions"]))
            ws = _as_2d(np.asarray(ep["ws"]))
            costs = np.asarray(ep["costs"], dtype=float).reshape(-1)

            if state_index >= traj.shape[1]:
                raise ValueError(f"state_index={state_index} out of range for dS={traj.shape[1]}")
            if action_index >= actions.shape[1]:
                raise ValueError(f"action_index={action_index} out of range for dX={actions.shape[1]}")
            if exog_index >= ws.shape[1]:
                raise ValueError(f"exog_index={exog_index} out of range for dW={ws.shape[1]}")

            t_ep = min(int(costs.shape[0]), int(actions.shape[0]), int(ws.shape[0]), int(traj.shape[0]) - 1)
            if t_ep <= 0:
                continue

            t_use = min(t_common, t_ep)

            ax_s.plot(
                np.arange(t_use + 1),
                traj[: t_use + 1, state_index],
                color=color,
                alpha=float(alpha_episode),
                linewidth=float(linewidth_episode),
            )
            ax_x.plot(
                np.arange(t_use),
                actions[:t_use, action_index],
                color=color,
                alpha=float(alpha_episode),
                linewidth=float(linewidth_episode),
            )
            ax_w.plot(
                np.arange(t_use),
                ws[:t_use, exog_index],
                color=color,
                alpha=float(alpha_episode),
                linewidth=float(linewidth_episode),
                linestyle="--",
            )
            ax_c.plot(
                np.arange(t_use),
                costs[:t_use],
                color=color,
                alpha=float(alpha_episode),
                linewidth=float(linewidth_episode),
            )
            ax_cum.plot(
                np.arange(t_use),
                np.cumsum(costs[:t_use]),
                color=color,
                alpha=float(alpha_episode),
                linewidth=float(linewidth_episode),
            )

        # mean line (bold)
        s_stack = []
        x_stack = []
        w_stack = []
        c_stack = []
        for ep in eps:
            traj = _as_2d(np.asarray(ep["traj"]))
            actions = _as_2d(np.asarray(ep["actions"]))
            ws = _as_2d(np.asarray(ep["ws"]))
            costs = np.asarray(ep["costs"], dtype=float).reshape(-1)
            t_ep = min(int(costs.shape[0]), int(actions.shape[0]), int(ws.shape[0]), int(traj.shape[0]) - 1)
            if t_ep <= 0:
                continue
            t_use = min(t_common, t_ep)
            s_stack.append(traj[: t_use + 1, state_index])
            x_stack.append(actions[:t_use, action_index])
            w_stack.append(ws[:t_use, exog_index])
            c_stack.append(costs[:t_use])

        if s_stack:
            s_mean = np.mean(np.vstack(s_stack), axis=0)
            x_mean = np.mean(np.vstack(x_stack), axis=0)
            w_mean = np.mean(np.vstack(w_stack), axis=0)
            c_mean = np.mean(np.vstack(c_stack), axis=0)

            ax_s.plot(np.arange(t_common + 1), s_mean, color=color, linewidth=float(linewidth_mean), alpha=0.95)
            ax_x.plot(np.arange(t_common), x_mean, color=color, linewidth=float(linewidth_mean), alpha=0.95)
            ax_w.plot(
                np.arange(t_common),
                w_mean,
                color=color,
                linewidth=float(linewidth_mean),
                alpha=0.95,
                linestyle="--",
            )
            ax_c.plot(np.arange(t_common), c_mean, color=color, linewidth=float(linewidth_mean), alpha=0.95)
            ax_cum.plot(
                np.arange(t_common),
                np.cumsum(c_mean),
                color=color,
                linewidth=float(linewidth_mean),
                alpha=0.95,
            )

    ax_s.set_title(f"State trajectories {title_suffix}")
    ax_s.set_xlabel("t")
    ax_s.set_ylabel("State (S)")
    ax_s.grid(True, alpha=0.2)

    ax_x.set_title(f"Actions {title_suffix}")
    ax_x.set_xlabel("t")
    ax_x.set_ylabel("Action (X)")
    ax_x.grid(True, alpha=0.2)

    ax_w.set_title(f"Exogenous paths {title_suffix}")
    ax_w.set_xlabel("t")
    ax_w.set_ylabel("Exogenous (W)")
    ax_w.grid(True, alpha=0.2)

    ax_c.set_title(f"Per-step costs {title_suffix}")
    ax_c.set_xlabel("t")
    ax_c.set_ylabel("Cost (C)")
    ax_c.grid(True, alpha=0.2)

    ax_cum.set_title(f"Cumulative costs {title_suffix}")
    ax_cum.set_xlabel("t")
    ax_cum.set_ylabel("Cumulative cost")
    ax_cum.grid(True, alpha=0.2)

    if handles:
        ax_s.legend(handles=handles, title="policy")

    if show:
        plt.show()

    return fig, axes


def plot_regime_sample_path(
    exog,
    T: int,
    episode_seed: int,
    *,
    regime0: float = 0.0,
    figsize: Tuple[float, float] = (12, 14),
    show: bool = True,
):
    """Plot an exogenous-only sample path for a regime-switching demand model.

    Assumes the exogenous model returns W_{t+1} = [D_{t+1}, R_{t+1}] and uses a
    regime stored in the state vector at index ``exog.regime_index``.

    This matches the strict-CRN convention used in `DynamicSystemMVP`: one
    deterministic RNG seed per time step, spawned from `episode_seed`.

    Returns:
      (fig, axes, demand_path, regime_path)
    """

    plt = _require_matplotlib_pyplot()

    T = int(T)
    if T <= 0:
        raise ValueError(f"T must be positive. Got T={T}.")

    if not hasattr(exog, "sample"):
        raise TypeError("exog must have a .sample(state, action, t, rng) method.")
    if not hasattr(exog, "regime_index"):
        raise TypeError("exog must define .regime_index (state index for regime).")

    regime_index = int(getattr(exog, "regime_index"))
    if regime_index < 0:
        raise ValueError(f"exog.regime_index must be >= 0. Got {regime_index}.")

    ss = np.random.SeedSequence(int(episode_seed))
    child = ss.spawn(int(T))
    step_seeds = np.array([int(c.generate_state(1)[0]) for c in child], dtype=np.int64)

    # Minimal state: only the regime component is used by the exogenous model.
    d_s_min = max(1, regime_index + 1)
    state = np.zeros(d_s_min, dtype=float)
    state[regime_index] = float(regime0)
    dummy_action = np.zeros(1, dtype=float)

    demand_path = np.empty(T, dtype=float)
    regime_path = np.empty(T + 1, dtype=float)
    regime_path[0] = float(state[regime_index])

    for t in range(T):
        rng_t = np.random.default_rng(int(step_seeds[t]))
        w_tp1 = np.asarray(exog.sample(state, dummy_action, int(t), rng_t), dtype=float).reshape(-1)
        if w_tp1.shape[0] < 2:
            raise ValueError(
                "exog.sample(...) must return at least 2 elements [demand, regime_next] "
                f"for regime plotting. Got shape {w_tp1.shape}."
            )

        demand_path[t] = float(w_tp1[0])
        state[regime_index] = float(w_tp1[1])
        regime_path[t + 1] = float(state[regime_index])

    fig, (ax_d, ax_r) = plt.subplots(2, 1, figsize=figsize, constrained_layout=True, sharex=True)

    ax_d.plot(np.arange(1, T + 1), demand_path, marker="o", linewidth=1.5)
    ax_d.set_title(f"Sampled demand path D_t (exog-only; strict CRN seed={int(episode_seed)})")
    ax_d.set_ylabel("Demand")
    ax_d.grid(True, alpha=0.2)

    ax_r.step(np.arange(0, T + 1), regime_path, where="post")
    ax_r.set_title("Regime path R_t")
    ax_r.set_xlabel("t")
    ax_r.set_ylabel("Regime")
    ax_r.set_yticks(np.unique(regime_path))
    ax_r.grid(True, alpha=0.2)

    if show:
        plt.show()

    return fig, (ax_d, ax_r), demand_path, regime_path


def plot_multi_regime_sample_path(
    exog,
    T: int,
    episode_seed: int,
    *,
    state0: np.ndarray,
    figsize: Tuple[float, float] = (12, 14),
    demand_only: bool = False,
    show: bool = True,
):
    """Plot an exogenous-only sample path for a *multi-regime* demand model.

    Intended for models like `ExogenousPoissonMultiRegime` that store regimes in the
    state vector at indices ``season_index``, ``day_index``, and ``weather_index`` and
    return W_{t+1} = [D_{t+1}, season_next, day_next, weather_next].

        If ``exog`` exposes ``lambda_for_regimes(season, day, weather)``, the default
        multi-panel plot also shows the realized effective Poisson mean used to generate
        each demand sample.

        Set ``demand_only=True`` to show only the sampled demand curve.

        Returns:
            (fig, axes, demand_path, regimes_path)

    where regimes_path has shape (T+1, 3) with columns (season, day, weather).
    """

    plt = _require_matplotlib_pyplot()

    T = int(T)
    if T <= 0:
        raise ValueError(f"T must be positive. Got T={T}.")
    state0 = np.asarray(state0, dtype=float).reshape(-1)

    for attr in ("sample", "season_index", "day_index", "weather_index"):
        if not hasattr(exog, attr):
            raise TypeError(f"exog must define .{attr} for multi-regime plotting")

    season_index = int(getattr(exog, "season_index"))
    day_index = int(getattr(exog, "day_index"))
    weather_index = int(getattr(exog, "weather_index"))
    if min(season_index, day_index, weather_index) < 0:
        raise ValueError("season/day/weather indices must be >= 0")
    if max(season_index, day_index, weather_index) >= state0.shape[0]:
        raise ValueError(
            "state0 is missing regime components. "
            "Expected indices season/day/weather to be within state0."
        )

    ss = np.random.SeedSequence(int(episode_seed))
    child = ss.spawn(int(T))
    step_seeds = np.array([int(c.generate_state(1)[0]) for c in child], dtype=np.int64)

    state = state0.copy()
    dummy_action = np.zeros(1, dtype=float)

    demand_path = np.empty(T, dtype=float)
    regimes_path = np.empty((T + 1, 3), dtype=float)
    lambda_path = np.full(T, np.nan, dtype=float)
    regimes_path[0, 0] = float(state[season_index])
    regimes_path[0, 1] = float(state[day_index])
    regimes_path[0, 2] = float(state[weather_index])

    for t in range(T):
        rng_t = np.random.default_rng(int(step_seeds[t]))
        w_tp1 = np.asarray(exog.sample(state, dummy_action, int(t), rng_t), dtype=float).reshape(-1)
        if w_tp1.shape[0] < 4:
            raise ValueError(
                "exog.sample(...) must return at least 4 elements "
                "[demand, season_next, day_next, weather_next] for multi-regime plotting. "
                f"Got shape {w_tp1.shape}."
            )

        demand_path[t] = float(w_tp1[0])
        state[season_index] = float(w_tp1[1])
        state[day_index] = float(w_tp1[2])
        state[weather_index] = float(w_tp1[3])
        if hasattr(exog, "lambda_for_regimes"):
            lambda_path[t] = float(exog.lambda_for_regimes(w_tp1[1], w_tp1[2], w_tp1[3]))
        regimes_path[t + 1, 0] = float(state[season_index])
        regimes_path[t + 1, 1] = float(state[day_index])
        regimes_path[t + 1, 2] = float(state[weather_index])

    if demand_only:
        fig, ax_d = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        ax_d.plot(np.arange(1, T + 1), demand_path, marker="o", linewidth=1.5)
        ax_d.set_title(f"Sampled demand path D_t (exog-only; strict CRN seed={int(episode_seed)})")
        ax_d.set_xlabel("t")
        ax_d.set_ylabel("Demand")
        ax_d.grid(True, alpha=0.2)
        axes = (ax_d,)
    else:
        fig, axes = plt.subplots(5, 1, figsize=figsize, constrained_layout=True, sharex=True)
        ax_d, ax_lam, ax_s, ax_day, ax_w = axes

        ax_d.plot(np.arange(1, T + 1), demand_path, marker="o", linewidth=1.5)
        ax_d.set_title(f"Sampled demand path D_t (exog-only; strict CRN seed={int(episode_seed)})")
        ax_d.set_ylabel("Demand")
        ax_d.grid(True, alpha=0.2)

        if np.any(np.isfinite(lambda_path)):
            ax_lam.plot(np.arange(1, T + 1), lambda_path, marker="o", linewidth=1.5)
            ax_lam.set_title("Realized effective Poisson mean λ_t")
        else:
            ax_lam.set_title("Realized effective Poisson mean λ_t (not available on exog)")
        ax_lam.set_ylabel("λ")
        ax_lam.grid(True, alpha=0.2)

        ts = np.arange(0, T + 1)
        ax_s.step(ts, regimes_path[:, 0], where="post", linewidth=1.5)
        ax_s.set_ylabel("Season")
        ax_s.grid(True, alpha=0.2)

        ax_day.step(ts, regimes_path[:, 1], where="post", linewidth=1.5)
        ax_day.set_ylabel("Day")
        ax_day.grid(True, alpha=0.2)

        ax_w.step(ts, regimes_path[:, 2], where="post", linewidth=1.5)
        ax_w.set_xlabel("t")
        ax_w.set_ylabel("Weather")
        ax_w.grid(True, alpha=0.2)

    if show:
        plt.show()

    return fig, axes, demand_path, regimes_path


def plot_seasonal_reference_episode_sample_path(
    exog,
    T: int,
    *,
    seed0: int = 1234,
    n_episodes: int = 200,
    figsize: Tuple[float, float] = (12, 14),
    demand_only: bool = False,
    show: bool = True,
):
    """Plot an exogenous-only seasonal demand path matching the CRN *reference episode*.

    `DynamicSystemMVP.evaluate_policies_crn_mc` generates a list of `episode_seeds` by
    drawing from `rng = default_rng(seed0)`, then uses the *first* episode's per-step
    seed stream for the reference episode rollout.

    This helper recreates that seed stream and samples the exogenous model with one
    deterministic RNG seed per time step.

        Set ``demand_only=True`` to show only the sampled demand curve.

        Returns:
            (fig, axes, demand_path, lambda_path, ref_episode_seed)
    """

    plt = _require_matplotlib_pyplot()

    T = int(T)
    if T <= 0:
        raise ValueError(f"T must be positive. Got T={T}.")

    if not hasattr(exog, "sample"):
        raise TypeError("exog must have a .sample(state, action, t, rng) method.")

    seed0 = int(seed0)
    n_episodes = int(n_episodes)
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive. Got n_episodes={n_episodes}.")

    rng = np.random.default_rng(seed0)
    episode_seeds = [int(s) for s in rng.integers(1, 2**31 - 1, size=n_episodes)]
    ref_episode_seed = int(episode_seeds[0])

    ss = np.random.SeedSequence(int(ref_episode_seed))
    child = ss.spawn(int(T))
    step_seeds = np.array([int(c.generate_state(1)[0]) for c in child], dtype=np.int64)

    # Seasonal demand does not depend on state/action, but the interface requires them.
    S_dummy = np.zeros(1, dtype=float)
    X_dummy = np.zeros(1, dtype=float)

    demand_path = np.empty(T, dtype=float)
    lambda_path = np.full(T, np.nan, dtype=float)

    has_lambda = hasattr(exog, "lambda_t")

    for t in range(T):
        rng_t = np.random.default_rng(int(step_seeds[t]))
        w_tp1 = np.asarray(exog.sample(S_dummy, X_dummy, int(t), rng_t), dtype=float).reshape(-1)
        if w_tp1.shape[0] < 1:
            raise ValueError(f"exog.sample(...) must return at least 1 element [demand]. Got {w_tp1.shape}.")
        demand_path[t] = float(w_tp1[0])
        if has_lambda:
            lambda_path[t] = float(exog.lambda_t(int(t)))

    if demand_only:
        fig, ax_d = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        ax_d.plot(np.arange(1, T + 1), demand_path, marker="o", linewidth=1.5)
        ax_d.set_title(
            f"Sampled demand path D_t (exog-only; CRN reference; seed0={seed0}, ref_episode_seed={ref_episode_seed})"
        )
        ax_d.set_xlabel("t")
        ax_d.set_ylabel("Demand")
        ax_d.grid(True, alpha=0.2)
        axes = (ax_d,)
    else:
        fig, (ax_d, ax_lam) = plt.subplots(2, 1, figsize=figsize, constrained_layout=True, sharex=True)

        ax_d.plot(np.arange(1, T + 1), demand_path, marker="o", linewidth=1.5)
        ax_d.set_title(
            f"Sampled demand path D_t (exog-only; CRN reference; seed0={seed0}, ref_episode_seed={ref_episode_seed})"
        )
        ax_d.set_ylabel("Demand")
        ax_d.grid(True, alpha=0.2)

        if has_lambda:
            ax_lam.plot(np.arange(1, T + 1), lambda_path, linewidth=2.0)
            ax_lam.set_title("Seasonal mean λ(t)")
            ax_lam.set_ylabel("λ")
        else:
            ax_lam.set_title("Seasonal mean λ(t) (not available on exog)")
            ax_lam.set_ylabel("λ")
        ax_lam.set_xlabel("t")
        ax_lam.grid(True, alpha=0.2)
        axes = (ax_d, ax_lam)

    if show:
        plt.show()

    return fig, axes, demand_path, lambda_path, ref_episode_seed


def plot_seasonal_adapter_sample_paths(
    adapter,
    *,
    n_samples: int,
    seed: int,
    t_start: int = 0,
    n_paths: int = 1,
    figsize: Tuple[float, float] = (12, 8),
    alpha_path: float = 0.18,
    linewidth_path: float = 1.0,
    linewidth_mean: float = 2.6,
    show: bool = True,
):
    """Plot one (or many) synthetic demand sample paths from a SeasonalFeatureAdapter.

    The adapter's `generate_dataset` produces a one-step dataset:
      X_i = φ(t_i)
      y_i = D_{t_i+1}

    This helper visualizes y over time for one or more seeds (overlay), and also
    plots the underlying seasonal mean λ(t) if available via `adapter.exog.lambda_t`.

    Returns:
      (fig, axes, ys)
      where ys has shape (n_paths, n_samples)
    """

    plt = _require_matplotlib_pyplot()

    n_samples = int(n_samples)
    n_paths = int(n_paths)
    t_start = int(t_start)
    seed = int(seed)

    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive. Got n_samples={n_samples}.")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive. Got n_paths={n_paths}.")

    if not hasattr(adapter, "generate_dataset"):
        raise TypeError("adapter must have a .generate_dataset(n_samples, seed, t_start=...) method.")

    ts = np.arange(t_start, t_start + n_samples, dtype=int)
    t_for_y = ts + 1  # y_i corresponds to D_{t_i+1}

    ys = np.empty((n_paths, n_samples), dtype=float)
    for i in range(n_paths):
        _X, y = adapter.generate_dataset(n_samples=n_samples, seed=seed + i, t_start=t_start)
        ys[i] = np.asarray(y, dtype=float).reshape(-1)

    y_mean = np.mean(ys, axis=0)

    fig, (ax_y, ax_lam) = plt.subplots(2, 1, figsize=figsize, constrained_layout=True, sharex=True)

    for i in range(n_paths):
        ax_y.plot(t_for_y, ys[i], alpha=float(alpha_path), linewidth=float(linewidth_path))
    ax_y.plot(t_for_y, y_mean, color="orange", linewidth=float(linewidth_mean), label="mean")
    ax_y.set_title(f"Adapter-sampled demand paths (n_paths={n_paths}, n_samples={n_samples}, seed={seed})")
    ax_y.set_ylabel("Demand")
    ax_y.grid(True, alpha=0.2)
    ax_y.legend()

    lam = None
    if hasattr(adapter, "exog") and hasattr(adapter.exog, "lambda_t"):
        lam = np.array([float(adapter.exog.lambda_t(int(t))) for t in ts], dtype=float)

    if lam is not None:
        ax_lam.plot(t_for_y, lam, linewidth=2.0)
        ax_lam.set_title("Seasonal mean λ(t)")
        ax_lam.set_ylabel("λ")
    else:
        ax_lam.set_title("Seasonal mean λ(t) (not available)")
        ax_lam.set_ylabel("λ")
    ax_lam.set_xlabel("t")
    ax_lam.grid(True, alpha=0.2)

    if show:
        plt.show()

    return fig, (ax_y, ax_lam), ys


def plot_regime_adapter_sample_paths(
    adapter,
    *,
    n_samples: int,
    seed: int,
    t_start: int = 0,
    n_paths: int = 1,
    r0: int | None = None,
    figsize: Tuple[float, float] = (12, 8),
    alpha_path: float = 0.18,
    linewidth_path: float = 1.0,
    linewidth_mean: float = 2.6,
    show_pi: bool = False,
    show: bool = True,
):
    """Plot synthetic regime-demand sample paths from a RegimeFeatureAdapter.

    The adapter's `generate_dataset` returns (X,y) where:
      - X[i] = [1, pi_i[0], ..., pi_i[K-1]] for the distribution of R_{t+1} given R_t
      - y[i] is a sampled demand D_{t+1}

    We overlay multiple y paths (different seeds) and plot a mean line.
    On the second axis we plot the implied expected demand mean:
      E[D_{t+1} | R_t] = sum_j pi_i[j] * lam[j]

        If `show_pi=True`, adds a third subplot that visualizes the mean regime
        probabilities \u03c0(t) implied by the adapter features.

        Returns:
            (fig, axes, ys, mu_paths)
            where ys and mu_paths have shape (n_paths, n_samples)
    """

    plt = _require_matplotlib_pyplot()

    n_samples = int(n_samples)
    n_paths = int(n_paths)
    t_start = int(t_start)
    seed = int(seed)

    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive. Got n_samples={n_samples}.")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive. Got n_paths={n_paths}.")

    if not hasattr(adapter, "generate_dataset"):
        raise TypeError("adapter must have a .generate_dataset(...) method.")
    if not hasattr(adapter, "exog") or not hasattr(adapter.exog, "lam_by_regime"):
        raise TypeError("adapter must expose .exog.lam_by_regime for regime mean plotting.")

    lam = np.asarray(adapter.exog.lam_by_regime, dtype=float).reshape(-1)
    K = int(lam.shape[0])

    ts = np.arange(t_start, t_start + n_samples, dtype=int)
    t_for_y = ts + 1

    ys = np.empty((n_paths, n_samples), dtype=float)
    mu_paths = np.empty((n_paths, n_samples), dtype=float)
    pi_paths = np.empty((n_paths, n_samples, K), dtype=float)

    for i in range(n_paths):
        kwargs = {"n_samples": n_samples, "seed": seed + i, "t_start": t_start}
        if r0 is not None:
            kwargs["r0"] = int(r0)

        X, y = adapter.generate_dataset(**kwargs)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape[0] != n_samples:
            raise ValueError(f"adapter.generate_dataset returned y with length {y.shape[0]} != n_samples={n_samples}.")
        if X.ndim != 2 or X.shape[0] != n_samples or X.shape[1] != 1 + K:
            raise ValueError(
                f"Expected X shape (n_samples, 1+K)=({n_samples}, {1+K}) but got {X.shape}."
            )

        pi = X[:, 1 : 1 + K]
        mu = pi @ lam

        ys[i] = y
        mu_paths[i] = mu
        pi_paths[i] = pi

    y_mean = np.mean(ys, axis=0)
    mu_mean = np.mean(mu_paths, axis=0)

    if bool(show_pi):
        fig, axes = plt.subplots(3, 1, figsize=figsize, constrained_layout=True, sharex=True)
        ax_y, ax_mu, ax_pi = axes
    else:
        fig, (ax_y, ax_mu) = plt.subplots(2, 1, figsize=figsize, constrained_layout=True, sharex=True)
        ax_pi = None

    for i in range(n_paths):
        ax_y.plot(t_for_y, ys[i], alpha=float(alpha_path), linewidth=float(linewidth_path))
    ax_y.plot(t_for_y, y_mean, color="orange", linewidth=float(linewidth_mean), label="mean")
    ax_y.set_title(f"Regime adapter sampled demand paths (n_paths={n_paths}, n_samples={n_samples}, seed={seed})")
    ax_y.set_ylabel("Demand")
    ax_y.grid(True, alpha=0.2)
    ax_y.legend()

    ax_mu.plot(t_for_y, mu_mean, linewidth=2.0)
    ax_mu.set_title(r"Implied expected demand mean $\mathbb{E}[D_{t+1} \mid R_t]$ from features")
    ax_mu.set_ylabel("Expected demand")
    ax_mu.set_xlabel("t")
    ax_mu.grid(True, alpha=0.2)

    if ax_pi is not None:
        pi_mean = np.mean(pi_paths, axis=0)  # (n_samples, K)
        for j in range(K):
            ax_pi.plot(t_for_y, pi_mean[:, j], linewidth=2.0, label=f"regime {j}")
        ax_pi.set_title(r"Mean regime probabilities $\pi(t)$ implied by features")
        ax_pi.set_ylabel("Probability")
        ax_pi.set_xlabel("t")
        ax_pi.set_ylim(-0.02, 1.02)
        ax_pi.grid(True, alpha=0.2)
        ax_pi.legend(ncols=min(K, 4))

    if show:
        plt.show()

    # Return signature preserved for backward compatibility.
    return fig, (ax_y, ax_mu), ys, mu_paths


def plot_multi_regime_adapter_sample_paths(
    adapter,
    *,
    n_samples: int,
    seed: int,
    t_start: int = 0,
    n_paths: int = 1,
    r0: tuple[int, int, int] | None = None,
    figsize: Tuple[float, float] = (12, 9),
    alpha_path: float = 0.18,
    linewidth_path: float = 1.0,
    linewidth_mean: float = 2.6,
    show_pi: bool = False,
    show: bool = True,
):
    """Plot synthetic multi-regime demand sample paths from a MultiRegimeFeatureAdapter.

    Expected adapter behavior:
    - `adapter.generate_dataset(...) -> (X, y)`
    - X columns: [1, pi_season..., pi_day..., pi_weather...]
    - adapter.exog provides lambda/coeff arrays and lam_min.

    We overlay multiple y paths (different seeds) and plot a mean line.
    On the second axis we plot an implied mean demand (approx.):
      E[λ_eff] = E[base_season] * (1 + E[coeff_day] + E[coeff_weather])
    computed from the regime probability features.

    If `show_pi=True`, adds three additional subplots for the mean regime
    probabilities of season/day/weather.

    Returns:
      (fig, axes, ys, mu_paths)
      where ys and mu_paths have shape (n_paths, n_samples).
    """

    plt = _require_matplotlib_pyplot()

    n_samples = int(n_samples)
    n_paths = int(n_paths)
    t_start = int(t_start)
    seed = int(seed)

    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive. Got n_samples={n_samples}.")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive. Got n_paths={n_paths}.")
    if not hasattr(adapter, "generate_dataset"):
        raise TypeError("adapter must have a .generate_dataset(...) method.")
    if not hasattr(adapter, "exog"):
        raise TypeError("adapter must expose .exog for mean plotting.")

    exog = adapter.exog
    base = np.asarray(getattr(exog, "lambda_base_season"), dtype=float).reshape(-1)
    coeff_day = np.asarray(getattr(exog, "lambda_coeff_day"), dtype=float).reshape(-1)
    coeff_weather = np.asarray(getattr(exog, "lambda_coeff_weather"), dtype=float).reshape(-1)
    lam_min = float(getattr(exog, "lam_min", 1.0))

    Ks = int(base.shape[0])
    Kd = int(coeff_day.shape[0])
    Kw = int(coeff_weather.shape[0])
    d_expected = 1 + Ks + Kd + Kw

    ts = np.arange(t_start, t_start + n_samples, dtype=int)
    t_for_y = ts + 1

    ys = np.empty((n_paths, n_samples), dtype=float)
    mu_paths = np.empty((n_paths, n_samples), dtype=float)

    pi_season_paths = np.empty((n_paths, n_samples, Ks), dtype=float)
    pi_day_paths = np.empty((n_paths, n_samples, Kd), dtype=float)
    pi_weather_paths = np.empty((n_paths, n_samples, Kw), dtype=float)

    for i in range(n_paths):
        kwargs = {"n_samples": n_samples, "seed": seed + i, "t_start": t_start}
        if r0 is not None:
            s0, d0, w0 = r0
            kwargs.update({"season0": int(s0), "day0": int(d0), "weather0": int(w0)})

        X, y = adapter.generate_dataset(**kwargs)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape[0] != n_samples:
            raise ValueError(f"adapter.generate_dataset returned y with length {y.shape[0]} != n_samples={n_samples}.")
        if X.ndim != 2 or X.shape[0] != n_samples or X.shape[1] != d_expected:
            raise ValueError(f"Expected X shape (n_samples, {d_expected}) but got {X.shape}.")

        j = 1
        pi_s = X[:, j : j + Ks]
        j += Ks
        pi_d = X[:, j : j + Kd]
        j += Kd
        pi_w = X[:, j : j + Kw]

        E_base = pi_s @ base
        E_day = pi_d @ coeff_day
        E_weather = pi_w @ coeff_weather
        mu = E_base * (1.0 + E_day + E_weather)
        mu = np.maximum(mu, lam_min)

        ys[i] = y
        mu_paths[i] = mu
        pi_season_paths[i] = pi_s
        pi_day_paths[i] = pi_d
        pi_weather_paths[i] = pi_w

    y_mean = np.mean(ys, axis=0)
    mu_mean = np.mean(mu_paths, axis=0)

    if bool(show_pi):
        fig, axes = plt.subplots(5, 1, figsize=figsize, constrained_layout=True, sharex=True)
        ax_y, ax_mu, ax_ps, ax_pd, ax_pw = axes
    else:
        fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True, sharex=True)
        ax_y, ax_mu = axes
        ax_ps = ax_pd = ax_pw = None

    for i in range(n_paths):
        ax_y.plot(t_for_y, ys[i], alpha=float(alpha_path), linewidth=float(linewidth_path))
    ax_y.plot(t_for_y, y_mean, color="orange", linewidth=float(linewidth_mean), label="mean")
    ax_y.set_title(f"Multi-regime adapter sampled demand paths (n_paths={n_paths}, n_samples={n_samples}, seed={seed})")
    ax_y.set_ylabel("Demand")
    ax_y.grid(True, alpha=0.2)
    ax_y.legend()

    for i in range(n_paths):
        ax_mu.plot(t_for_y, mu_paths[i], alpha=float(alpha_path), linewidth=float(linewidth_path))
    ax_mu.plot(t_for_y, mu_mean, color="orange", linewidth=float(linewidth_mean), label="mean")
    ax_mu.set_title("Implied expected demand mean E[λ_eff] from adapter features")
    ax_mu.set_ylabel("E[demand]")
    ax_mu.set_xlabel("t")
    ax_mu.grid(True, alpha=0.2)
    ax_mu.legend()

    if bool(show_pi):
        ps_mean = np.mean(pi_season_paths, axis=0)
        pd_mean = np.mean(pi_day_paths, axis=0)
        pw_mean = np.mean(pi_weather_paths, axis=0)

        for k in range(Ks):
            ax_ps.plot(t_for_y, ps_mean[:, k], linewidth=1.6, label=f"season {k}")
        ax_ps.set_title("Mean season probabilities π_season(t)")
        ax_ps.set_ylabel("π")
        ax_ps.grid(True, alpha=0.2)
        ax_ps.legend(ncol=min(4, Ks), fontsize=9)

        for k in range(Kd):
            ax_pd.plot(t_for_y, pd_mean[:, k], linewidth=1.2, label=f"day {k}")
        ax_pd.set_title("Mean day probabilities π_day(t)")
        ax_pd.set_ylabel("π")
        ax_pd.grid(True, alpha=0.2)
        ax_pd.legend(ncol=min(4, Kd), fontsize=8)

        for k in range(Kw):
            ax_pw.plot(t_for_y, pw_mean[:, k], linewidth=1.6, label=f"weather {k}")
        ax_pw.set_title("Mean weather probabilities π_weather(t)")
        ax_pw.set_ylabel("π")
        ax_pw.set_xlabel("t")
        ax_pw.grid(True, alpha=0.2)
        ax_pw.legend(ncol=min(4, Kw), fontsize=9)

    if show:
        plt.show()

    return fig, axes, ys, mu_paths


def plot_multi_regime_adapter_sample_path(adapter, *, n_samples: int, seed: int, t_start: int = 0, r0: tuple[int, int, int] | None = None, **kwargs):
    """Convenience wrapper for a single multi-regime adapter path (n_paths=1)."""

    return plot_multi_regime_adapter_sample_paths(
        adapter,
        n_samples=int(n_samples),
        seed=int(seed),
        t_start=int(t_start),
        n_paths=1,
        r0=r0,
        **kwargs,
    )


__all__ = [
    "plot_results_grid",
    "plot_multiple_results_grid",
    "show_forecast_paths",
    "plot_totals_hist_and_box",
    "PairedDeltaPlotSummary",
    "plot_paired_deltas_vs_baseline",
    "plot_reference_episode_rollouts",
    "plot_reference_episode_rollouts_grid",
    "plot_rollouts_overlay_grid",
    "plot_regime_sample_path",
    "plot_seasonal_reference_episode_sample_path",
    "plot_seasonal_adapter_sample_paths",
    "plot_regime_adapter_sample_paths",
]
