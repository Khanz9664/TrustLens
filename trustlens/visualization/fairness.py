"""
Fairness Visualizations
=======================
Visualizations for fairness diagnostics across subgroups.

Built on top of ``equalized_odds()`` and ``subgroup_performance()`` outputs.
Inputs are expected to come from ``report.results["bias"]``.
"""

from __future__ import annotations

import os
import re

import matplotlib.pyplot as plt

from trustlens.visualization.style import apply_style, get_categorical_colors


def _safe_name(s: str) -> str:
    """Sanitize a string for safe use in filenames."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)


def _plot_multi_helper(
    data: dict,
    plot_fn,
    prefix: str,
    save_dir: str | None = None,
    show: bool = True,
    **kwargs,
) -> dict[str, plt.Figure]:
    """Internal helper — iterates over features in sorted order, delegates to
    a single-feature plot function, and collects results.

    Parameters
    ----------
    data : dict
        Mapping of ``{feature_name: feature_data}``.
    plot_fn : callable
        Single-feature plot function to delegate to.
    prefix : str
        Filename prefix used when ``save_dir`` is provided.
    save_dir : str, optional
        Directory in which per-feature PNG files are saved.
    show : bool, optional
        Whether to display each figure interactively.
    **kwargs
        Extra keyword arguments forwarded to *plot_fn*.
    """
    figures: dict[str, plt.Figure] = {}
    for feature_name in sorted(data.keys()):
        save_path = (
            os.path.join(save_dir, f"{prefix}_{_safe_name(feature_name)}.png") if save_dir else None
        )
        figures[feature_name] = plot_fn(
            data[feature_name],
            feature_name,
            save_path=save_path,
            show=show,
            **kwargs,
        )
    return figures


def plot_subgroup_performance(
    subgroup_data: dict,
    feature_name: str,
    metric: str = "accuracy",
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Bar chart of model performance broken down by subgroup.

    Parameters
    ----------
    subgroup_data : dict
        Output from ``subgroup_performance()`` for a single feature.
        Example: ``results["gender"]``.
    feature_name : str
        Name of the sensitive feature (used as axis label and title).
    metric : str, optional
        Which metric to plot. Supports ``"accuracy"`` and ``"f1"``.
        Default: ``"accuracy"``.
    save_path : str, optional
        If provided, saves figure to this path.
    show : bool, optional
        Whether to display the figure interactively. Default: ``True``.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    ``subgroup_data`` may optionally contain a ``"__summary__"`` key
    (as returned by ``subgroup_performance()``) for gap annotation.

    Examples
    --------
    >>> results = subgroup_performance(y_true, y_pred, {"gender": gender})
    >>> fig = plot_subgroup_performance(results["gender"], "gender")
    """
    with apply_style() as theme:
        groups = [g for g in subgroup_data if g != "__summary__"]
        values = [subgroup_data[g].get(metric, 0.0) for g in groups]
        colors = get_categorical_colors(len(groups))

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

        bars = ax.bar(
            groups,
            values,
            color=colors,
            edgecolor=theme.semantic["neutral"]["edge"],
            linewidth=1.2,
            alpha=0.85,
        )

        # Annotate bars with values
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        # Performance gap annotation
        if "__summary__" in subgroup_data:
            gap = subgroup_data["__summary__"].get("performance_gap", None)
            if gap is not None:
                ax.text(
                    0.97,
                    0.97,
                    f"Performance gap = {gap:.3f}",
                    transform=ax.transAxes,
                    fontsize=11,
                    ha="right",
                    va="top",
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        facecolor=theme.semantic["neutral"]["annotation_face"],
                        edgecolor=theme.semantic["neutral"]["annotation_edge"],
                        alpha=0.9,
                    ),
                    fontfamily="monospace",
                )

        ax.set_ylim(0, 1.1)
        ax.set_xlabel(feature_name.capitalize(), fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(
            f"Subgroup {metric.capitalize()} by {feature_name.capitalize()}",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.35)

        if save_path:
            fig.savefig(save_path, dpi=theme.fig_defaults["savefig_dpi"], bbox_inches="tight")
        if show:
            if "agg" not in plt.get_backend().lower():
                plt.show()
        plt.close(fig)
        return fig


def plot_equalized_odds(
    equalized_odds_data: dict,
    feature_name: str,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Grouped bar chart of TPR and FPR per subgroup.

    Displays TPR and FPR side by side for each group, making disparities
    between subgroups immediately visible.

    Parameters
    ----------
    equalized_odds_data : dict
        Output from ``equalized_odds()`` for a single feature.
        Example: ``results["gender"]``.
    feature_name : str
        Name of the sensitive feature (used as axis label and title).
    save_path : str, optional
        If provided, saves figure to this path.
    show : bool, optional
        Whether to display the figure interactively. Default: ``True``.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    ``equalized_odds_data`` must contain a ``"__summary__"`` key
    (as returned by ``equalized_odds()``). Missing summary data
    will result in annotations being silently skipped.

    Examples
    --------
    >>> results = equalized_odds(y_true, y_pred, {"gender": gender})
    >>> fig = plot_equalized_odds(results["gender"], "gender")
    """
    with apply_style() as theme:
        import numpy as np

        groups = [g for g in equalized_odds_data if g != "__summary__"]
        tpr_values = [equalized_odds_data[g]["tpr"] for g in groups]
        fpr_values = [equalized_odds_data[g]["fpr"] for g in groups]

        x = np.arange(len(groups))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

        bars_tpr = ax.bar(
            x - width / 2,
            tpr_values,
            width,
            label="TPR",
            color=theme.brand["blue"],
            edgecolor=theme.semantic["neutral"]["edge"],
            linewidth=1.2,
            alpha=0.85,
        )
        bars_fpr = ax.bar(
            x + width / 2,
            fpr_values,
            width,
            label="FPR",
            color=theme.brand["orange"],
            edgecolor=theme.semantic["neutral"]["edge"],
            linewidth=1.2,
            alpha=0.85,
        )

        # Annotate bars
        for bar, val in zip(bars_tpr, tpr_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        for bar, val in zip(bars_fpr, fpr_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # Summary annotation
        if "__summary__" in equalized_odds_data:
            summary = equalized_odds_data["__summary__"]
            tpr_gap = summary.get("tpr_gap", 0.0)
            fpr_gap = summary.get("fpr_gap", 0.0)
            ax.text(
                0.97,
                0.97,
                f"TPR gap = {tpr_gap:.3f}  |  FPR gap = {fpr_gap:.3f}",
                transform=ax.transAxes,
                fontsize=10,
                ha="right",
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor=theme.semantic["neutral"]["annotation_face"],
                    edgecolor=theme.semantic["neutral"]["annotation_edge"],
                    alpha=0.9,
                ),
                fontfamily="monospace",
            )

        ax.set_ylim(0, 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_xlabel(feature_name.capitalize(), fontsize=12)
        ax.set_ylabel("Rate", fontsize=12)
        ax.set_title(
            f"Equalized Odds by {feature_name.capitalize()}", fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.35)

        if save_path:
            fig.savefig(save_path, dpi=theme.fig_defaults["savefig_dpi"], bbox_inches="tight")
        if show:
            if "agg" not in plt.get_backend().lower():
                plt.show()
        plt.close(fig)
        return fig


def plot_fairness_gap(
    equalized_odds_data: dict,
    feature_name: str,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Bar chart showing TPR gap and FPR gap between best and worst subgroups.

    Parameters
    ----------
    equalized_odds_data : dict
        Output from ``equalized_odds()`` for a single feature.
        Example: ``results["gender"]``.
    feature_name : str
        Name of the sensitive feature (used as axis label and title).
    save_path : str, optional
        If provided, saves figure to this path.
    show : bool, optional
        Whether to display the figure interactively. Default: ``True``.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    ``equalized_odds_data`` must contain a ``"__summary__"`` key
    (as returned by ``equalized_odds()``). Missing summary data
    will result in annotations being silently skipped.

    Examples
    --------
    >>> results = equalized_odds(y_true, y_pred, {"gender": gender})
    >>> fig = plot_fairness_gap(results["gender"], "gender")
    """
    with apply_style() as theme:
        summary = equalized_odds_data.get("__summary__", {})
        tpr_gap = summary.get("tpr_gap", 0.0)
        fpr_gap = summary.get("fpr_gap", 0.0)

        labels = ["TPR Gap", "FPR Gap"]
        values = [tpr_gap, fpr_gap]
        colors = [theme.brand["blue"], theme.brand["orange"]]

        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

        bars = ax.bar(
            labels,
            values,
            color=colors,
            edgecolor=theme.semantic["neutral"]["edge"],
            linewidth=1.2,
            alpha=0.85,
        )

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        # Violation level annotations
        tpr_violation = summary.get("tpr_violation", "")
        fpr_violation = summary.get("fpr_violation", "")
        violation_colors = theme.semantic["severity"]

        for i, (violation, _label) in enumerate(zip([tpr_violation, fpr_violation], labels)):
            color = violation_colors.get(violation, violation_colors["unknown"])
            ax.text(
                i,
                -0.06,
                violation,
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color=color,
                transform=ax.get_xaxis_transform(),
            )

        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Gap", fontsize=12)
        ax.set_title(f"Fairness Gap — {feature_name.capitalize()}", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.35)

        if save_path:
            fig.savefig(save_path, dpi=theme.fig_defaults["savefig_dpi"], bbox_inches="tight")
        if show:
            if "agg" not in plt.get_backend().lower():
                plt.show()
        plt.close(fig)
        return fig


# ---------------------------------------------------------------------------
# Multi-feature wrappers
# ---------------------------------------------------------------------------


def plot_subgroup_performance_multi(
    subgroup_data: dict,
    metric: str = "accuracy",
    save_dir: str | None = None,
    show: bool = True,
) -> dict[str, plt.Figure]:
    """
    Plot subgroup performance for **each** sensitive feature.

    Iterates over all features in ``subgroup_data`` and delegates to
    :func:`plot_subgroup_performance` for each one.  No feature is silently
    dropped.  Features are processed in sorted order for deterministic output.

    Parameters
    ----------
    subgroup_data : dict
        Mapping of ``{feature_name: feature_data}`` as returned by
        ``subgroup_performance()`` when multiple sensitive features are passed.
        Example: ``results["bias"]["subgroup_performance"]``.
    metric : str, optional
        Which metric to plot. Supports ``"accuracy"`` and ``"f1"``.
        Default: ``"accuracy"``.
    save_dir : str, optional
        Directory in which per-feature PNG files are saved.
        Files are named ``subgroup_performance_<feature_name>.png``.
        If ``None``, no files are written.
    show : bool, optional
        Whether to display each figure interactively. Default: ``True``.

    Returns
    -------
    dict[str, matplotlib.figure.Figure]
        Mapping of ``{feature_name: Figure}`` for every feature processed.

    Examples
    --------
    >>> results = subgroup_performance(y_true, y_pred, sensitive_features)
    >>> figs = plot_subgroup_performance_multi(results)
    >>> fig_gender = figs["gender"]
    """
    return _plot_multi_helper(
        subgroup_data,
        plot_subgroup_performance,
        "subgroup_performance",
        save_dir=save_dir,
        show=show,
        metric=metric,
    )


def plot_equalized_odds_multi(
    equalized_odds_data: dict,
    save_dir: str | None = None,
    show: bool = True,
) -> dict[str, plt.Figure]:
    """
    Plot equalized odds for **each** sensitive feature.

    Iterates over all features in ``equalized_odds_data`` and delegates to
    :func:`plot_equalized_odds` for each one.  No feature is silently dropped.
    Features are processed in sorted order for deterministic output.

    Parameters
    ----------
    equalized_odds_data : dict
        Mapping of ``{feature_name: feature_data}`` as returned by
        ``equalized_odds()`` when multiple sensitive features are passed.
        Example: ``results["bias"]["equalized_odds"]``.
    save_dir : str, optional
        Directory in which per-feature PNG files are saved.
        Files are named ``equalized_odds_<feature_name>.png``.
        If ``None``, no files are written.
    show : bool, optional
        Whether to display each figure interactively. Default: ``True``.

    Returns
    -------
    dict[str, matplotlib.figure.Figure]
        Mapping of ``{feature_name: Figure}`` for every feature processed.

    Examples
    --------
    >>> results = equalized_odds(y_true, y_pred, sensitive_features)
    >>> figs = plot_equalized_odds_multi(results)
    >>> fig_age = figs["age"]
    """
    return _plot_multi_helper(
        equalized_odds_data,
        plot_equalized_odds,
        "equalized_odds",
        save_dir=save_dir,
        show=show,
    )


def plot_fairness_gap_multi(
    equalized_odds_data: dict,
    save_dir: str | None = None,
    show: bool = True,
) -> dict[str, plt.Figure]:
    """
    Plot fairness gap for **each** sensitive feature.

    Iterates over all features in ``equalized_odds_data`` and delegates to
    :func:`plot_fairness_gap` for each one.  No feature is silently dropped.
    Features are processed in sorted order for deterministic output.

    Parameters
    ----------
    equalized_odds_data : dict
        Mapping of ``{feature_name: feature_data}`` as returned by
        ``equalized_odds()`` when multiple sensitive features are passed.
        Example: ``results["bias"]["equalized_odds"]``.
    save_dir : str, optional
        Directory in which per-feature PNG files are saved.
        Files are named ``fairness_gap_<feature_name>.png``.
        If ``None``, no files are written.
    show : bool, optional
        Whether to display each figure interactively. Default: ``True``.

    Returns
    -------
    dict[str, matplotlib.figure.Figure]
        Mapping of ``{feature_name: Figure}`` for every feature processed.

    Examples
    --------
    >>> results = equalized_odds(y_true, y_pred, sensitive_features)
    >>> figs = plot_fairness_gap_multi(results)
    >>> fig_gender = figs["gender"]
    """
    return _plot_multi_helper(
        equalized_odds_data,
        plot_fairness_gap,
        "fairness_gap",
        save_dir=save_dir,
        show=show,
    )
