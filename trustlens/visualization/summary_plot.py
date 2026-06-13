"""
trustlens.visualization.summary_plot.
======================================
The TrustLens Summary Dashboard — a single-figure overview that makes
model trustworthiness immediately readable.

Layout (2 rows x 3 columns):

 Trust Score   Reliability    Confidence Gap
 Gauge + Grade  Diagram      Histogram

 Error Rate    Class       Sub-score Radar/Bars
 Distribution   Distribution


Design principles:
- Clean white background
- Consistent brand colors (blue primary, traffic-light status)
- Every panel is self-contained and annotated
- Trust Score is the hero element - large, centred, unmissable
"""

from __future__ import annotations

from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from trustlens.visualization.style import (
    BRAND_COLORS,
    FIG_DEFAULTS,
    SEMANTIC_COLORS,
    apply_style,
)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def plot_summary_dashboard(
    trust_score,
    results: dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    model_name: str = "Model",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Render the complete TrustLens Summary Dashboard.

    Parameters
    ----------
    trust_score : TrustScoreResult
      Output from ``compute_trust_score()``.
    results : dict
      ``TrustReport.results`` dictionary.
    y_true, y_pred, y_prob : np.ndarray
      Ground-truth labels, predictions, and probabilities.
    model_name : str
      Model class name for the title.
    save_path : str, optional
      If provided, saves to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    with apply_style():
        # apply_style() already sets font, spines (top/right off), and white
        # figure/axes facecolors via the base theme. Only the grid needs to be
        # turned off here, since the base style (seaborn-v0_8-whitegrid) enables
        # it; the dashboard draws its own per-panel grids instead.
        plt.rcParams.update({"axes.grid": False})

        fig = plt.figure(figsize=(18, 10), facecolor=BRAND_COLORS["white"])

        # Layout Grid: 2 rows x 3 cols
        gs = gridspec.GridSpec(
            2,
            3,
            figure=fig,
            hspace=0.45,
            wspace=0.38,
            left=0.06,
            right=0.97,
            top=0.88,
            bottom=0.08,
        )

        ax_trust = fig.add_subplot(gs[0, 0])
        ax_calib = fig.add_subplot(gs[0, 1])
        ax_gap = fig.add_subplot(gs[0, 2])
        ax_err = fig.add_subplot(gs[1, 0])
        ax_bias = fig.add_subplot(gs[1, 1])
        ax_subs = fig.add_subplot(gs[1, 2])

        # Header Section
        fig.text(
            0.5,
            0.95,
            f"TrustLens Report - {model_name}",
            ha="center",
            va="top",
            fontsize=18,
            fontweight="bold",
            color=BRAND_COLORS["dark"],
        )
        fig.text(
            0.5,
            0.915,
            f"Trust Score {trust_score.score}/100 | Grade {trust_score.grade} | {trust_score.verdict}",
            ha="center",
            va="top",
            fontsize=12,
            color=_color_for_grade(trust_score.grade),
            fontweight="semibold",
        )

        # Panel 1: Trust Score Gauge

        _draw_trust_gauge(ax_trust, trust_score)

        # Panel 2: Reliability Diagram

        _draw_reliability(ax_calib, results)

        # Panel 3: Confidence Gap

        _draw_confidence_gap(ax_gap, results, y_true, y_pred, y_prob)

        # Panel 4: Error Distribution

        _draw_error_distribution(ax_err, y_true, y_pred, y_prob)

        # Panel 5: Class Distribution

        _draw_class_distribution(ax_bias, results)

        # Panel 6: Sub-score bars

        _draw_subscores(ax_subs, trust_score)

        if save_path:
            fig.savefig(
                save_path,
                dpi=FIG_DEFAULTS["savefig_dpi"],
                bbox_inches="tight",
                facecolor=BRAND_COLORS["white"],
            )

        return fig


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------


def _draw_trust_gauge(ax: plt.Axes, ts) -> None:
    """Semi-circular gauge showing Trust Score."""
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.5, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Trust Score", fontsize=11, fontweight="bold", color=BRAND_COLORS["dark"], pad=6)

    score = ts.score
    grade_color = _color_for_grade(ts.grade)

    # Background arc (grey)
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(
        np.cos(theta),
        np.sin(theta),
        color=BRAND_COLORS["light"],
        linewidth=18,
        solid_capstyle="round",
        zorder=1,
    )

    # Filled arc (colored by grade)
    fill_end = np.pi - (score / 100) * np.pi
    theta_fill = np.linspace(np.pi, fill_end, 200)
    ax.plot(
        np.cos(theta_fill),
        np.sin(theta_fill),
        color=grade_color,
        linewidth=18,
        solid_capstyle="round",
        zorder=2,
    )

    # Score text
    ax.text(
        0,
        0.18,
        f"{score}",
        ha="center",
        va="center",
        fontsize=36,
        fontweight="bold",
        color=grade_color,
        zorder=3,
    )
    ax.text(
        0,
        -0.08,
        "/100",
        ha="center",
        va="center",
        fontsize=13,
        color=BRAND_COLORS["gray"],
        zorder=3,
    )

    # Grade badge
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor=grade_color, edgecolor="none", alpha=0.15)
    ax.text(
        0,
        -0.32,
        f"Grade {ts.grade}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=grade_color,
        bbox=bbox_props,
        zorder=3,
    )

    # Tick marks at 0, 25, 50, 75, 100
    for val in [0, 25, 50, 75, 100]:
        a = np.pi - (val / 100) * np.pi
        ax.plot(
            [0.88 * np.cos(a), 1.0 * np.cos(a)],
            [0.88 * np.sin(a), 1.0 * np.sin(a)],
            color=BRAND_COLORS["gray"],
            lw=1.5,
            zorder=4,
        )
        ax.text(
            1.18 * np.cos(a),
            1.18 * np.sin(a),
            str(val),
            ha="center",
            va="center",
            fontsize=7.5,
            color=BRAND_COLORS["gray"],
        )


def _draw_reliability(ax: plt.Axes, results: dict) -> None:
    """Reliability (calibration) curve panel."""
    ax.set_title("Calibration", fontsize=11, fontweight="bold", color=BRAND_COLORS["dark"], pad=6)

    if "calibration" not in results or "reliability_curve" not in results["calibration"]:
        ax.text(
            0.5,
            0.5,
            "Not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=BRAND_COLORS["gray"],
            fontsize=10,
        )
        return

    frac_pos, mean_pred, _ = results["calibration"]["reliability_curve"]
    ece = results["calibration"].get("ece", None)
    bs = results["calibration"].get("brier_score", None)

    frac_pos = np.asarray(frac_pos)
    mean_pred = np.asarray(mean_pred)

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], "--", color=BRAND_COLORS["gray"], lw=1.3, label="Perfect", zorder=1)

    # Gap shading
    ax.fill_between(
        mean_pred, frac_pos, mean_pred, alpha=0.12, color=BRAND_COLORS["orange"], zorder=2
    )

    # Calibration curve
    ax.plot(
        mean_pred,
        frac_pos,
        "o-",
        color=BRAND_COLORS["blue"],
        lw=2.2,
        markersize=6,
        label="Model",
        zorder=3,
    )

    # Annotations
    ann_lines = []
    if ece is not None:
        ann_lines.append(f"ECE = {ece:.3f}")
    if bs is not None:
        ann_lines.append(f"BS = {bs:.3f}")
    if ann_lines:
        ax.text(
            0.04,
            0.96,
            "\n".join(ann_lines),
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#DDD", alpha=0.9),
            fontfamily="monospace",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Prob.", fontsize=9, color=BRAND_COLORS["gray"])
    ax.set_ylabel("Fraction of Positives", fontsize=9, color=BRAND_COLORS["gray"])
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, framealpha=0.6)
    ax.grid(True, alpha=0.25, linestyle="--")


def _draw_confidence_gap(
    ax: plt.Axes,
    results: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
) -> None:
    """Confidence distribution: correct vs. incorrect."""
    ax.set_title(
        "Confidence Gap", fontsize=11, fontweight="bold", color=BRAND_COLORS["dark"], pad=6
    )

    if y_prob is None:
        ax.text(
            0.5,
            0.5,
            "Not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=BRAND_COLORS["gray"],
        )
        return

    yp = np.asarray(y_prob)
    max_conf = yp.max(axis=1) if yp.ndim == 2 else yp
    correct_mask = np.asarray(y_true) == np.asarray(y_pred)

    correct_conf = max_conf[correct_mask]
    incorrect_conf = max_conf[~correct_mask]

    bins = np.linspace(0, 1, 21)

    with np.errstate(divide="ignore", invalid="ignore"):
        ax.hist(
            correct_conf,
            bins=bins.tolist(),
            color=BRAND_COLORS["green"],
            alpha=0.65,
            label=f"Correct (n={len(correct_conf):,})",
            edgecolor="white",
            density=True,
        )
        ax.hist(
            incorrect_conf,
            bins=bins.tolist(),
            color=BRAND_COLORS["red"],
            alpha=0.55,
            label=f"Incorrect (n={len(incorrect_conf):,})",
            edgecolor="white",
            density=True,
        )

    if "failure" in results:
        gap_data = results["failure"].get("confidence_gap", {})
        gap = gap_data.get("gap", None)
        if gap is not None:
            ax.text(
                0.97,
                0.97,
                f"Gap = {gap:.3f}",
                transform=ax.transAxes,
                fontsize=9,
                ha="right",
                va="top",
                fontweight="bold",
                color=BRAND_COLORS["blue"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#DDD"),
            )

    ax.set_xlabel("Predicted Confidence", fontsize=9, color=BRAND_COLORS["gray"])
    ax.set_ylabel("Density", fontsize=9, color=BRAND_COLORS["gray"])
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, framealpha=0.6)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linestyle="--")


def _draw_error_distribution(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
) -> None:
    """Per-class error rate bar chart."""
    _ = y_prob  # Optional for this plot, handled by y_true/y_pred
    ax.set_title(
        "Error Rate by Class", fontsize=11, fontweight="bold", color=BRAND_COLORS["dark"], pad=6
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)

    error_rates = []
    for cls in classes:
        mask = y_true == cls
        rate = float((y_pred[mask] != cls).mean()) if mask.sum() > 0 else 0.0
        error_rates.append(rate)

    colors = [
        BRAND_COLORS["red"]
        if r > 0.2
        else BRAND_COLORS["amber"]
        if r > 0.1
        else BRAND_COLORS["green"]
        for r in error_rates
    ]
    bars = ax.bar(
        [str(c) for c in classes],
        error_rates,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
    )

    for bar, rate in zip(bars, error_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Class", fontsize=9, color=BRAND_COLORS["gray"])
    ax.set_ylabel("Error Rate", fontsize=9, color=BRAND_COLORS["gray"])
    ax.set_ylim(0, min(1.0, max(error_rates) * 1.45 + 0.05))
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")

    # Legend
    patches = [
        mpatches.Patch(color=BRAND_COLORS["green"], label="< 10% error"),
        mpatches.Patch(color=BRAND_COLORS["amber"], label="10–20%"),
        mpatches.Patch(color=BRAND_COLORS["red"], label="> 20%"),
    ]
    ax.legend(handles=patches, fontsize=8, framealpha=0.6)


def _draw_class_distribution(ax: plt.Axes, results: dict) -> None:
    """Class count distribution."""
    ax.set_title(
        "Class Distribution", fontsize=11, fontweight="bold", color=BRAND_COLORS["dark"], pad=6
    )

    if "bias" not in results or "class_imbalance" not in results["bias"]:
        ax.text(
            0.5,
            0.5,
            "Not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=BRAND_COLORS["gray"],
            fontsize=10,
        )
        return

    imb = results["bias"]["class_imbalance"]
    class_counts = imb["class_counts"]
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    total = sum(counts)

    palette = [
        BRAND_COLORS["blue"],
        BRAND_COLORS["orange"],
        BRAND_COLORS["purple"],
        BRAND_COLORS["green"],
        BRAND_COLORS["amber"],
        BRAND_COLORS["red"],
    ]
    colors = [palette[i % len(palette)] for i in range(len(classes))]

    bars = ax.bar(
        [str(c) for c in classes],
        counts,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
    )

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.004,
            f"{100 * count / total:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ratio = imb.get("imbalance_ratio", 1.0)
    ax.text(
        0.97,
        0.97,
        f"Imbalance: {ratio:.2f}×",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#DDD"),
    )

    ax.set_xlabel("Class", fontsize=9, color=BRAND_COLORS["gray"])
    ax.set_ylabel("Count", fontsize=9, color=BRAND_COLORS["gray"])
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")


def _draw_subscores(ax: plt.Axes, ts) -> None:
    """Horizontal bar chart of Trust Score sub-scores."""
    ax.set_title(
        "Sub-score Breakdown", fontsize=11, fontweight="bold", color=BRAND_COLORS["dark"], pad=6
    )

    dims = list(ts.sub_scores.keys())
    scores = [ts.sub_scores[d] for d in dims]

    if not dims:
        ax.text(
            0.5,
            0.5,
            "No sub-scores available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=BRAND_COLORS["gray"],
            fontsize=10,
        )
        return

    y_pos = np.arange(len(dims))
    colors = [_color_for_score(s) for s in scores]

    bars = ax.barh(
        y_pos, scores, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8, height=0.55
    )

    for bar, score in zip(bars, scores):
        ax.text(
            min(score + 1.5, 103),
            bar.get_y() + bar.get_height() / 2,
            f"{score:.0f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color=_color_for_score(score),
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([d.capitalize() for d in dims], fontsize=10)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Score (0–100)", fontsize=9, color=BRAND_COLORS["gray"])
    ax.tick_params(axis="x", labelsize=8)
    ax.axvline(
        x=80, color=BRAND_COLORS["green"], linewidth=1, linestyle="--", alpha=0.6, label="Good (80)"
    )
    ax.axvline(
        x=60, color=BRAND_COLORS["amber"], linewidth=1, linestyle="--", alpha=0.6, label="OK (60)"
    )
    ax.axvline(
        x=40, color=BRAND_COLORS["red"], linewidth=1, linestyle="--", alpha=0.6, label="Poor (40)"
    )
    ax.legend(fontsize=7.5, framealpha=0.6, loc="lower right")
    ax.grid(True, axis="x", alpha=0.2, linestyle="--")

    # Reference background bands
    ax.axvspan(0, 40, alpha=0.04, color=BRAND_COLORS["red"])
    ax.axvspan(40, 60, alpha=0.04, color=BRAND_COLORS["amber"])
    ax.axvspan(60, 80, alpha=0.03, color=BRAND_COLORS["blue"])
    ax.axvspan(80, 100, alpha=0.04, color=BRAND_COLORS["green"])


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def _color_for_grade(grade: str) -> str:
    return SEMANTIC_COLORS["grade"].get(grade, BRAND_COLORS["gray"])


def _color_for_score(score: float) -> str:
    if score >= 80:
        return BRAND_COLORS["green"]
    if score >= 60:
        return BRAND_COLORS["blue"]
    if score >= 40:
        return BRAND_COLORS["amber"]
    return BRAND_COLORS["red"]
