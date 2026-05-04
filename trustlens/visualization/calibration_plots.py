"""
trustlens.visualization.calibration_plots.
==========================================
Reliability (calibration) diagram and related plots.

The reliability diagram is the canonical tool for assessing probability
calibration. A perfectly calibrated model's curve hugs the diagonal.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_reliability_diagram(
    fraction_of_positives: np.ndarray,
    mean_predicted_value: np.ndarray,
    ece: float | None = None,
    brier_score: float | None = None,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Render a reliability (calibration) diagram.

    The plot shows:
    * **Calibration curve** — predicted probability vs. fraction of positives.
    * **Perfect calibration** diagonal line for reference.
    * **Gap fills** — shaded areas indicating over/under-confidence.
    * **Metric annotations** — ECE and Brier Score as a text box.

    Parameters
    ----------
    fraction_of_positives : np.ndarray
      Actual fraction of positives per bin (from ``reliability_curve()``).
    mean_predicted_value : np.ndarray
      Mean predicted probability per bin (x-axis values).
    ece : float, optional
      Expected Calibration Error to annotate.
    brier_score : float, optional
      Brier Score to annotate.
    n_bins : int
      Used only for aesthetics (y-axis secondary histogram).
    title : str
      Plot title.
    save_path : str, optional
      If provided, saves the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Style
    plt.style.use("seaborn-v0_8-whitegrid")
    BLUE = "#4B8BF5"
    ORANGE = "#F5784B"
    GRAY = "#AAAAAA"

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    # Calibration Curve
    # Perfect calibration reference
    ax1.plot([0, 1], [0, 1], linestyle="--", color=GRAY, lw=1.5, label="Perfect calibration")

    # Gap fill (over/under-confident regions)
    fop = np.asarray(fraction_of_positives)
    mpv = np.asarray(mean_predicted_value)
    ax1.fill_between(mpv, fop, mpv, alpha=0.15, color=ORANGE, label="Calibration gap")

    # Model calibration curve
    ax1.plot(mpv, fop, marker="o", color=BLUE, lw=2.5, markersize=8, label="Model")

    # Metric annotations
    annotation_lines = []
    if ece is not None:
        annotation_lines.append(f"ECE = {ece:.4f}")
    if brier_score is not None:
        annotation_lines.append(f"BS  = {brier_score:.4f}")

    if annotation_lines:
        annotation_text = "\n".join(annotation_lines)
        ax1.text(
            0.04,
            0.95,
            annotation_text,
            transform=ax1.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
            fontfamily="monospace",
        )

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax1.set_ylabel("Fraction of Positives", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.4)

    # Confidence Histogram (lower panel)
    ax2.bar(
        mpv,
        mpv - fop,
        width=1.0 / (n_bins * 1.5),
        color=[BLUE if v >= 0 else ORANGE for v in (mpv - fop)],
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
    )
    ax2.axhline(0, color=GRAY, lw=1)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax2.set_ylabel("Avg. Confidence Gap\n(Pred − True)", fontsize=10)
    ax2.set_title("Confidence Gap per Bin", fontsize=11)
    ax2.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        if "agg" not in plt.get_backend().lower():
            plt.show()

    plt.close(fig)
    return fig
