"""
trustlens.visualization.failure_plots.
======================================

Visualizations for failure-mode analysis.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from trustlens.visualization.style import apply_style


def plot_confidence_gap(
    gap_data: dict,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot confidence distributions for correct vs. incorrect predictions.

    Two overlapping histograms reveal how well-separated the model's
    confidence is for its successes and failures. A large gap between
    the distributions is desirable.

    Parameters
    ----------
    gap_data : dict
      Output from ``confidence_gap()`` — must contain histogram arrays.
    save_path : str, optional
      If provided, saves figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    with apply_style() as theme:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

        bins = gap_data["histogram_bins"]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        width = bins[1] - bins[0]

        correct_hist = np.asarray(gap_data["correct_hist"], dtype=float)
        incorrect_hist = np.asarray(gap_data["incorrect_hist"], dtype=float)

        # Normalize to density
        correct_total = correct_hist.sum() or 1
        incorrect_total = incorrect_hist.sum() or 1
        correct_density = correct_hist / correct_total
        incorrect_density = incorrect_hist / incorrect_total

        ax.bar(
            bin_centers,
            correct_density,
            width=width * 0.8,
            color=theme.brand["green"],
            alpha=0.65,
            label=f"Correct (n={gap_data['n_correct']:,}, "
            f"mean={gap_data['correct_confidence_mean']:.3f})",
            edgecolor=theme.semantic["neutral"]["edge"],
        )
        ax.bar(
            bin_centers,
            incorrect_density,
            width=width * 0.8,
            color=theme.brand["red"],
            alpha=0.55,
            label=f"Incorrect (n={gap_data['n_incorrect']:,}, "
            f"mean={gap_data['incorrect_confidence_mean']:.3f})",
            edgecolor=theme.semantic["neutral"]["edge"],
        )

        # Confidence gap annotation
        gap = gap_data["gap"]
        ax.text(
            0.97,
            0.96,
            f"Confidence Gap = {gap:.3f}",
            transform=ax.transAxes,
            fontsize=12,
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

        ax.set_xlabel("Predicted Confidence", fontsize=12)
        ax.set_ylabel("Normalised Frequency", fontsize=12)
        ax.set_title(
            "Confidence Distribution: Correct vs. Incorrect Predictions",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.35)

        if save_path:
            fig.savefig(save_path, dpi=theme.fig_defaults["savefig_dpi"], bbox_inches="tight")

        if show:
            if "agg" not in plt.get_backend().lower():
                plt.show()
        plt.close(fig)
        return fig
