"""
trustlens.visualization.
========================
Visualization sub-package for TrustLens reports.

All plotting functions follow a consistent interface:
 * Accept pre-computed metric data (never raw model/data)
 * Return matplotlib Figure objects (for integration flexibility)
 * Accept optional ``save_path`` to write PNG files
 * Default to a clean, publication-quality style

The ``plot_module()`` dispatcher routes data to the appropriate plotter.
"""

from typing import Optional

from trustlens.visualization.bias_plots import plot_class_distribution
from trustlens.visualization.calibration_plots import plot_reliability_diagram
from trustlens.visualization.failure_plots import plot_confidence_gap
from trustlens.visualization.fairness import (
    plot_equalized_odds,
    plot_equalized_odds_multi,
    plot_fairness_gap,
    plot_fairness_gap_multi,
    plot_subgroup_performance,
    plot_subgroup_performance_multi,
)
from trustlens.visualization.representation_plots import plot_embedding_separability

__all__ = [
    "plot_reliability_diagram",
    "plot_confidence_gap",
    "plot_class_distribution",
    "plot_embedding_separability",
    "plot_module",
    "plot_subgroup_performance",
    "plot_subgroup_performance_multi",
    "plot_equalized_odds",
    "plot_equalized_odds_multi",
    "plot_fairness_gap",
    "plot_fairness_gap_multi",
]


def plot_module(module_name: str, data: dict, save_dir: Optional[str] = None) -> None:
    """
    Dispatch a module's result data to the appropriate visualization function.

    Parameters
    ----------
    module_name : str
      Name of the analysis module (e.g., ``"calibration"``).
    data : dict
      Module result data from TrustReport.results[module_name].
    save_dir : str, optional
      Directory to save the resulting PNG file.
    """
    import os

    dispatch = {
        "calibration": _plot_calibration,
        "failure": _plot_failure,
        "bias": _plot_bias,
        "representation": _plot_representation,
    }

    plotter = dispatch.get(module_name)
    if plotter is None:
        return

    fig = plotter(data)
    if fig is None:
        return

    if save_dir:
        save_path = os.path.join(save_dir, f"{module_name}_plot.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    import matplotlib.pyplot as plt

    plt.close(fig)


def _plot_calibration(data: dict):
    if "reliability_curve" not in data:
        return None
    frac_pos, mean_pred, counts = data["reliability_curve"]
    return plot_reliability_diagram(
        frac_pos,
        mean_pred,
        ece=data.get("ece"),
        brier_score=data.get("brier_score"),
    )


def _plot_failure(data: dict):
    if "confidence_gap" not in data:
        return None
    return plot_confidence_gap(data["confidence_gap"])


def _plot_bias(data: dict):
    if "class_imbalance" in data:
        return plot_class_distribution(data["class_imbalance"])
    if "equalized_odds" in data:
        figures = plot_equalized_odds_multi(data["equalized_odds"])
        # Return the first figure for compatibility with plot_module's single-fig flow
        return next(iter(figures.values()), None) if figures else None
    return None


def _plot_representation(data: dict):
    if "separability" not in data:
        return None
    return plot_embedding_separability(data["separability"])
