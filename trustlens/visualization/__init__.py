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

import os
from typing import Optional

import matplotlib.pyplot as plt

from trustlens.visualization.bias_plots import plot_class_distribution
from trustlens.visualization.calibration_plots import plot_reliability_diagram
from trustlens.visualization.failure_plots import plot_confidence_gap
from trustlens.visualization.fairness import (
    _safe_name,
    plot_equalized_odds,
    plot_equalized_odds_multi,
    plot_fairness_gap,
    plot_fairness_gap_multi,
    plot_subgroup_performance,
    plot_subgroup_performance_multi,
)
from trustlens.visualization.representation_plots import (
    plot_embedding_2d,
    plot_embedding_separability,
)

__all__ = [
    "plot_reliability_diagram",
    "plot_confidence_gap",
    "plot_class_distribution",
    "plot_embedding_2d",
    "plot_embedding_separability",
    "plot_module",
    "plot_subgroup_performance",
    "plot_subgroup_performance_multi",
    "plot_equalized_odds",
    "plot_equalized_odds_multi",
    "plot_fairness_gap",
    "plot_fairness_gap_multi",
]

# ---------------------------------------------------------------------------
# Bias plot-type registry — deterministic ordering
# ---------------------------------------------------------------------------
_BIAS_PLOT_TYPES = (
    ("subgroup", plot_subgroup_performance_multi, "subgroup_performance"),
    ("equalized_odds", plot_equalized_odds_multi, "equalized_odds"),
    ("fairness_gap", plot_fairness_gap_multi, "equalized_odds"),
)


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
      Directory to save the resulting PNG file(s).
    """
    dispatch = {
        "calibration": _plot_calibration,
        "failure": _plot_failure,
        "bias": _plot_bias,
        "representation": _plot_representation,
    }

    plotter = dispatch.get(module_name)
    if plotter is None:
        return

    # All plotters called uniformly — no save_dir threading
    result = plotter(data)

    if result is None:
        return

    # Short-circuit empty dict
    if isinstance(result, dict) and not result:
        return

    # Ensure output directory exists
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, dict):
                # Nested: dict[str, dict[str, Figure]]
                for subkey, subfig in value.items():
                    if subfig is not None:
                        if save_dir:
                            path = os.path.join(
                                save_dir,
                                f"{module_name}_{key}_{_safe_name(subkey)}.png",
                            )
                            subfig.savefig(path, dpi=150, bbox_inches="tight")
                        plt.close(subfig)
            else:
                # Flat: dict[str, Figure]
                if value is not None:
                    if save_dir:
                        path = os.path.join(
                            save_dir,
                            f"{module_name}_{key}.png",
                        )
                        value.savefig(path, dpi=150, bbox_inches="tight")
                    plt.close(value)
    else:
        # Single Figure (existing behaviour)
        if save_dir:
            save_path = os.path.join(save_dir, f"{module_name}_plot.png")
            result.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(result)


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
    """Route bias data to the appropriate fairness visualizations.

    .. note::
        Internal use only. Called by ``plot_module()``.
        Returns a single ``Figure`` for class-imbalance data, or a nested
        ``dict[str, dict[str, Figure]]`` keyed by plot type then feature
        when fairness metrics are present. File saving is handled
        exclusively by ``plot_module()``.
    """
    if "class_imbalance" in data:
        return plot_class_distribution(data["class_imbalance"])

    result = {}
    for key, multi_fn, data_key in _BIAS_PLOT_TYPES:
        if data_key in data:
            figures = multi_fn(data[data_key], save_dir=None, show=False)
            if figures:
                result[key] = figures

    return result if result else None


def _plot_representation(data: dict):
    if "separability" not in data:
        return None
    return plot_embedding_separability(data["separability"])
