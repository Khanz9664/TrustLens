"""
tests/test_comparison_visualization.py.
========================================
Tests for the comparison visualization plotting functions.
"""

from __future__ import annotations

import pytest

from trustlens.visualization.comparison_plots import plot_radar_comparison


class TestPlotComparisonDiagramValidation:
    """Validate input checks for plot_radar_comparison."""

    def test_empty_data(self) -> None:
        empty_dict = {}
        with pytest.raises(ValueError, match="metrics_dict must not be empty"):
            plot_radar_comparison(empty_dict, save_path=None, show=False)

    def test_models_share_dimensions(self) -> None:
        metrics_dict = {
            "Random Forest": {
                "calibration": 82.4,
                "failure": 76.1,
                "bias": 91.0,
                "representation": 68.5,
            },
            "Logistic Regression": {
                "diff_calibration": 71.2,
                "diff_failure": 84.3,
                "diff_bias": 88.5,
                "diff_representation": 55.0,
            },
        }
        with pytest.raises(ValueError, match="every model should use the same dimensions"):
            plot_radar_comparison(metrics_dict, save_path=None, show=False)

    def test_successful_figure_generation(self):
        metrics_dict = {
            "Random Forest": {
                "calibration": 82.4,
                "failure": 76.1,
                "bias": 91.0,
                "representation": 68.5,
            },
            "Logistic Regression": {
                "calibration": 71.2,
                "failure": 84.3,
                "bias": 88.5,
                "representation": 55.0,
            },
        }
        fig = plot_radar_comparison(metrics_dict, save_path=None, show=False)
        assert fig is not None
