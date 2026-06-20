"""
tests/test_calibration_visualization.py.
========================================
Tests for the calibration visualization plotting functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from trustlens.visualization.calibration_plots import plot_reliability_diagram


class TestPlotReliabilityDiagramValidation:
    """Validate input checks for plot_reliability_diagram."""

    def test_zero_n_bins_raises(self) -> None:
        fop = np.array([0.1, 0.5, 0.9])
        mpv = np.array([0.2, 0.5, 0.8])
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            plot_reliability_diagram(fop, mpv, n_bins=0, show=False)

    def test_negative_n_bins_raises(self) -> None:
        fop = np.array([0.1, 0.5, 0.9])
        mpv = np.array([0.2, 0.5, 0.8])
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            plot_reliability_diagram(fop, mpv, n_bins=-1, show=False)
