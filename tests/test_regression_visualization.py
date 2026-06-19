"""
tests/test_regression_visualization.py.
=======================================
Tests for the regression visualization plotting functions and their TrustReport
methods: residual analysis and error-distribution plots, the optional
prediction-interval band, regression-only guards, and a no-display smoke test.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display required for the test run

import matplotlib.pyplot as plt
import numpy as np
import pytest

from trustlens import analyze
from trustlens.report import TrustReport
from trustlens.visualization.regression_plots import (
    plot_error_distribution,
    plot_residuals,
)


@pytest.fixture
def reg_arrays():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 3))
    y_true = X @ np.array([1.5, -2.0, 0.7]) + rng.normal(scale=1.0, size=n)
    y_pred = y_true + rng.normal(scale=1.0, size=n)
    return X, y_true, y_pred


def _legend_labels(ax) -> list[str]:
    legend = ax.get_legend()
    return [t.get_text() for t in legend.get_texts()] if legend is not None else []


# --------------------------------------------------------------------------- #
# Module-level functions
# --------------------------------------------------------------------------- #
class TestPlotResiduals:
    def test_returns_figure_with_residual_axes(self, reg_arrays):
        _, y_true, y_pred = reg_arrays
        fig = plot_residuals(y_true, y_pred, show=False)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "Residual" in ax.get_ylabel()
        assert "Predicted" in ax.get_xlabel()
        # scatter cloud present
        assert len(ax.collections) >= 1
        assert "Residuals" in _legend_labels(ax)

    def test_interval_band_added_when_supplied(self, reg_arrays):
        from matplotlib.collections import PolyCollection

        _, y_true, y_pred = reg_arrays
        lo, hi = y_pred - 2.0, y_pred + 2.0
        fig = plot_residuals(y_true, y_pred, prediction_intervals=(lo, hi), show=False)
        ax = fig.axes[0]
        assert "Prediction interval" in _legend_labels(ax)
        # The band must actually be drawn in residual space spanning ~[-2, +2]
        # (lower - pred = -2, upper - pred = +2), not merely a legend entry.
        bands = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert bands, "expected a fill_between PolyCollection for the interval band"
        ys = np.concatenate([p.vertices[:, 1] for p in bands[0].get_paths()])
        assert ys.min() == pytest.approx(-2.0, abs=0.1)
        assert ys.max() == pytest.approx(2.0, abs=0.1)

    def test_malformed_intervals_raise(self, reg_arrays):
        _, y_true, y_pred = reg_arrays
        with pytest.raises(ValueError, match=r"\(lower, upper\)"):
            plot_residuals(y_true, y_pred, prediction_intervals=(y_pred,), show=False)

    def test_no_band_without_intervals(self, reg_arrays):
        _, y_true, y_pred = reg_arrays
        fig = plot_residuals(y_true, y_pred, show=False)
        assert "Prediction interval" not in _legend_labels(fig.axes[0])

    def test_singleton_column_flattened(self, reg_arrays):
        _, y_true, y_pred = reg_arrays
        fig = plot_residuals(y_true.reshape(-1, 1), y_pred.reshape(-1, 1), show=False)
        assert isinstance(fig, plt.Figure)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            plot_residuals(np.zeros(5), np.zeros(6), show=False)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            plot_residuals(np.array([]), np.array([]), show=False)

    def test_mismatched_interval_bounds_raise(self, reg_arrays):
        _, y_true, y_pred = reg_arrays
        with pytest.raises(ValueError, match="prediction_intervals"):
            plot_residuals(y_true, y_pred, prediction_intervals=(y_pred[:-1], y_pred), show=False)


class TestPlotErrorDistribution:
    def test_returns_figure_with_error_axes(self, reg_arrays):
        _, y_true, y_pred = reg_arrays
        fig = plot_error_distribution(y_true, y_pred, show=False)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "Error" in ax.get_xlabel()
        # histogram bars present
        assert len(ax.patches) >= 1
        # fitted-normal overlay + zero-error line present
        assert len(ax.lines) >= 1

    def test_normal_overlay_present_for_dispersed_errors(self, reg_arrays):
        _, y_true, y_pred = reg_arrays
        fig = plot_error_distribution(y_true, y_pred, show=False)
        labels = _legend_labels(fig.axes[0])
        assert any(lbl.startswith("Normal fit") for lbl in labels)

    def test_zero_bins_raises(self, reg_arrays):
        _, y_true, y_pred = reg_arrays
        with pytest.raises(ValueError, match="bins must be >= 1"):
            plot_error_distribution(y_true, y_pred, bins=0, show=False)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            plot_error_distribution(np.zeros(5), np.zeros(6), show=False)


# --------------------------------------------------------------------------- #
# TrustReport methods
# --------------------------------------------------------------------------- #
class TestReportMethods:
    def _reg_report(self, reg_arrays, with_intervals=False):
        X, y_true, y_pred = reg_arrays
        kwargs = {}
        if with_intervals:
            kwargs["prediction_intervals"] = (y_pred - 2.0, y_pred + 2.0)
        return analyze(
            model=None,
            X=X,
            y_true=y_true,
            y_pred=y_pred,
            task="regression",
            verbose=False,
            **kwargs,
        )

    def test_report_plot_residuals(self, reg_arrays):
        rep = self._reg_report(reg_arrays)
        assert isinstance(rep, TrustReport) and rep.task_type == "regression"
        fig = rep.plot_residuals(show=False)
        assert isinstance(fig, plt.Figure)

    def test_report_plot_residuals_band_from_intervals(self, reg_arrays):
        rep = self._reg_report(reg_arrays, with_intervals=True)
        fig = rep.plot_residuals(show=False)
        assert "Prediction interval" in _legend_labels(fig.axes[0])

    def test_report_plot_error_distribution(self, reg_arrays):
        rep = self._reg_report(reg_arrays)
        fig = rep.plot_error_distribution(show=False)
        assert isinstance(fig, plt.Figure)

    def test_classification_report_guards_residuals(self, reg_arrays):
        X, y_true, _ = reg_arrays
        y_cls = (y_true > np.median(y_true)).astype(int)
        yprob = np.column_stack([1 - y_cls, y_cls]).astype(float)
        rep = analyze(
            model=None,
            X=X,
            y_true=y_cls,
            y_pred=y_cls,
            y_prob=yprob,
            task="classification",
            verbose=False,
        )
        with pytest.raises(NotImplementedError):
            rep.plot_residuals(show=False)
        with pytest.raises(NotImplementedError):
            rep.plot_error_distribution(show=False)


def test_no_display_smoke(reg_arrays):
    """Under the Agg backend, rendering with show=True must not raise."""
    import warnings

    _, y_true, y_pred = reg_arrays
    with warnings.catch_warnings():
        # Agg is non-interactive; plt.show() warns but must not raise.
        warnings.simplefilter("ignore", UserWarning)
        fig1 = plot_residuals(y_true, y_pred, show=True)
        fig2 = plot_error_distribution(y_true, y_pred, show=True)
    assert isinstance(fig1, plt.Figure)
    assert isinstance(fig2, plt.Figure)
