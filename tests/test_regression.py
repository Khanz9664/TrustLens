"""
tests/test_regression.py.
=========================
Unit tests for trustlens.metrics.regression.
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from trustlens.metrics import regression
from trustlens.metrics.regression import (
    error_distribution,
    error_variance_correlation,
    prediction_interval_coverage,
)


@pytest.fixture
def regression_data():
    """A small, deterministic regression fit via make_regression + LinearRegression."""
    X, y = make_regression(n_samples=200, n_features=5, noise=10.0, random_state=42)
    model = LinearRegression().fit(X, y)
    return np.asarray(y, dtype=float), model.predict(X)


class TestErrorDistribution:
    def test_perfect_predictor_zero_error(self):
        y = np.arange(10, dtype=float)
        dist = error_distribution(y, y.copy())
        assert dist["median_absolute_error"] == pytest.approx(0.0)
        assert dist["p90_absolute_error"] == pytest.approx(0.0)
        assert dist["max_error"] == pytest.approx(0.0)
        assert dist["rmse"] == pytest.approx(0.0)

    def test_medae_and_p90_known_values(self):
        # abs errors are exactly 0,1,...,9 -> MedAE=4.5, p90=8.1, max=9
        y_true = np.zeros(10, dtype=float)
        y_pred = -np.arange(10, dtype=float)
        dist = error_distribution(y_true, y_pred)
        assert dist["median_absolute_error"] == pytest.approx(4.5)
        assert dist["p90_absolute_error"] == pytest.approx(8.1)
        assert dist["max_error"] == pytest.approx(9.0)

    def test_on_make_regression_returns_finite(self, regression_data):
        y_true, y_pred = regression_data
        dist = error_distribution(y_true, y_pred)
        for key in (
            "median_absolute_error",
            "p90_absolute_error",
            "max_error",
            "mean_absolute_error",
            "rmse",
        ):
            assert isinstance(dist[key], float)
            assert np.isfinite(dist[key])
        assert dist["n_samples"] == len(y_true)
        assert dist["error_hist"].sum() == len(y_true)

    def test_shape_mismatch_raises_clear_error(self):
        with pytest.raises(ValueError, match="same shape"):
            error_distribution(np.zeros(5), np.zeros(4))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            error_distribution(np.array([]), np.array([]))

    def test_invalid_n_bins_raises(self):
        with pytest.raises(ValueError, match="n_bins"):
            error_distribution(np.zeros(5), np.zeros(5), n_bins=0)


class TestPredictionIntervalCoverage:
    def test_skips_when_no_intervals(self):
        result = prediction_interval_coverage(np.arange(10, dtype=float))
        assert result["status"] == "skipped"
        assert result["reason"] == "missing_intervals"

    def test_full_coverage_is_under_confident(self):
        y = np.arange(50, dtype=float)
        result = prediction_interval_coverage(y, y - 1e9, y + 1e9, confidence_level=0.95)
        assert result["picp"] == pytest.approx(1.0)
        assert result["verdict"] == "under-confident"

    def test_zero_width_intervals_are_over_confident(self):
        y = np.arange(50, dtype=float)
        point = y + 1.0  # point estimate never equals the truth
        result = prediction_interval_coverage(y, point, point, confidence_level=0.95)
        assert result["picp"] == pytest.approx(0.0)
        assert result["verdict"] == "over-confident"

    def test_well_calibrated_normal_intervals(self):
        rng = np.random.default_rng(7)
        y = rng.normal(0.0, 1.0, 4000)
        lo = np.full_like(y, -1.96)
        hi = np.full_like(y, 1.96)
        result = prediction_interval_coverage(y, lo, hi, confidence_level=0.95)
        assert result["picp"] == pytest.approx(0.95, abs=0.03)
        assert result["verdict"] == "well-calibrated"

    def test_lower_above_upper_raises(self):
        y = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="lower bound"):
            prediction_interval_coverage(y, y + 1.0, y - 1.0)

    def test_invalid_confidence_level_raises(self):
        y = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="confidence_level"):
            prediction_interval_coverage(y, y - 1.0, y + 1.0, confidence_level=1.5)


class TestErrorVarianceCorrelation:
    def test_skips_when_no_variance(self):
        y = np.arange(10, dtype=float)
        result = error_variance_correlation(y, y.copy())
        assert result["status"] == "skipped"
        assert result["reason"] == "missing_variance"

    def test_informative_when_variance_tracks_error(self):
        rng = np.random.default_rng(1)
        y_true = rng.normal(0.0, 1.0, 500)
        abs_err = rng.uniform(0.0, 5.0, 500)
        # construct predictions with the chosen error magnitude, variance ~ error
        y_pred = y_true + abs_err
        predicted_variance = abs_err + rng.normal(0.0, 0.1, 500)
        result = error_variance_correlation(y_true, y_pred, predicted_variance)
        assert result["spearman"] > 0.8
        assert result["verdict"] == "informative"

    def test_uninformative_when_variance_is_random(self):
        rng = np.random.default_rng(2)
        y_true = rng.normal(0.0, 1.0, 500)
        y_pred = y_true + rng.normal(0.0, 1.0, 500)
        predicted_variance = rng.uniform(0.0, 1.0, 500)  # independent of error
        result = error_variance_correlation(y_true, y_pred, predicted_variance)
        assert abs(result["spearman"]) < 0.2
        assert result["verdict"] == "uninformative"

    def test_constant_variance_returns_zero(self):
        y_true = np.arange(20, dtype=float)
        y_pred = y_true + 1.0
        predicted_variance = np.ones(20)
        result = error_variance_correlation(y_true, y_pred, predicted_variance)
        assert result["pearson"] == 0.0
        assert result["spearman"] == 0.0

    def test_shape_mismatch_raises_clear_error(self):
        with pytest.raises(ValueError, match="same shape"):
            error_variance_correlation(np.zeros(5), np.zeros(5), np.zeros(4))


def test_regression_module_exports_match_all():
    # mirror tests/test_metrics_all_exports.py's guard for the new module
    assert set(regression.__all__) == {
        "error_distribution",
        "prediction_interval_coverage",
        "error_variance_correlation",
    }
