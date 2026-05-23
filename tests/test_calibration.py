"""
tests/test_calibration.py.
==========================
Unit tests for trustlens.metrics.calibration.
"""

import numpy as np
import pytest

from trustlens.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_perfect():
    """A perfectly calibrated and perfectly predicting binary classifier."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    return y_true, y_prob


@pytest.fixture
def binary_random():
    """A random (worst-case) binary classifier."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.uniform(0, 1, size=200)
    return y_true, y_prob


@pytest.fixture
def binary_overconfident():
    """A systematically overconfident classifier."""
    n = 100
    y_true = np.zeros(n)
    y_true[:50] = 1
    y_prob = np.ones(n) * 0.9  # always says 90% confident
    return y_true, y_prob


# ---------------------------------------------------------------------------
# Brier Score tests
# ---------------------------------------------------------------------------


class TestBrierScore:
    def test_perfect_predictor_is_zero(self, binary_perfect):
        y_true, y_prob = binary_perfect
        assert brier_score(y_true, y_prob) == pytest.approx(0.0, abs=1e-10)

    def test_worst_case_predictor(self):
        """Always wrong with full confidence → BS = 1.0."""
        y_true = np.array([1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0])
        assert brier_score(y_true, y_prob) == pytest.approx(1.0)

    def test_coin_flip_approx_quarter(self, binary_random):
        """Random predictions should score ~0.25."""
        y_true, y_prob = binary_random
        bs = brier_score(y_true, y_prob)
        assert 0.15 < bs < 0.40, f"Expected ~0.25, got {bs}"

    def test_shape_mismatch_raises_clear_error(self):
        with pytest.raises(ValueError, match="Invalid input shapes for brier_score") as exc_info:
            brier_score(np.array([0, 1]), np.array([0.5, 0.5, 0.5]))

        message = str(exc_info.value)
        assert "y_true has shape (2,)" in message
        assert "y_prob has shape (3,)" in message
        assert "Both arrays must be 1D and have the same length" in message

    def test_non_binary_labels_raise(self):
        with pytest.raises(ValueError, match="binary labels"):
            brier_score(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))

    def test_returns_float(self, binary_random):
        y_true, y_prob = binary_random
        result = brier_score(y_true, y_prob)
        assert isinstance(result, float)

    def test_range_is_zero_to_one(self, binary_random):
        y_true, y_prob = binary_random
        result = brier_score(y_true, y_prob)
        assert 0.0 <= result <= 1.0

    def test_symmetric(self):
        """BS(y_true, y_prob) == BS(1-y_true, 1-y_prob) for balanced datasets."""
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.8, 0.7, 0.2, 0.3])
        bs1 = brier_score(y_true, y_prob)
        bs2 = brier_score(1 - y_true, 1 - y_prob)
        assert bs1 == pytest.approx(bs2, rel=1e-6)


# ---------------------------------------------------------------------------
# Expected Calibration Error tests
# ---------------------------------------------------------------------------


class TestExpectedCalibrationError:
    def test_perfect_calibration_is_zero(self):
        """
        A classifier where predicted = actual fraction => ECE ~ 0.
        Construct exactly: bin [0.4, 0.6) has 50% positives.
        """
        y_true = np.array([1] * 50 + [0] * 50)
        y_prob = np.array([0.5] * 100)
        ece = expected_calibration_error(y_true, y_prob, n_bins=1)
        assert ece == pytest.approx(0.0, abs=1e-6)

    def test_ece_is_nonnegative(self, binary_random):
        y_true, y_prob = binary_random
        ece = expected_calibration_error(y_true, y_prob)
        assert ece >= 0.0

    def test_ece_le_one(self, binary_overconfident):
        y_true, y_prob = binary_overconfident
        ece = expected_calibration_error(y_true, y_prob)
        assert ece <= 1.0

    def test_overconfident_has_higher_ece(self, binary_random, binary_overconfident):
        rand_ece = expected_calibration_error(*binary_random)
        over_ece = expected_calibration_error(*binary_overconfident)
        assert over_ece >= rand_ece * 0.5  # overconfident should be worse

    def test_uniform_strategy(self, binary_random):
        ece = expected_calibration_error(*binary_random, strategy="uniform")
        assert isinstance(ece, float)

    def test_quantile_strategy(self, binary_random):
        ece = expected_calibration_error(*binary_random, strategy="quantile")
        assert isinstance(ece, float)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            expected_calibration_error(np.array([0, 1]), np.array([0.3, 0.7]), strategy="invalid")


# ---------------------------------------------------------------------------
# Reliability Curve tests
# ---------------------------------------------------------------------------


class TestReliabilityCurve:
    def test_returns_three_arrays(self, binary_random):
        result = reliability_curve(*binary_random)
        assert len(result) == 3

    def test_fraction_of_positives_in_range(self, binary_random):
        frac_pos, _, _ = reliability_curve(*binary_random)
        assert np.all(frac_pos >= 0.0)
        assert np.all(frac_pos <= 1.0)

    def test_mean_predicted_in_range(self, binary_random):
        _, mean_pred, _ = reliability_curve(*binary_random)
        assert np.all(mean_pred >= 0.0)
        assert np.all(mean_pred <= 1.0)

    def test_fewer_bins_than_unique_predictions(self, binary_random):
        y_true, y_prob = binary_random
        frac_pos, mean_pred, counts = reliability_curve(y_true, y_prob, n_bins=5)
        assert len(frac_pos) <= 5
        assert len(mean_pred) <= 5
