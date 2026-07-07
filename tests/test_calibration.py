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
    maximum_calibration_error,
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
        assert "y_true has shape" in message
        assert "y_prob has shape" in message
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
# Maximum Calibration Error tests (issue #134)
# ---------------------------------------------------------------------------


class TestMaximumCalibrationError:
    def test_mce_perfect_calibration_is_zero(self, binary_perfect):
        y_true, y_prob = binary_perfect
        mce = maximum_calibration_error(y_true, y_prob)
        assert mce == pytest.approx(0.0, abs=1e-10)

    def test_mce_geq_ece(self, binary_random):
        """MCE >= ECE always: the max of the per-bin gaps is at least their
        weighted average (weights sum to <= 1)."""
        y_true, y_prob = binary_random
        for strategy in ("uniform", "quantile"):
            ece = expected_calibration_error(y_true, y_prob, strategy=strategy)
            mce = maximum_calibration_error(y_true, y_prob, strategy=strategy)
            assert mce >= ece - 1e-12

    def test_mce_worst_bin_dominates(self):
        """One catastrophic bin: ECE averages it away, MCE reports it.

        90 samples perfectly calibrated at p=0.5 (gap 0) + 10 samples at
        p=0.95 that are ALL wrong (gap 0.95). ECE = 0.1 * 0.95 = 0.095 looks
        acceptable; MCE = 0.95 exposes the catastrophic confidence region.
        """
        y_true = np.array([1] * 45 + [0] * 45 + [0] * 10)
        y_prob = np.array([0.5] * 90 + [0.95] * 10)
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        mce = maximum_calibration_error(y_true, y_prob, n_bins=10)
        assert mce == pytest.approx(0.95, abs=1e-10)
        assert ece == pytest.approx(0.095, abs=1e-10)
        assert mce > 5 * ece

    def test_mce_empty_bins_handled(self):
        """Skewed distributions leaving most bins empty must not raise."""
        y_true = np.array([1, 0, 1, 1, 0, 1, 1, 1])
        y_prob = np.array([0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98])
        mce = maximum_calibration_error(y_true, y_prob, n_bins=10)
        assert isinstance(mce, float)
        assert 0.0 <= mce <= 1.0

    def test_mce_in_pipeline_results(self):
        """End-to-end: analyze() exposes results['calibration']['mce']."""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        from trustlens import analyze

        X, y = make_classification(n_samples=300, n_features=10, n_classes=2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)
        report = analyze(clf, X_val, y_val, y_prob=clf.predict_proba(X_val), verbose=False)

        mce = report.results["calibration"]["mce"]
        assert isinstance(mce, float)
        assert 0.0 <= mce <= 1.0
        assert mce >= report.results["calibration"]["ece"] - 1e-12

    def test_mce_in_pipeline_results_multiclass(self):
        """Pipeline wiring on the multiclass (top-label) path exposes MCE too.

        The metric core is shared with the binary path; this covers the
        separate multiclass branch of the pipeline (n_classes >= 3) so the
        wiring, not just the metric, is exercised end to end.
        """
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        from trustlens import analyze

        X, y = make_classification(
            n_samples=450,
            n_features=12,
            n_informative=6,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42,
        )
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)
        report = analyze(clf, X_val, y_val, y_prob=clf.predict_proba(X_val), verbose=False)

        mce = report.results["calibration"]["mce"]
        assert isinstance(mce, float)
        assert 0.0 <= mce <= 1.0
        assert mce >= report.results["calibration"]["ece"] - 1e-12

    def test_mce_quantile_bin_collapse(self):
        """Quantile binning that collapses to fewer effective bins stays stable.

        Constant probabilities collapse every quantile edge into one; heavy
        duplication collapses ten requested bins into one wide bin. Both must
        return a finite, correct value rather than raising.
        """
        # Fully constant: perfectly calibrated at p=0.5 -> MCE 0.0
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        assert maximum_calibration_error(y_true, y_prob, strategy="quantile") == pytest.approx(0.0)

        # Two duplicated values collapse 10 requested bins into one wide bin:
        # acc = 0.5, conf = 0.5 -> gap 0; shifting the labels moves the gap.
        y_true = np.array([1] * 50 + [0] * 50)
        y_prob = np.array([0.1] * 50 + [0.9] * 50)
        mce = maximum_calibration_error(y_true, y_prob, n_bins=10, strategy="quantile")
        assert isinstance(mce, float)
        assert 0.0 <= mce <= 1.0

    def test_mce_collapsed_bins_miscalibrated_constant_predictor(self):
        """Fully collapsed quantile edges must not hide a miscalibrated constant.

        Every prediction is p=0.5 but every label is 1 (accuracy 1.0), so the
        true gap is 0.5. When ``np.unique`` collapses all quantile edges to a
        single value the bin loop iterates zero times; the single-active-bin
        fallback must still report 0.5 for BOTH metrics rather than a spurious
        0.0 (regression guard for issue #163 review).
        """
        y_true = np.ones(100)
        y_prob = np.full(100, 0.5)
        mce = maximum_calibration_error(y_true, y_prob, n_bins=10, strategy="quantile")
        ece = expected_calibration_error(y_true, y_prob, n_bins=10, strategy="quantile")
        assert mce == pytest.approx(0.5, abs=1e-12)
        assert ece == pytest.approx(0.5, abs=1e-12)

    def test_binned_core_rejects_non_1d_and_bad_n_bins(self):
        """The shared core validates its own inputs so ECE/MCE can't diverge.

        A same-shaped 2D array previously slipped through and corrupted the
        per-bin weights (``len(y_true)`` as denominator); ``n_bins < 1`` fell
        into the degenerate-bin path instead of erroring.
        """
        y2d = np.full((10, 2), 0.5)
        with pytest.raises(ValueError, match="1D arrays"):
            maximum_calibration_error(y2d, y2d)
        with pytest.raises(ValueError, match="1D arrays"):
            expected_calibration_error(y2d, y2d)
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            maximum_calibration_error(np.array([0, 1.0]), np.array([0.3, 0.7]), n_bins=0)

    def test_mce_empty_input_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            maximum_calibration_error(np.array([]), np.array([]))

    def test_mce_matches_ece_binning_logic(self, binary_random):
        """Shared binning core: with a single bin the weighted average and the
        maximum are the same number, so ECE == MCE exactly. Any divergence in
        edge computation or sample assignment would break this identity."""
        y_true, y_prob = binary_random
        ece = expected_calibration_error(y_true, y_prob, n_bins=1)
        mce = maximum_calibration_error(y_true, y_prob, n_bins=1)
        assert mce == pytest.approx(ece, abs=1e-15)

    def test_mce_extreme_probabilities_included(self):
        """p = 1.0 must land in the (closed) last bin, exactly like ECE."""
        y_true = np.array([1] * 9 + [0])
        y_prob = np.ones(10)
        mce = maximum_calibration_error(y_true, y_prob, n_bins=10)
        assert mce == pytest.approx(0.1, abs=1e-10)

    def test_mce_shape_mismatch_raises_clear_error(self):
        with pytest.raises(ValueError, match="Invalid input shapes") as exc_info:
            maximum_calibration_error(np.array([0, 1]), np.array([0.5, 0.5, 0.5]))
        assert "y_true has shape" in str(exc_info.value)

    def test_mce_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            maximum_calibration_error(np.array([0, 1]), np.array([0.3, 0.7]), strategy="invalid")

    def test_mce_returns_float(self, binary_random):
        assert isinstance(maximum_calibration_error(*binary_random), float)


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

    def test_quantile_zero_variance_predictions(self):
        y_true = np.zeros(100)
        y_prob = np.zeros(100)

        frac_pos, mean_pred, counts = reliability_curve(y_true, y_prob, strategy="quantile")

        np.testing.assert_allclose(frac_pos, np.array([0.0]))
        np.testing.assert_allclose(mean_pred, np.array([0.0]))
        np.testing.assert_array_equal(counts, np.array([100]))

    def test_quantile_duplicate_edges_returns_single_bin(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])

        frac_pos, mean_pred, counts = reliability_curve(y_true, y_prob, strategy="quantile")

        np.testing.assert_allclose(frac_pos, np.array([0.5]))
        np.testing.assert_allclose(mean_pred, np.array([0.5]))
        np.testing.assert_array_equal(counts, np.array([4]))
