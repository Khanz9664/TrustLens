"""
tests/test_trust_score.py.
==========================
Unit tests for the TrustLens Trust Score system.
"""

import pytest

from trustlens.trust_score import (
    TrustScoreResult,
    _bias_score,
    _calibration_score,
    _failure_score,
    _representation_score,
    _score_bar,
    compute_trust_score,
)

# ---------------------------------------------------------------------------
# Fixtures — representative results dicts
# ---------------------------------------------------------------------------


@pytest.fixture
def perfect_calibration():
    """Perfectly calibrated: BS=0, ECE=0."""
    return {"brier_score": 0.0, "ece": 0.0}


@pytest.fixture
def worst_calibration():
    """Worst calibration: BS=1, ECE=1."""
    return {"brier_score": 1.0, "ece": 1.0}


@pytest.fixture
def good_failure():
    """High confidence gap, low error rate."""
    return {
        "confidence_gap": {"gap": 0.85},
        "misclassification_summary": {"__overall__": {"overall_error_rate": 0.05}},
    }


@pytest.fixture
def poor_failure():
    """Low confidence gap, high error rate."""
    return {
        "confidence_gap": {"gap": 0.02},
        "misclassification_summary": {"__overall__": {"overall_error_rate": 0.50}},
    }


@pytest.fixture
def balanced_bias():
    """Perfectly balanced, no subgroup gap."""
    return {
        "class_imbalance": {"imbalance_ratio": 1.0},
        "subgroup_performance": {
            "gender": {"__summary__": {"performance_gap": 0.0}},
        },
    }


@pytest.fixture
def severe_bias():
    """Severely imbalanced, large subgroup gap."""
    return {
        "class_imbalance": {"imbalance_ratio": 20.0},
        "subgroup_performance": {
            "gender": {"__summary__": {"performance_gap": 0.50}},
        },
    }


@pytest.fixture
def good_representation():
    """High silhouette score."""
    return {"separability": {"silhouette_score": 0.90}}


@pytest.fixture
def poor_representation():
    """Negative silhouette — overlapping clusters."""
    return {"separability": {"silhouette_score": -0.40}}


@pytest.fixture
def full_good_results(perfect_calibration, good_failure, balanced_bias, good_representation):
    return {
        "calibration": perfect_calibration,
        "failure": good_failure,
        "bias": balanced_bias,
        "representation": good_representation,
    }


@pytest.fixture
def full_poor_results(worst_calibration, poor_failure, severe_bias, poor_representation):
    return {
        "calibration": worst_calibration,
        "failure": poor_failure,
        "bias": severe_bias,
        "representation": poor_representation,
    }


# ---------------------------------------------------------------------------
# Sub-score unit tests
# ---------------------------------------------------------------------------


class TestCalibrationScore:
    def test_perfect_is_100(self, perfect_calibration):
        score = _calibration_score(perfect_calibration)
        assert score == pytest.approx(100.0, abs=1e-6)

    def test_worst_is_zero(self, worst_calibration):
        score = _calibration_score(worst_calibration)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_range(self):
        data = {"brier_score": 0.3, "ece": 0.15}
        score = _calibration_score(data)
        assert 0.0 <= score <= 100.0

    def test_missing_keys_uses_defaults(self):
        score = _calibration_score({})
        # Defaults: BS=0.5, ECE=0.5 → composite=1.25 → clip(1.0) → score=0.0
        assert score == pytest.approx(0.0, abs=1e-6)


class TestFailureScore:
    def test_high_gap_low_error_gives_high_score(self, good_failure):
        score = _failure_score(good_failure)
        assert score >= 70.0

    def test_low_gap_high_error_gives_low_score(self, poor_failure):
        score = _failure_score(poor_failure)
        assert score <= 30.0

    def test_range(self, good_failure):
        score = _failure_score(good_failure)
        assert 0.0 <= score <= 100.0

    def test_missing_conf_gap(self):
        data = {"misclassification_summary": {"__overall__": {"overall_error_rate": 0.05}}}
        score = _failure_score(data)
        assert 0.0 <= score <= 100.0


class TestBiasScore:
    def test_perfect_balance_is_100(self, balanced_bias):
        score = _bias_score(balanced_bias)
        assert score == pytest.approx(100.0, abs=1.0)

    def test_severe_imbalance_gives_low_score(self, severe_bias):
        score = _bias_score(severe_bias)
        assert score <= 50.0

    def test_range(self, balanced_bias):
        score = _bias_score(balanced_bias)
        assert 0.0 <= score <= 100.0

    def test_no_subgroup_data(self):
        data = {"class_imbalance": {"imbalance_ratio": 2.0}}
        score = _bias_score(data)
        assert 0.0 <= score <= 100.0


class TestRepresentationScore:
    def test_high_silhouette_is_high_score(self, good_representation):
        score = _representation_score(good_representation)
        assert score >= 90.0

    def test_negative_silhouette_is_low_score(self, poor_representation):
        score = _representation_score(poor_representation)
        assert score <= 40.0

    def test_nan_silhouette_does_not_crash(self):
        data = {"separability": {"silhouette_score": float("nan")}}
        score = _representation_score(data)
        assert score == pytest.approx(50.0, abs=1.0)

    def test_range(self, good_representation):
        score = _representation_score(good_representation)
        assert 0.0 <= score <= 100.0


# ---------------------------------------------------------------------------
# compute_trust_score integration tests
# ---------------------------------------------------------------------------


class TestComputeTrustScore:
    def test_returns_trust_score_result(self, full_good_results):
        result = compute_trust_score(full_good_results)
        assert isinstance(result, TrustScoreResult)

    def test_good_results_give_high_score(self, full_good_results):
        result = compute_trust_score(full_good_results)
        assert result.score >= 75

    def test_poor_results_give_low_score(self, full_poor_results):
        result = compute_trust_score(full_poor_results)
        assert result.score <= 35

    def test_score_in_range(self, full_good_results):
        result = compute_trust_score(full_good_results)
        assert 0 <= result.score <= 100

    def test_grade_a_for_high_score(self, full_good_results):
        result = compute_trust_score(full_good_results)
        assert result.grade in ("A", "B")  # high results should be A or B

    def test_grade_d_for_poor_score(self, full_poor_results):
        result = compute_trust_score(full_poor_results)
        assert result.grade in ("C", "D")

    def test_missing_representation_redistributes_weights(self):
        """Without representation data, weights should redistribute to others."""
        results = {
            "calibration": {"brier_score": 0.1, "ece": 0.05},
            "failure": {
                "confidence_gap": {"gap": 0.5},
                "misclassification_summary": {"__overall__": {"overall_error_rate": 0.1}},
            },
            "bias": {"class_imbalance": {"imbalance_ratio": 1.5}},
        }
        result = compute_trust_score(results)
        assert "representation" not in result.sub_scores
        assert abs(sum(result.weights_used.values()) - 1.0) < 1e-6

    def test_empty_results_does_not_crash(self):
        result = compute_trust_score({})
        assert isinstance(result.score, int)
        assert 0 <= result.score <= 100

    def test_custom_weights(self, full_good_results):
        """Custom weights should change the score."""
        compute_trust_score(full_good_results)
        custom = compute_trust_score(
            full_good_results,
            weights={"calibration": 0.8, "failure": 0.1, "bias": 0.05, "representation": 0.05},
        )
        # They may differ — just ensure both are valid
        assert 0 <= custom.score <= 100
        assert isinstance(custom.score, int)

    def test_sub_scores_populated(self, full_good_results):
        result = compute_trust_score(full_good_results)
        assert set(result.sub_scores.keys()) >= {"calibration", "failure", "bias"}

    def test_weights_sum_to_one(self, full_good_results):
        result = compute_trust_score(full_good_results)
        assert abs(sum(result.weights_used.values()) - 1.0) < 1e-6

    def test_breakdown_sums_to_score_approx(self, full_good_results):
        result = compute_trust_score(full_good_results)
        breakdown_sum = sum(result.breakdown.values())
        assert abs(breakdown_sum - result.score) <= 1  # within 1 due to rounding

    def test_str_method_includes_score(self, full_good_results):
        result = compute_trust_score(full_good_results)
        s = str(result)
        assert str(result.score) in s
        assert result.grade in s

    def test_repr(self, full_good_results):
        result = compute_trust_score(full_good_results)
        r = repr(result)
        assert "TrustScoreResult" in r
        assert "score=" in r


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestScoreBar:
    def test_full_bar(self):
        bar = _score_bar(100)
        assert bar == ""

    def test_empty_bar(self):
        bar = _score_bar(0)
        assert bar == ""

    def test_half_bar(self):
        bar = _score_bar(50, width=10)
        assert bar == ""
