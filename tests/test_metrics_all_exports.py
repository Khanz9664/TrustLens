"""Tests for public __all__ exports in metrics modules."""

from trustlens.metrics import bias, calibration, failure, representation


def test_calibration_all_exports_public_metrics():
    assert set(calibration.__all__) == {
        "brier_score",
        "expected_calibration_error",
        "maximum_calibration_error",
        "reliability_curve",
    }


def test_failure_all_exports_public_metrics():
    assert set(failure.__all__) == {
        "misclassification_summary",
        "confidence_gap",
    }


def test_bias_all_exports_public_metrics():
    assert set(bias.__all__) == {
        "class_imbalance_report",
        "subgroup_performance",
        "equalized_odds",
    }


def test_representation_all_exports_public_metrics():
    assert set(representation.__all__) == {
        "embedding_separability",
        "centered_kernel_alignment",
    }
