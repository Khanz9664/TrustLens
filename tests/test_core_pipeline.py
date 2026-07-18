import numpy as np
import pytest

from trustlens.core.pipeline import _encode_labels_for_probability_columns


def test_encode_labels_rejects_class_label_length_mismatch():
    y_true = np.array(["mouse", "cat"])
    class_labels = np.array(["mouse", "cat"])

    with pytest.raises(
        ValueError, match=r"class_labels length \(2\).*probability column shape \(3 columns\)"
    ):
        _encode_labels_for_probability_columns(y_true, 3, class_labels)


def test_binary_calibration_with_semantic_labels():
    from trustlens.core.pipeline import _run_analysis_pipeline

    y_true_int = np.array([0, 1, 0, 1])
    y_true_str = np.array(["NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE"])
    class_labels = np.array(["NEGATIVE", "POSITIVE"])
    y_prob = np.array([[0.8, 0.2], [0.1, 0.9], [0.6, 0.4], [0.3, 0.7]])

    # Run with integer labels
    report_int = _run_analysis_pipeline(
        model=None,
        X=np.zeros((4, 2)),
        y_true=y_true_int,
        y_pred=y_true_int,
        y_prob=y_prob,
        modules=["calibration"],
        class_labels=np.array([0, 1]),
        verbose=False,
    )

    # Run with string labels
    report_str = _run_analysis_pipeline(
        model=None,
        X=np.zeros((4, 2)),
        y_true=y_true_str,
        y_pred=y_true_str,
        y_prob=y_prob,
        modules=["calibration"],
        class_labels=class_labels,
        verbose=False,
    )

    # Verify metrics are identical
    metrics_int = report_int.results["calibration"]
    metrics_str = report_str.results["calibration"]

    assert metrics_int["ece"] == metrics_str["ece"]
    assert metrics_int["brier_score"] == metrics_str["brier_score"]
    assert metrics_int["mce"] == metrics_str["mce"]
