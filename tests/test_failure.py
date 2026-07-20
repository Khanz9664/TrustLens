"""
tests/test_failure.py.
======================
Unit tests for trustlens.metrics.failure.
"""

import numpy as np
import pytest

from trustlens.metrics.failure import confidence_gap, misclassification_summary


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    n = 100
    y_true = rng.integers(0, 2, n)
    y_pred = y_true.copy()
    # Flip 20% of predictions
    flip_idx = rng.choice(n, 20, replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]
    y_prob = np.zeros((n, 2))
    y_prob[np.arange(n), y_pred] = 0.9
    y_prob[np.arange(n), 1 - y_pred] = 0.1
    return y_true, y_pred, y_prob


class TestMisclassificationSummary:
    def test_overall_error_rate_correct(self, binary_data):
        y_true, y_pred, y_prob = binary_data
        summary = misclassification_summary(y_true, y_pred, y_prob)
        expected_errors = int((y_true != y_pred).sum())
        assert summary["__overall__"]["total_errors"] == expected_errors

    def test_per_class_entries_exist(self, binary_data):
        y_true, y_pred, y_prob = binary_data
        summary = misclassification_summary(y_true, y_pred, y_prob)
        for cls in np.unique(y_true):
            assert int(cls) in summary

    def test_error_rate_range(self, binary_data):
        y_true, y_pred, y_prob = binary_data
        summary = misclassification_summary(y_true, y_pred, y_prob)
        for cls in np.unique(y_true):
            assert 0.0 <= summary[int(cls)]["error_rate"] <= 1.0

    def test_perfect_classifier_has_zero_errors(self):
        y_true = np.array([0, 1, 0, 1])
        summary = misclassification_summary(y_true, y_true, np.eye(2)[y_true])
        assert summary["__overall__"]["total_errors"] == 0

    def test_top_mistake_indices_are_global_dataset_indices(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        y_prob = np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.7, 0.3],
                [0.1, 0.9],
                [0.9, 0.1],
                [0.2, 0.8],
            ]
        )

        summary = misclassification_summary(y_true, y_pred, y_prob)

        assert summary[0]["top_mistake_indices"] == [1]
        assert summary[1]["top_mistake_indices"] == [4]


class TestConfidenceGap:
    def test_gap_nonnegative_for_good_model(self, binary_data):
        y_true, y_pred, y_prob = binary_data
        result = confidence_gap(y_true, y_pred, y_prob)
        # A model predicting 0.9 for correct class should have positive gap
        assert result["gap"] >= 0.0

    def test_contains_expected_keys(self, binary_data):
        y_true, y_pred, y_prob = binary_data
        result = confidence_gap(y_true, y_pred, y_prob)
        expected_keys = {
            "correct_confidence_mean",
            "incorrect_confidence_mean",
            "gap",
            "histogram_bins",
            "correct_hist",
            "incorrect_hist",
            "n_correct",
            "n_incorrect",
        }
        assert expected_keys.issubset(result.keys())

    def test_sample_counts_sum_to_total(self, binary_data):
        y_true, y_pred, y_prob = binary_data
        result = confidence_gap(y_true, y_pred, y_prob)
        assert result["n_correct"] + result["n_incorrect"] == len(y_true)

    def test_gap_zero_when_all_correct(self, binary_data):
        y_true, _, y_prob = binary_data
        y_pred = y_true.copy()  # 100% accuracy
        result = confidence_gap(y_true, y_pred, y_prob)
        assert result["gap"] == 0.0
        assert not np.isnan(result["gap"])
        assert result["n_incorrect"] == 0
        assert result["n_correct"] == len(y_true)

    def test_gap_zero_when_all_incorrect(self, binary_data):
        y_true, _, y_prob = binary_data
        y_pred = 1 - y_true  # 0% accuracy
        result = confidence_gap(y_true, y_pred, y_prob)
        assert result["gap"] == 0.0
        assert not np.isnan(result["gap"])
        assert result["n_correct"] == 0
        assert result["n_incorrect"] == len(y_true)
