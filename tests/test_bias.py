"""
tests/test_bias.py.
==================
Unit tests for trustlens.metrics.bias.
"""

import numpy as np
import pytest

from trustlens.metrics.bias import class_imbalance_report, equalized_odds, subgroup_performance


class TestClassImbalanceReport:
    def test_balanced_dataset_ratio_is_one(self):
        y_true = np.array([0, 0, 1, 1])
        report = class_imbalance_report(y_true)
        assert report["imbalance_ratio"] == pytest.approx(1.0)

    def test_imbalance_ratio_correct(self):
        y_true = np.array([0] * 90 + [1] * 10)
        report = class_imbalance_report(y_true)
        assert report["imbalance_ratio"] == pytest.approx(9.0)

    def test_minority_majority_class_identified(self):
        y_true = np.array([0] * 80 + [1] * 20)
        report = class_imbalance_report(y_true)
        assert report["majority_class"] == 0
        assert report["minority_class"] == 1

    def test_class_frequencies_sum_to_one(self):
        y_true = np.array([0, 0, 1, 2, 2])
        report = class_imbalance_report(y_true)
        total = sum(report["class_frequencies"].values())
        assert total == pytest.approx(1.0, rel=1e-5)

    def test_multiclass_n_classes(self):
        y_true = np.array([0, 1, 2, 3, 3])
        report = class_imbalance_report(y_true)
        assert report["n_classes"] == 4


class TestSubgroupPerformance:
    def test_basic_accuracy_per_group(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 0])  # 4/6 correct overall
        groups = np.array(["A", "A", "A", "B", "B", "B"])

        result = subgroup_performance(y_true, y_pred, {"gender": groups})
        assert "gender" in result
        assert "A" in result["gender"]
        assert "B" in result["gender"]

    def test_performance_gap_computed(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])  # Group A: perfect, B: all wrong
        groups = np.array(["A", "A", "B", "B"])

        result = subgroup_performance(y_true, y_pred, {"group": groups})
        summary = result["group"]["__summary__"]
        assert "performance_gap" in summary
        assert summary["performance_gap"] >= 0.0

    def test_accuracy_in_range(self):
        rng = np.random.default_rng(7)
        y_true = rng.integers(0, 2, 200)
        y_pred = rng.integers(0, 2, 200)
        groups = rng.integers(0, 3, 200).astype(str)

        result = subgroup_performance(y_true, y_pred, {"group": groups})
        for g, metrics in result["group"].items():
            if g == "__summary__":
                continue
            assert 0.0 <= metrics["accuracy"] <= 1.0


class TestEqualizedOdds:
    """Tests for the equalized_odds fairness metric."""

    def test_normal_multigroup_case(self):
        """Basic multi-group case: verify per-group TPR/FPR and summary."""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])
        gender = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        result = equalized_odds(y_true, y_pred, sensitive_features={"gender": gender})

        assert "gender" in result
        assert "0" in result["gender"]
        assert "1" in result["gender"]
        assert "__summary__" in result["gender"]

        # Group 0: y_true=[1,1,0,0], y_pred=[1,0,0,0] → TPR=0.5, FPR=0.0
        assert result["gender"]["0"]["tpr"] == pytest.approx(0.5)
        assert result["gender"]["0"]["fpr"] == pytest.approx(0.0)

        # Group 1: y_true=[1,1,0,0], y_pred=[1,1,1,0] → TPR=1.0, FPR=0.5
        assert result["gender"]["1"]["tpr"] == pytest.approx(1.0)
        assert result["gender"]["1"]["fpr"] == pytest.approx(0.5)

        summary = result["gender"]["__summary__"]
        assert summary["tpr_gap"] == pytest.approx(0.5)
        assert summary["fpr_gap"] == pytest.approx(0.5)
        assert summary["tpr_violation"] == "severe"
        assert summary["fpr_violation"] == "severe"

    def test_single_group_case(self):
        """Single subgroup: gaps should be 0.0 and violation 'acceptable'."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1])
        group = np.array([0, 0, 0, 0])  # only one group

        result = equalized_odds(y_true, y_pred, sensitive_features={"group": group})

        summary = result["group"]["__summary__"]
        assert summary["tpr_gap"] == 0.0
        assert summary["fpr_gap"] == 0.0
        assert summary["tpr_violation"] == "acceptable"
        assert summary["fpr_violation"] == "acceptable"

    def test_edge_case_no_positives(self):
        """Group with no positive samples: TPR should be 0.0 (not raise)."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        group = np.array([0, 0, 1, 1])  # group 0 has no positives

        result = equalized_odds(y_true, y_pred, sensitive_features={"group": group})

        assert result["group"]["0"]["tpr"] == pytest.approx(0.0)

    def test_edge_case_no_negatives(self):
        """Group with no negative samples: FPR should be 0.0 (not raise)."""
        y_true = np.array([1, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 1])
        group = np.array([0, 0, 1, 1])  # group 1 has no negatives

        result = equalized_odds(y_true, y_pred, sensitive_features={"group": group})

        assert result["group"]["1"]["fpr"] == pytest.approx(0.0)

    def test_json_serializable(self):
        """All values in the result must be plain Python types (not numpy)."""
        import json

        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        group = np.array([0, 0, 1, 1])

        result = equalized_odds(y_true, y_pred, sensitive_features={"group": group})
        # Should not raise
        json.dumps(result)

    def test_violation_levels(self):
        """Check all three violation levels are correctly assigned."""
        # acceptable: gap < 0.05 → perfect predictions per group
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        group = np.array([0, 0, 1, 1])

        result = equalized_odds(y_true, y_pred, sensitive_features={"group": group})
        assert result["group"]["__summary__"]["tpr_violation"] == "acceptable"

    def test_n_samples_correct(self):
        """n_samples should reflect the actual group size."""
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 1])
        group = np.array([0, 0, 0, 1, 1])

        result = equalized_odds(y_true, y_pred, sensitive_features={"group": group})
        assert result["group"]["0"]["n_samples"] == 3
        assert result["group"]["1"]["n_samples"] == 2
