"""
tests/test_bias.py.
==================
Unit tests for trustlens.metrics.bias.
"""

import numpy as np
import pytest

from trustlens.visualization.bias_plots import plot_class_distribution
from trustlens.metrics.bias import class_imbalance_report, subgroup_performance


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

class TestBiasPlots:
    def test_single_class_distribution_plot(self):
        y_true = np.array([0, 0, 0, 0])

        imbalance_data = class_imbalance_report(y_true)

        fig = plot_class_distribution(
            imbalance_data,
            show=False,
        )

        assert fig is not None
        assert len(fig.axes) > 0

        title = fig.axes[0].get_title()

        assert "Single class detected" in title