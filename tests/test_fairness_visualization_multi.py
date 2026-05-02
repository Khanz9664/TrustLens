"""
tests/test_fairness_visualization.py.
=====================================
Unit tests for trustlens.visualization.fairness multi-feature wrappers.
"""

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trustlens.visualization.fairness import (
    plot_equalized_odds_multi,
    plot_fairness_gap_multi,
    plot_subgroup_performance_multi,
)


@pytest.fixture
def equalized_odds_data():
    """Multi-feature equalized_odds output fixture."""
    return {
        "gender": {
            "male": {"tpr": 0.80, "fpr": 0.10},
            "female": {"tpr": 0.65, "fpr": 0.20},
            "__summary__": {
                "tpr_gap": 0.15,
                "fpr_gap": 0.10,
                "tpr_violation": "moderate",
                "fpr_violation": "acceptable",
            },
        },
        "age": {
            "young": {"tpr": 0.78, "fpr": 0.12},
            "old": {"tpr": 0.60, "fpr": 0.25},
            "__summary__": {
                "tpr_gap": 0.18,
                "fpr_gap": 0.13,
                "tpr_violation": "moderate",
                "fpr_violation": "moderate",
            },
        },
    }


@pytest.fixture
def subgroup_data():
    """Multi-feature subgroup_performance output fixture."""
    return {
        "gender": {
            "male": {"accuracy": 0.82, "f1": 0.80},
            "female": {"accuracy": 0.70, "f1": 0.68},
            "__summary__": {"performance_gap": 0.12},
        },
        "age": {
            "young": {"accuracy": 0.79, "f1": 0.77},
            "old": {"accuracy": 0.65, "f1": 0.63},
            "__summary__": {"performance_gap": 0.14},
        },
    }


class TestPlotSubgroupPerformanceMulti:
    """Tests for plot_subgroup_performance_multi."""

    def test_all_features_processed(self, subgroup_data):
        """No feature should be silently dropped."""
        figs = plot_subgroup_performance_multi(subgroup_data, show=False)
        assert set(figs.keys()) == set(subgroup_data.keys())

    def test_output_keys_match_input(self, subgroup_data):
        """Output dict keys must exactly match input feature names."""
        figs = plot_subgroup_performance_multi(subgroup_data, show=False)
        assert list(figs.keys()) == list(subgroup_data.keys())

    def test_returns_figure_instances(self, subgroup_data):
        """Each value in the returned dict must be a matplotlib Figure."""
        figs = plot_subgroup_performance_multi(subgroup_data, show=False)
        for feature_name, fig in figs.items():
            assert isinstance(fig, plt.Figure), f"{feature_name} is not a Figure"

    def test_empty_input_returns_empty_dict(self):
        """Empty input dict should produce an empty output dict."""
        figs = plot_subgroup_performance_multi({}, show=False)
        assert figs == {}

    def test_save_dir_creates_per_feature_files(self, subgroup_data, tmp_path):
        """save_dir should produce one file per feature, named correctly."""
        plot_subgroup_performance_multi(subgroup_data, save_dir=str(tmp_path), show=False)
        for feature_name in subgroup_data:
            expected = tmp_path / f"subgroup_performance_{feature_name}.png"
            assert expected.exists(), f"{expected} was not created"

    def test_metric_argument_is_forwarded(self, subgroup_data):
        """The metric argument should be accepted and not raise."""
        figs = plot_subgroup_performance_multi(subgroup_data, metric="f1", show=False)
        assert set(figs.keys()) == set(subgroup_data.keys())


class TestPlotEqualizedOddsMulti:
    """Tests for plot_equalized_odds_multi."""

    def test_all_features_processed(self, equalized_odds_data):
        """No feature should be silently dropped."""
        figs = plot_equalized_odds_multi(equalized_odds_data, show=False)
        assert set(figs.keys()) == set(equalized_odds_data.keys())

    def test_output_keys_match_input(self, equalized_odds_data):
        """Output dict keys must exactly match input feature names."""
        figs = plot_equalized_odds_multi(equalized_odds_data, show=False)
        assert list(figs.keys()) == list(equalized_odds_data.keys())

    def test_returns_figure_instances(self, equalized_odds_data):
        """Each value in the returned dict must be a matplotlib Figure."""
        figs = plot_equalized_odds_multi(equalized_odds_data, show=False)
        for feature_name, fig in figs.items():
            assert isinstance(fig, plt.Figure), f"{feature_name} is not a Figure"

    def test_empty_input_returns_empty_dict(self):
        """Empty input dict should produce an empty output dict."""
        figs = plot_equalized_odds_multi({}, show=False)
        assert figs == {}

    def test_save_dir_creates_per_feature_files(self, equalized_odds_data, tmp_path):
        """save_dir should produce one file per feature, named correctly."""
        plot_equalized_odds_multi(equalized_odds_data, save_dir=str(tmp_path), show=False)
        for feature_name in equalized_odds_data:
            expected = tmp_path / f"equalized_odds_{feature_name}.png"
            assert expected.exists(), f"{expected} was not created"


class TestPlotFairnessGapMulti:
    """Tests for plot_fairness_gap_multi."""

    def test_all_features_processed(self, equalized_odds_data):
        """No feature should be silently dropped."""
        figs = plot_fairness_gap_multi(equalized_odds_data, show=False)
        assert set(figs.keys()) == set(equalized_odds_data.keys())

    def test_output_keys_match_input(self, equalized_odds_data):
        """Output dict keys must exactly match input feature names."""
        figs = plot_fairness_gap_multi(equalized_odds_data, show=False)
        assert list(figs.keys()) == list(equalized_odds_data.keys())

    def test_returns_figure_instances(self, equalized_odds_data):
        """Each value in the returned dict must be a matplotlib Figure."""
        figs = plot_fairness_gap_multi(equalized_odds_data, show=False)
        for feature_name, fig in figs.items():
            assert isinstance(fig, plt.Figure), f"{feature_name} is not a Figure"

    def test_empty_input_returns_empty_dict(self):
        """Empty input dict should produce an empty output dict."""
        figs = plot_fairness_gap_multi({}, show=False)
        assert figs == {}

    def test_save_dir_creates_per_feature_files(self, equalized_odds_data, tmp_path):
        """save_dir should produce one file per feature, named correctly."""
        plot_fairness_gap_multi(equalized_odds_data, save_dir=str(tmp_path), show=False)
        for feature_name in equalized_odds_data:
            expected = tmp_path / f"fairness_gap_{feature_name}.png"
            assert expected.exists(), f"{expected} was not created"
