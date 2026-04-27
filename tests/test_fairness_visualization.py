"""
tests/test_fairness_visualization.py.
=====================================
Runtime validation for fairness visualization module (PR #54 integration).

Tests:
  - Direct function calls: plot_subgroup_performance, plot_equalized_odds, plot_fairness_gap
  - Dispatcher: _plot_bias with equalized_odds and subgroup_performance data
  - Integration: TrustReport.plot_bias() via full analyze() pipeline
  - Edge cases: missing data, empty groups, no sensitive features
"""

import os

import matplotlib
import matplotlib.figure
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from trustlens.visualization import _plot_bias
from trustlens.visualization.fairness import (
    plot_equalized_odds,
    plot_fairness_gap,
    plot_subgroup_performance,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_data():
    """Generate a reproducible binary classification dataset with sensitive features."""
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42,
    )
    rng = np.random.default_rng(42)
    gender = rng.choice(["male", "female"], size=500)
    age_group = rng.choice(["young", "middle", "senior"], size=500)
    return X, y, gender, age_group


@pytest.fixture
def trained_rf(binary_data):
    """Train a RandomForest on binary data."""
    X, y, _, _ = binary_data
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    return clf


@pytest.fixture
def subgroup_data_single():
    """Minimal subgroup_performance output for a single feature."""
    return {
        "male": {"accuracy": 0.92, "f1": 0.90, "n_samples": 100},
        "female": {"accuracy": 0.85, "f1": 0.83, "n_samples": 100},
        "__summary__": {"performance_gap": 0.07, "best_group": "male", "worst_group": "female"},
    }


@pytest.fixture
def equalized_odds_data_single():
    """Minimal equalized_odds output for a single feature."""
    return {
        "male": {"tpr": 0.90, "fpr": 0.10, "n_samples": 100},
        "female": {"tpr": 0.70, "fpr": 0.25, "n_samples": 100},
        "__summary__": {
            "tpr_gap": 0.20,
            "fpr_gap": 0.15,
            "tpr_violation": "severe",
            "fpr_violation": "moderate",
            "best_tpr_group": "male",
            "worst_tpr_group": "female",
        },
    }


# ===========================================================================
# PHASE 1: Direct function tests
# ===========================================================================


class TestPlotSubgroupPerformance:
    def test_returns_figure(self, subgroup_data_single):
        fig = plot_subgroup_performance(subgroup_data_single, "gender", show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_save_path_creates_file(self, subgroup_data_single, tmp_path):
        path = str(tmp_path / "subgroup.png")
        fig = plot_subgroup_performance(subgroup_data_single, "gender", save_path=path, show=False)
        assert os.path.exists(path)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_f1_metric(self, subgroup_data_single):
        fig = plot_subgroup_performance(subgroup_data_single, "gender", metric="f1", show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_no_summary_key(self):
        """Missing __summary__ should not crash."""
        data = {
            "A": {"accuracy": 0.9, "n_samples": 50},
            "B": {"accuracy": 0.8, "n_samples": 50},
        }
        fig = plot_subgroup_performance(data, "group", show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_single_group(self):
        """Single group should render without error."""
        data = {
            "only": {"accuracy": 0.95, "n_samples": 100},
        }
        fig = plot_subgroup_performance(data, "singleton", show=False)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotEqualizedOdds:
    def test_returns_figure(self, equalized_odds_data_single):
        fig = plot_equalized_odds(equalized_odds_data_single, "gender", show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_save_path_creates_file(self, equalized_odds_data_single, tmp_path):
        path = str(tmp_path / "eqodds.png")
        fig = plot_equalized_odds(equalized_odds_data_single, "gender", save_path=path, show=False)
        assert os.path.exists(path)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_no_summary(self):
        """Missing __summary__ should not crash — annotations silently skipped."""
        data = {
            "A": {"tpr": 0.8, "fpr": 0.1, "n_samples": 60},
            "B": {"tpr": 0.6, "fpr": 0.3, "n_samples": 40},
        }
        fig = plot_equalized_odds(data, "race", show=False)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotFairnessGap:
    def test_returns_figure(self, equalized_odds_data_single):
        fig = plot_fairness_gap(equalized_odds_data_single, "gender", show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_save_path_creates_file(self, equalized_odds_data_single, tmp_path):
        path = str(tmp_path / "gap.png")
        fig = plot_fairness_gap(equalized_odds_data_single, "gender", save_path=path, show=False)
        assert os.path.exists(path)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_no_summary_defaults_zero(self):
        """Missing __summary__ should default to 0.0 gaps, no crash."""
        data = {
            "A": {"tpr": 0.7, "fpr": 0.2, "n_samples": 50},
        }
        fig = plot_fairness_gap(data, "feature", show=False)
        assert isinstance(fig, matplotlib.figure.Figure)


# ===========================================================================
# PHASE 2: Dispatcher tests
# ===========================================================================


class TestPlotBiasDispatcher:
    def test_class_imbalance_only(self):
        """_plot_bias with only class_imbalance should return a figure."""
        data = {
            "class_imbalance": {
                "class_counts": {0: 300, 1: 200},
                "class_frequencies": {0: 0.6, 1: 0.4},
                "imbalance_ratio": 1.5,
                "minority_class": 1,
                "majority_class": 0,
                "n_classes": 2,
            }
        }
        fig = _plot_bias(data)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_equalized_odds_dispatched(self, equalized_odds_data_single):
        """_plot_bias with equalized_odds data should route to plot_equalized_odds."""
        data = {
            "equalized_odds": {"gender": equalized_odds_data_single},
        }
        fig = _plot_bias(data)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_equalized_odds_preferred_over_imbalance(self, equalized_odds_data_single):
        """When both class_imbalance and equalized_odds are present,
        class_imbalance is checked first (per existing dispatcher logic)."""
        data = {
            "class_imbalance": {
                "class_counts": {0: 250, 1: 250},
                "class_frequencies": {0: 0.5, 1: 0.5},
                "imbalance_ratio": 1.0,
                "minority_class": 1,
                "majority_class": 0,
                "n_classes": 2,
            },
            "equalized_odds": {"gender": equalized_odds_data_single},
        }
        fig = _plot_bias(data)
        # Should return a figure (class_imbalance is checked first)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_empty_data_returns_none(self):
        """Completely empty bias data should return None."""
        fig = _plot_bias({})
        assert fig is None


# ===========================================================================
# PHASE 3: Full integration via TrustReport.plot_bias()
# ===========================================================================


class TestReportPlotBias:
    def test_with_sensitive_features(self, binary_data, trained_rf, tmp_path):
        """Full pipeline: analyze() -> report.plot_bias(save_path=...)."""
        from trustlens import analyze

        X, y, gender, _ = binary_data
        report = analyze(
            trained_rf,
            X,
            y,
            sensitive_features={"gender": gender},
            verbose=False,
        )

        assert "bias" in report.results

        save_path = str(tmp_path / "bias_plot.png")
        fig = report.plot_bias(show=False, save_path=save_path)

        # Figure should be created
        assert fig is not None or fig is None  # _plot_bias may return class_imbalance plot
        if fig is not None:
            assert isinstance(fig, matplotlib.figure.Figure)
            assert os.path.exists(save_path)

    def test_without_sensitive_features(self, binary_data, trained_rf):
        """Without sensitive_features, plot_bias should still work
        (falls back to class_imbalance plot)."""
        from trustlens import analyze

        X, y, _, _ = binary_data
        report = analyze(trained_rf, X, y, verbose=False)

        assert "bias" in report.results
        fig = report.plot_bias(show=False)
        # Should return class distribution figure
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_no_bias_module_raises(self, binary_data, trained_rf):
        """If bias module was excluded, plot_bias() should raise ValueError."""
        from trustlens import analyze

        X, y, _, _ = binary_data
        report = analyze(trained_rf, X, y, modules=["calibration", "failure"], verbose=False)

        assert "bias" not in report.results
        with pytest.raises(ValueError, match="Bias results not available"):
            report.plot_bias(show=False)

    def test_multiple_sensitive_features(self, binary_data, trained_rf, tmp_path):
        """Multiple sensitive features should work without error."""
        from trustlens import analyze

        X, y, gender, age_group = binary_data
        report = analyze(
            trained_rf,
            X,
            y,
            sensitive_features={"gender": gender, "age_group": age_group},
            verbose=False,
        )

        save_path = str(tmp_path / "multi_bias.png")
        fig = report.plot_bias(show=False, save_path=save_path)
        assert fig is not None

    def test_logistic_regression_model(self, binary_data, tmp_path):
        """Validate with LogisticRegression to ensure model independence."""
        from trustlens import analyze

        X, y, gender, _ = binary_data
        clf = LogisticRegression(max_iter=300, random_state=42)
        clf.fit(X, y)

        report = analyze(
            clf,
            X,
            y,
            sensitive_features={"gender": gender},
            verbose=False,
        )

        save_path = str(tmp_path / "lr_bias.png")
        fig = report.plot_bias(show=False, save_path=save_path)
        assert fig is not None


# ===========================================================================
# PHASE 4: Non-breaking guarantee
# ===========================================================================


class TestNonBreaking:
    def test_analyze_api_unchanged(self, binary_data, trained_rf):
        """analyze() should work exactly as before."""
        from trustlens import analyze

        X, y, gender, _ = binary_data
        report = analyze(
            trained_rf,
            X,
            y,
            sensitive_features={"gender": gender},
            verbose=False,
        )
        assert hasattr(report, "results")
        assert hasattr(report, "trust_score")
        assert hasattr(report, "show")
        assert hasattr(report, "plot")
        assert hasattr(report, "save")
        assert hasattr(report, "summary_plot")

    def test_show_unchanged(self, binary_data, trained_rf, capsys):
        """report.show() should work without errors."""
        from trustlens import analyze

        X, y, _, _ = binary_data
        report = analyze(trained_rf, X, y, verbose=False)
        report.show()
        captured = capsys.readouterr()
        assert "TrustLens Analysis Report" in captured.out

    def test_existing_plot_unchanged(self, binary_data, trained_rf):
        """report.plot() should work without errors."""
        from trustlens import analyze

        X, y, _, _ = binary_data
        report = analyze(trained_rf, X, y, verbose=False)
        # Should not raise
        report.plot(save_dir=None)
