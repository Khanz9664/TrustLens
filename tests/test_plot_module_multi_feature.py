"""
tests/test_plot_module_multi_feature.py.
========================================
Unit tests for plot_module and _plot_bias integration with multi-feature
fairness visualizations.
"""

import os

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trustlens.visualization import plot_module
from trustlens.visualization import _plot_bias


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def equalized_odds_data():
    """Multi-feature equalized_odds bias data keyed by ``equalized_odds``."""
    return {
        "equalized_odds": {
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
    }


@pytest.fixture
def subgroup_data():
    """Multi-feature subgroup_performance bias data."""
    return {
        "subgroup_performance": {
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
    }


@pytest.fixture
def combined_data(equalized_odds_data, subgroup_data):
    """Bias data with both equalized_odds and subgroup_performance."""
    return {**equalized_odds_data, **subgroup_data}


@pytest.fixture
def class_imbalance_data():
    """Bias data with only class_imbalance."""
    return {
        "class_imbalance": {
            "class_counts": {0: 900, 1: 100},
            "imbalance_ratio": 9.0,
        }
    }


# ---------------------------------------------------------------------------
# _plot_bias — pure figure generation
# ---------------------------------------------------------------------------


class TestPlotBiasReturnStructure:
    """Verify _plot_bias returns correct shapes without any file I/O."""

    def test_equalized_odds_returns_nested_dict(self, equalized_odds_data):
        """equalized_odds data produces nested dict with both plot types."""
        result = _plot_bias(equalized_odds_data)
        assert isinstance(result, dict)
        assert "equalized_odds" in result
        assert "fairness_gap" in result
        assert set(result["equalized_odds"].keys()) == {"gender", "age"}
        assert set(result["fairness_gap"].keys()) == {"gender", "age"}

    def test_subgroup_returns_nested_dict(self, subgroup_data):
        """subgroup_performance data produces nested dict with subgroup key."""
        result = _plot_bias(subgroup_data)
        assert isinstance(result, dict)
        assert "subgroup" in result
        assert set(result["subgroup"].keys()) == {"gender", "age"}

    def test_combined_returns_all_three_keys(self, combined_data):
        """Combined data produces all three plot-type keys."""
        result = _plot_bias(combined_data)
        assert isinstance(result, dict)
        assert "subgroup" in result
        assert "equalized_odds" in result
        assert "fairness_gap" in result

    def test_class_imbalance_returns_single_figure(self, class_imbalance_data):
        """class_imbalance data returns a single Figure, not a dict."""
        result = _plot_bias(class_imbalance_data)
        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_empty_data_returns_none(self):
        """Empty bias data returns None."""
        result = _plot_bias({})
        assert result is None

    def test_all_figures_are_valid(self, equalized_odds_data):
        """Every value in the nested dict must be a matplotlib Figure."""
        result = _plot_bias(equalized_odds_data)
        for plot_type, feature_figs in result.items():
            for feature_name, fig in feature_figs.items():
                assert isinstance(fig, plt.Figure), (
                    f"{plot_type}/{feature_name} is not a Figure"
                )

    def test_mixed_availability_subgroup_only(self, subgroup_data):
        """Only subgroup_performance → result has only 'subgroup' key."""
        result = _plot_bias(subgroup_data)
        assert isinstance(result, dict)
        assert "subgroup" in result
        assert "equalized_odds" not in result
        assert "fairness_gap" not in result

    def test_no_empty_keys_in_result(self, equalized_odds_data):
        """Result should never contain empty dict values."""
        result = _plot_bias(equalized_odds_data)
        for key, value in result.items():
            assert value, f"Key '{key}' has an empty value"

    def test_deterministic_feature_order(self, equalized_odds_data):
        """Feature keys should be in sorted order."""
        result = _plot_bias(equalized_odds_data)
        for plot_type, feature_figs in result.items():
            keys = list(feature_figs.keys())
            assert keys == sorted(keys), (
                f"{plot_type}: keys {keys} are not sorted"
            )


# ---------------------------------------------------------------------------
# plot_module — saving + orchestration
# ---------------------------------------------------------------------------


class TestPlotModuleSaving:
    """Verify plot_module saves ALL figures correctly."""

    def test_equalized_odds_saves_all_files(self, equalized_odds_data, tmp_path):
        """All equalized_odds + fairness_gap files should be saved."""
        save_dir = str(tmp_path)
        plot_module("bias", equalized_odds_data, save_dir=save_dir)

        expected_files = [
            "bias_equalized_odds_age.png",
            "bias_equalized_odds_gender.png",
            "bias_fairness_gap_age.png",
            "bias_fairness_gap_gender.png",
        ]
        for fname in expected_files:
            assert (tmp_path / fname).exists(), f"{fname} was not created"

    def test_subgroup_saves_all_files(self, subgroup_data, tmp_path):
        """All subgroup files should be saved."""
        save_dir = str(tmp_path)
        plot_module("bias", subgroup_data, save_dir=save_dir)

        expected_files = [
            "bias_subgroup_age.png",
            "bias_subgroup_gender.png",
        ]
        for fname in expected_files:
            assert (tmp_path / fname).exists(), f"{fname} was not created"

    def test_combined_saves_all_files(self, combined_data, tmp_path):
        """Combined data saves all 6 files."""
        save_dir = str(tmp_path)
        plot_module("bias", combined_data, save_dir=save_dir)

        expected_files = [
            "bias_subgroup_age.png",
            "bias_subgroup_gender.png",
            "bias_equalized_odds_age.png",
            "bias_equalized_odds_gender.png",
            "bias_fairness_gap_age.png",
            "bias_fairness_gap_gender.png",
        ]
        for fname in expected_files:
            assert (tmp_path / fname).exists(), f"{fname} was not created"

    def test_class_imbalance_saves_single_file(self, class_imbalance_data, tmp_path):
        """class_imbalance saves as bias_plot.png (existing behaviour)."""
        save_dir = str(tmp_path)
        plot_module("bias", class_imbalance_data, save_dir=save_dir)
        assert (tmp_path / "bias_plot.png").exists()

    def test_returns_none(self, equalized_odds_data, tmp_path):
        """plot_module is a void function — always returns None."""
        result = plot_module("bias", equalized_odds_data, save_dir=str(tmp_path))
        assert result is None

    def test_empty_data_no_crash(self, tmp_path):
        """Empty bias data does not crash and saves nothing."""
        plot_module("bias", {}, save_dir=str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_no_save_dir_no_crash(self, equalized_odds_data):
        """Calling without save_dir generates figures but saves nothing."""
        plot_module("bias", equalized_odds_data, save_dir=None)

    def test_directory_created_if_missing(self, equalized_odds_data, tmp_path):
        """save_dir is created automatically if it doesn't exist."""
        nested_dir = tmp_path / "a" / "b" / "c"
        plot_module("bias", equalized_odds_data, save_dir=str(nested_dir))
        assert nested_dir.exists()
        assert any(nested_dir.iterdir())


# ---------------------------------------------------------------------------
# Filename sanitization
# ---------------------------------------------------------------------------


class TestFilenameSanitization:
    """Verify feature names with special characters produce safe filenames."""

    def test_spaces_sanitized(self, tmp_path):
        """Feature name 'age group' should become 'age_group'."""
        data = {
            "subgroup_performance": {
                "age group": {
                    "young": {"accuracy": 0.79, "f1": 0.77},
                    "old": {"accuracy": 0.65, "f1": 0.63},
                    "__summary__": {"performance_gap": 0.14},
                },
            }
        }
        plot_module("bias", data, save_dir=str(tmp_path))
        assert (tmp_path / "bias_subgroup_age_group.png").exists()

    def test_symbols_sanitized(self, tmp_path):
        """Feature name 'race/ethnicity' should become 'race_ethnicity'."""
        data = {
            "equalized_odds": {
                "race/ethnicity": {
                    "group_a": {"tpr": 0.80, "fpr": 0.10},
                    "group_b": {"tpr": 0.65, "fpr": 0.20},
                    "__summary__": {
                        "tpr_gap": 0.15,
                        "fpr_gap": 0.10,
                        "tpr_violation": "moderate",
                        "fpr_violation": "acceptable",
                    },
                },
            }
        }
        plot_module("bias", data, save_dir=str(tmp_path))
        assert (tmp_path / "bias_equalized_odds_race_ethnicity.png").exists()
        assert (tmp_path / "bias_fairness_gap_race_ethnicity.png").exists()


# ---------------------------------------------------------------------------
# Non-bias modules — untouched behaviour
# ---------------------------------------------------------------------------


class TestNonBiasModulesUnchanged:
    """Verify non-bias modules continue to work exactly as before."""

    def test_unknown_module_no_crash(self, tmp_path):
        """Unknown module name returns None silently."""
        result = plot_module("unknown", {}, save_dir=str(tmp_path))
        assert result is None

    def test_calibration_without_data_no_crash(self, tmp_path):
        """Calibration module with missing data returns None."""
        result = plot_module("calibration", {}, save_dir=str(tmp_path))
        assert result is None

    def test_failure_without_data_no_crash(self, tmp_path):
        """Failure module with missing data returns None."""
        result = plot_module("failure", {}, save_dir=str(tmp_path))
        assert result is None

    def test_representation_without_data_no_crash(self, tmp_path):
        """Representation module with missing data returns None."""
        result = plot_module("representation", {}, save_dir=str(tmp_path))
        assert result is None
