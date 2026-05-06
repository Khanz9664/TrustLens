"""
tests/test_plot_bias_multi_feature.py.
======================================
Tests for the ``multi_feature`` parameter of ``TrustReport.plot_bias()``.

These tests cover the four cells of the ``(mode, multi_feature)`` return-shape
table, partial-data handling, deterministic ordering, and the interaction
between ``multi_feature=True`` and an invalid ``mode``.
"""

from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from trustlens.report import TrustReport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_report(bias_data: dict) -> TrustReport:
    """Build a minimal TrustReport whose bias module holds ``bias_data``."""
    model = MagicMock()
    X = np.zeros((10, 2))
    y_true = np.array([0, 1] * 5)
    y_pred = np.array([0, 1] * 5)
    y_prob = np.random.rand(10, 2)
    return TrustReport({"bias": bias_data}, model, X, y_true, y_pred, y_prob)


@pytest.fixture
def multi_feature_bias_data() -> dict:
    """Bias data with two sensitive features (gender, age) on both modes."""
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
        },
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
        },
    }


@pytest.fixture
def multi_feature_report(multi_feature_bias_data) -> TrustReport:
    return _make_report(multi_feature_bias_data)


# ---------------------------------------------------------------------------
# 1. Return-shape matrix: four (mode, multi_feature) cells
# ---------------------------------------------------------------------------


class TestReturnShapeMatrix:
    """Cover the four cells of the (mode, multi_feature) return-shape table."""

    def test_single_mode_multi_feature_false_returns_figure(self, multi_feature_report):
        """Single mode + multi_feature=False -> Figure (current behavior)."""
        for mode in ("subgroup", "equalized_odds", "gap"):
            fig = multi_feature_report.plot_bias(mode=mode, multi_feature=False, show=False)
            assert isinstance(fig, plt.Figure), f"mode={mode} did not return a Figure"

    def test_single_mode_multi_feature_true_returns_dict_of_figures(self, multi_feature_report):
        """Single mode + multi_feature=True -> dict[str, Figure] keyed by feature."""
        for mode in ("subgroup", "equalized_odds", "gap"):
            result = multi_feature_report.plot_bias(mode=mode, multi_feature=True, show=False)
            assert isinstance(result, dict), f"mode={mode} did not return a dict"
            assert set(result.keys()) == {"gender", "age"}
            for feat, fig in result.items():
                assert isinstance(fig, plt.Figure), f"mode={mode}, feature={feat}: not a Figure"

    def test_all_mode_multi_feature_false_returns_dict_of_figures(self, multi_feature_report):
        """mode='all' + multi_feature=False -> dict[mode, Figure] (current behavior)."""
        result = multi_feature_report.plot_bias(mode="all", multi_feature=False, show=False)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"subgroup", "equalized_odds", "gap"}
        for fig in result.values():
            assert isinstance(fig, plt.Figure)

    def test_all_mode_multi_feature_true_returns_nested_dict(self, multi_feature_report):
        """mode='all' + multi_feature=True -> dict[mode, dict[feature, Figure]]."""
        result = multi_feature_report.plot_bias(mode="all", multi_feature=True, show=False)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"subgroup", "equalized_odds", "gap"}
        for mode_key, inner in result.items():
            assert isinstance(inner, dict), f"{mode_key} value is not a dict"
            assert set(inner.keys()) == {"gender", "age"}
            for fig in inner.values():
                assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# 2. summary + multi_feature=True passthrough (spec 1)
# ---------------------------------------------------------------------------


class TestSummaryMultiFeaturePassthrough:
    """``summary`` mode with ``multi_feature=True`` must behave identically
    to ``multi_feature=False`` (a feature-agnostic summary view)."""

    def test_summary_multi_true_matches_summary_multi_false_type(self, multi_feature_report):
        result_false = multi_feature_report.plot_bias(
            mode="summary", multi_feature=False, show=False
        )
        result_true = multi_feature_report.plot_bias(mode="summary", multi_feature=True, show=False)
        # Both must produce the same return *type* (single-figure flow);
        # this is the "pass-through" guarantee from the spec.
        assert type(result_false) is type(result_true)


# ---------------------------------------------------------------------------
# 3. Partial / missing data
# ---------------------------------------------------------------------------


class TestPartialDataHandling:
    """When some bias components are missing, the structure must be
    preserved and missing slots must be ``{}`` (never ``None``)."""

    def test_all_multi_only_subgroup_present(self, multi_feature_bias_data):
        """Only subgroup_performance available."""
        bias = {"subgroup_performance": multi_feature_bias_data["subgroup_performance"]}
        report = _make_report(bias)
        result = report.plot_bias(mode="all", multi_feature=True, show=False)

        assert set(result.keys()) == {"subgroup", "equalized_odds", "gap"}
        assert set(result["subgroup"].keys()) == {"gender", "age"}
        # No equalized_odds data -> empty dict, not None
        assert result["equalized_odds"] == {}
        # 'gap' falls back to subgroup_performance, so it is populated
        assert set(result["gap"].keys()) == {"gender", "age"}

    def test_all_multi_only_equalized_odds_present(self, multi_feature_bias_data):
        """Only equalized_odds available."""
        bias = {"equalized_odds": multi_feature_bias_data["equalized_odds"]}
        report = _make_report(bias)
        result = report.plot_bias(mode="all", multi_feature=True, show=False)

        assert set(result.keys()) == {"subgroup", "equalized_odds", "gap"}
        assert result["subgroup"] == {}
        assert set(result["equalized_odds"].keys()) == {"gender", "age"}
        # 'gap' falls back to equalized_odds when subgroup is absent
        assert set(result["gap"].keys()) == {"gender", "age"}

    def test_all_multi_no_values_are_none(self, multi_feature_bias_data):
        """Every value at every level of the return must be a dict or Figure;
        explicitly never ``None``."""
        bias = {"subgroup_performance": multi_feature_bias_data["subgroup_performance"]}
        report = _make_report(bias)
        result = report.plot_bias(mode="all", multi_feature=True, show=False)

        for mode_key, inner in result.items():
            assert inner is not None, f"{mode_key} is None"
            assert isinstance(inner, dict)
            for feat_key, fig in inner.items():
                assert fig is not None, f"{mode_key}/{feat_key} is None"
                assert isinstance(fig, plt.Figure)

    def test_single_mode_multi_true_missing_data_raises(self, multi_feature_bias_data):
        """If a single mode is requested with multi_feature=True but the
        relevant data is missing, ValueError is raised — same contract as
        the single-feature path."""
        # Only equalized_odds present -> 'subgroup' single mode must raise
        bias = {"equalized_odds": multi_feature_bias_data["equalized_odds"]}
        report = _make_report(bias)
        with pytest.raises(ValueError, match="subgroup_performance"):
            report.plot_bias(mode="subgroup", multi_feature=True, show=False)

    def test_no_bias_module_raises_with_multi_feature(self):
        """Bias module missing entirely -> ValueError regardless of multi_feature."""
        model = MagicMock()
        X = np.zeros((10, 2))
        y_true = np.array([0, 1] * 5)
        y_pred = np.array([0, 1] * 5)
        y_prob = np.random.rand(10, 2)
        report = TrustReport({}, model, X, y_true, y_pred, y_prob)
        with pytest.raises(ValueError, match="Bias results not available"):
            report.plot_bias(mode="all", multi_feature=True, show=False)


# ---------------------------------------------------------------------------
# 4. Deterministic ordering
# ---------------------------------------------------------------------------


class TestDeterministicOrdering:
    """Mode and feature ordering must be deterministic and independent of
    the original dict insertion order."""

    def test_all_multi_mode_keys_in_canonical_order(self, multi_feature_report):
        """Mode keys must appear in subgroup -> equalized_odds -> gap order."""
        result = multi_feature_report.plot_bias(mode="all", multi_feature=True, show=False)
        assert list(result.keys()) == ["subgroup", "equalized_odds", "gap"]

    def test_feature_keys_are_sorted(self, multi_feature_report):
        """Feature keys must be in ``sorted()`` order — alphabetical, not
        insertion order ('gender' was inserted before 'age')."""
        result = multi_feature_report.plot_bias(mode="all", multi_feature=True, show=False)
        for mode_key, inner in result.items():
            assert list(inner.keys()) == sorted(inner.keys()), (
                f"{mode_key} feature keys are not sorted: {list(inner.keys())}"
            )
            # Concretely: alphabetical -> 'age' before 'gender'
            assert list(inner.keys()) == ["age", "gender"]

    def test_single_mode_multi_feature_keys_are_sorted(self, multi_feature_report):
        """Single-mode multi_feature output must also use sorted feature order."""
        for mode in ("subgroup", "equalized_odds", "gap"):
            result = multi_feature_report.plot_bias(mode=mode, multi_feature=True, show=False)
            assert list(result.keys()) == ["age", "gender"], (
                f"mode={mode}: keys not in sorted order: {list(result.keys())}"
            )

    def test_repeated_calls_stable(self, multi_feature_report):
        """Repeated calls must produce identical key orderings."""
        r1 = multi_feature_report.plot_bias(mode="all", multi_feature=True, show=False)
        r2 = multi_feature_report.plot_bias(mode="all", multi_feature=True, show=False)
        assert list(r1.keys()) == list(r2.keys())
        for mode_key in r1:
            assert list(r1[mode_key].keys()) == list(r2[mode_key].keys())


# ---------------------------------------------------------------------------
# 5. Invalid mode + multi_feature=True
# ---------------------------------------------------------------------------


class TestInvalidModeWithMultiFeature:
    """``multi_feature=True`` must not bypass the ``mode`` validation."""

    def test_invalid_mode_with_multi_feature_true_raises(self, multi_feature_report):
        with pytest.raises(ValueError, match="Invalid mode"):
            multi_feature_report.plot_bias(mode="bogus", multi_feature=True, show=False)

    def test_invalid_mode_with_multi_feature_false_still_raises(self, multi_feature_report):
        """Sanity check: same error path is reachable with the default."""
        with pytest.raises(ValueError, match="Invalid mode"):
            multi_feature_report.plot_bias(mode="bogus", multi_feature=False, show=False)


# ---------------------------------------------------------------------------
# 6. Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """The default (``multi_feature=False``) must be unchanged."""

    def test_default_call_unchanged(self, multi_feature_report):
        """Default call (no mode, no multi_feature) must still work as before."""
        # Just ensure it does not raise; type can be Figure or dict depending on
        # how _plot_bias() summarises the data — this is the existing contract.
        result = multi_feature_report.plot_bias(show=False)
        assert result is not None

    def test_multi_feature_default_is_false(self, multi_feature_report):
        """Calling without ``multi_feature`` must equal calling with ``False``."""
        for mode in ("subgroup", "equalized_odds", "gap"):
            r_default = multi_feature_report.plot_bias(mode=mode, show=False)
            r_explicit = multi_feature_report.plot_bias(mode=mode, multi_feature=False, show=False)
            assert type(r_default) is type(r_explicit)
            assert isinstance(r_default, plt.Figure)
