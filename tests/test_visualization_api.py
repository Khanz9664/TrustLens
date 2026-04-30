import os
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from trustlens.report import TrustReport


@pytest.fixture
def dummy_report():
    results = {
        "bias": {
            "subgroup_performance": {
                "gender": {
                    "male": {"accuracy": 0.8},
                    "female": {"accuracy": 0.7},
                    "__summary__": {"performance_gap": 0.1},
                }
            },
            "equalized_odds": {
                "gender": {
                    "male": {"tpr": 0.8, "fpr": 0.1},
                    "female": {"tpr": 0.7, "fpr": 0.2},
                    "__summary__": {"tpr_gap": 0.1, "fpr_gap": 0.1},
                }
            },
            "class_imbalance": {"class_counts": {0: 5, 1: 5}, "imbalance_ratio": 1.0},
        }
    }
    model = MagicMock()
    X = np.zeros((10, 2))
    y_true = np.array([0, 1] * 5)
    y_pred = np.array([0, 1] * 5)
    y_prob = np.random.rand(10, 2)
    return TrustReport(results, model, X, y_true, y_pred, y_prob)


def test_plot_bias_default(dummy_report):
    """Test 1: Default Behavior (Backward Compatibility)"""
    fig = dummy_report.plot_bias()
    assert isinstance(fig, plt.Figure)


def test_plot_bias_modes(dummy_report):
    """Test 2: Each Mode"""
    for mode in ["subgroup", "equalized_odds", "gap"]:
        fig = dummy_report.plot_bias(mode=mode)
        assert isinstance(fig, plt.Figure)


def test_plot_bias_all(dummy_report):
    """Test 3: mode='all'"""
    result = dummy_report.plot_bias(mode="all")
    assert isinstance(result, dict)
    assert set(result.keys()) == {"subgroup", "equalized_odds", "gap"}
    for fig in result.values():
        assert isinstance(fig, plt.Figure)


def test_plot_bias_invalid_mode(dummy_report):
    """Test 4: Invalid Mode"""
    with pytest.raises(ValueError, match="Invalid mode"):
        dummy_report.plot_bias(mode="invalid")


def test_plot_bias_no_data():
    """Test 5: No Bias Data"""
    results = {}
    model = MagicMock()
    X = np.zeros((10, 2))
    y_true = np.array([0, 1] * 5)
    y_pred = np.array([0, 1] * 5)
    y_prob = np.random.rand(10, 2)
    report = TrustReport(results, model, X, y_true, y_pred, y_prob)
    with pytest.raises(ValueError, match="Bias results not available"):
        report.plot_bias()


def test_plot_bias_save_path(dummy_report, tmp_path):
    """Test 6: Save Path for individual mode"""
    save_path = tmp_path / "test_subgroup.png"
    dummy_report.plot_bias(mode="subgroup", save_path=str(save_path))
    assert save_path.exists()


def test_plot_bias_all_save_path(dummy_report, tmp_path):
    """Test 7: Multi-mode Save"""
    save_base = str(tmp_path / "test_plot")
    dummy_report.plot_bias(mode="all", save_path=save_base)
    assert os.path.exists(save_base + "_subgroup.png")
    assert os.path.exists(save_base + "_equalized_odds.png")
    assert os.path.exists(save_base + "_gap.png")


def test_plot_bias_show_false(dummy_report):
    """Test 8: show=False behavior"""
    # Just ensure it doesn't crash
    fig = dummy_report.plot_bias(mode="subgroup", show=False)
    assert fig is not None


def test_plot_bias_all_show_false(dummy_report):
    """Test 9: mode='all' with show=False"""
    result = dummy_report.plot_bias(mode="all", show=False)
    assert isinstance(result, dict)


def test_plot_bias_extension_handling(dummy_report, tmp_path):
    """Test 10: extension handling"""
    save_path = str(tmp_path / "test.png")
    dummy_report.plot_bias(mode="all", save_path=save_path)
    # Ensure no test.png_subgroup.png
    assert os.path.exists(str(tmp_path / "test_subgroup.png"))
    assert not os.path.exists(str(tmp_path / "test.png_subgroup.png"))


def test_plot_bias_deterministic_order(dummy_report):
    """Test 11: Deterministic keys and order in mode='all'"""
    result = dummy_report.plot_bias(mode="all")
    assert list(result.keys()) == ["subgroup", "equalized_odds", "gap"]


def test_plot_bias_summary_fallback_safety():
    """Test 12: Summary fallback safety (raise ValueError if _plot_bias returns None)"""
    # Create bias data that is "usable" for the validation check but _plot_bias returns None
    results = {"bias": {"subgroup_performance": {"gender": {"m": {"acc": 0.8}}}}}
    model = MagicMock()
    X = np.zeros((10, 2))
    y_true = np.array([0, 1] * 5)
    y_pred = np.array([0, 1] * 5)
    y_prob = np.random.rand(10, 2)
    report = TrustReport(results, model, X, y_true, y_pred, y_prob)

    with pytest.raises(ValueError, match="Failed to generate summary bias plot"):
        report.plot_bias(mode="summary")


def test_plot_bias_partial_data_all(dummy_report):
    """Test 13: Partial data in mode='all'"""
    # Remove subgroup_performance
    report = dummy_report
    del report.results["bias"]["subgroup_performance"]
    result = report.plot_bias(mode="all")
    assert set(result.keys()) == {"subgroup", "equalized_odds", "gap"}
    assert result["subgroup"] is None
    assert result["equalized_odds"] is not None
    # gap should still work if equalized_odds is present
    assert result["gap"] is not None


def test_plot_bias_single_mode_missing_data(dummy_report):
    """Test 14: Single mode missing data raises ValueError"""
    # Remove subgroup_performance
    del dummy_report.results["bias"]["subgroup_performance"]
    with pytest.raises(ValueError, match="Missing 'subgroup_performance' data"):
        dummy_report.plot_bias(mode="subgroup")


def test_plot_bias_no_mutation(dummy_report):
    """Test 15: Verify no mutation of report results"""
    import copy

    before = copy.deepcopy(dummy_report.results)
    dummy_report.plot_bias(mode="all")
    assert dummy_report.results == before


def test_plot_bias_mixed_availability(dummy_report):
    """Test 16: Verify mixed availability in mode='all'"""
    # One is None, others are not
    del dummy_report.results["bias"]["subgroup_performance"]
    result = dummy_report.plot_bias(mode="all")
    assert result["subgroup"] is None
    assert any(v is not None for v in result.values())


def test_plot_bias_all_fail_raises_error():
    """Test 17: All plots fail in mode='all' ValueError"""
    # Data that passes the initial validation but fails all specific plotting
    results = {"bias": {"class_imbalance": {"class_counts": {0: 5, 1: 5}, "imbalance_ratio": 1.0}}}
    model = MagicMock()
    X = np.zeros((10, 2))
    y_true = np.array([0, 1] * 5)
    y_pred = np.array([0, 1] * 5)
    y_prob = np.random.rand(10, 2)
    report = TrustReport(results, model, X, y_true, y_pred, y_prob)

    with pytest.raises(ValueError, match="Failed to generate any bias plots"):
        report.plot_bias(mode="all")


def test_plot_bias_repeated_calls_stable(dummy_report):
    """Test 18: Stability of repeated calls"""
    result1 = dummy_report.plot_bias(mode="all")
    result2 = dummy_report.plot_bias(mode="all")
    assert list(result1.keys()) == list(result2.keys())
    assert all(k1 == k2 for k1, k2 in zip(result1.keys(), result2.keys()))
