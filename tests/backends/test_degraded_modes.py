import numpy as np
import pytest

from trustlens import analyze


def test_model_none_pred_only():
    """Test model=None with only y_pred provided. Calibration should be skipped."""
    X = np.random.rand(100, 5)
    y = np.zeros(100)
    y_pred = np.zeros(100)

    report = analyze(None, X, y, y_pred=y_pred, verbose=False)

    assert report.model is None
    assert report.metadata["model_class"] == "Manual"
    assert report.metadata["backend"]["resolver"] == "manual_override"
    assert report.metadata["backend"]["degraded_mode"] is True
    assert "probabilities" in report.metadata["backend"]["missing_components"]
    assert report.results["calibration"]["status"] == "skipped"
    print("Model=None Pred Only: PASSED")


def test_model_none_prob_only():
    """Test model=None with only y_prob provided. y_pred should be derived."""
    X = np.random.rand(100, 5)
    y = np.zeros(100)
    y_prob = np.zeros((100, 2))
    y_prob[:, 0] = 1.0  # All class 0

    report = analyze(None, X, y, y_prob=y_prob, verbose=False)

    assert report.y_pred is not None
    assert np.all(report.y_pred == 0)
    assert report.metadata["backend"]["resolver"] == "manual_override"
    # Should NOT be in degraded mode if y_prob is provided and y_pred is derived
    assert "degraded_mode" not in report.metadata["backend"]
    print("Model=None Prob Only: PASSED")


def test_model_none_no_overrides():
    """Test model=None with no overrides. Should raise ValueError."""
    X = np.random.rand(100, 5)
    y = np.zeros(100)

    with pytest.raises(ValueError) as excinfo:
        analyze(None, X, y, verbose=False)
    assert "Manual override requires either y_pred or y_prob" in str(excinfo.value)
    print("Model=None No Overrides: PASSED")
