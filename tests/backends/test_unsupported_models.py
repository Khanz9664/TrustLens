import numpy as np
import pytest

from trustlens import analyze


class BrokenEstimator:
    _estimator_type = "classifier"

    def predict(self, X):
        return np.zeros(len(X))

    # Missing predict_proba


def test_broken_estimator():
    X = np.random.rand(100, 5)
    y = np.zeros(100)
    model = BrokenEstimator()

    # This should fail because predict_proba is missing and y_prob is not provided
    with pytest.raises(ValueError) as excinfo:
        analyze(model, X, y, verbose=False)
    assert "y_prob is required" in str(excinfo.value)
    print("Broken Estimator: PASSED")


def test_malformed_y_pred():
    X = np.random.rand(100, 5)
    y = np.zeros(100)
    y_pred = np.zeros((100, 2))  # Malformed (2D)

    with pytest.raises(ValueError) as excinfo:
        analyze(None, X, y, y_pred=y_pred, y_prob=np.zeros((100, 2)), verbose=False)
    assert "y_pred must be a 1D array" in str(excinfo.value)
    print("Malformed y_pred: PASSED")
