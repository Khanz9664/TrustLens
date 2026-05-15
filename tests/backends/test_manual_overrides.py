import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from trustlens import analyze


def test_manual_override_both():
    X, y = make_classification(n_samples=100, n_classes=2, random_state=42)
    model = RandomForestClassifier().fit(X, y)

    y_pred = np.zeros(100)
    y_prob = np.zeros((100, 2))
    y_prob[:, 0] = 1.0

    report = analyze(model, X, y, y_pred=y_pred, y_prob=y_prob, verbose=False)

    # Check that overrides were respected
    assert np.array_equal(report.y_pred, y_pred)
    assert np.array_equal(report.y_prob, y_prob)
    assert report.metadata["backend"]["resolver"] == "manual_override"
    print("Override Both: PASSED")


def test_manual_override_pred_only():
    X, y = make_classification(n_samples=100, n_classes=2, random_state=42)
    model = RandomForestClassifier().fit(X, y)

    y_pred = np.ones(100)

    report = analyze(model, X, y, y_pred=y_pred, verbose=False)

    # Check that y_pred was respected, but y_prob was still resolved from model
    assert np.array_equal(report.y_pred, y_pred)
    assert report.y_prob is not None
    # Precedence: if y_pred is provided, resolver should still be called for y_prob
    # Wait, how does analyze() handle this?
    print("Override Pred Only: PASSED")


def test_manual_override_prob_only():
    X, y = make_classification(n_samples=100, n_classes=2, random_state=42)
    model = RandomForestClassifier().fit(X, y)

    y_prob = np.zeros((100, 2))
    y_prob[:, 1] = 1.0

    report = analyze(model, X, y, y_prob=y_prob, verbose=False)

    # Check that y_prob was respected, and y_pred was derived from y_prob
    assert np.array_equal(report.y_prob, y_prob)
    assert np.all(report.y_pred == 1)
    print("Override Prob Only: PASSED")
