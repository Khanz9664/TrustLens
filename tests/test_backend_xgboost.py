import numpy as np
import pytest

from trustlens import TrustReport, analyze
from trustlens.backends.registry import detect_framework, get_resolver

# We only run these tests if xgboost is available
xgboost = pytest.importorskip("xgboost")


def test_xgboost_detection():
    from xgboost import XGBClassifier

    model = XGBClassifier()
    assert detect_framework(model) == "xgboost"


def test_xgboost_resolver_basic():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=2, random_state=42)
    model.fit(X_train, y_train)

    resolver = get_resolver(model)
    bundle = resolver(model, X_test)

    assert bundle.framework == "xgboost"
    assert bundle.y_pred.shape == (20,)
    assert bundle.y_prob.shape == (20, 2)
    assert bundle.metadata["resolver"] == "xgboost"


def test_xgboost_integration_analyze():
    from sklearn.datasets import make_classification
    from xgboost import XGBClassifier

    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)

    model = XGBClassifier(n_estimators=2, random_state=42)
    model.fit(X, y)

    report = analyze(model, X, y, verbose=False)

    assert isinstance(report, TrustReport)
    assert report.metadata["framework"] == "xgboost"
    assert "xgboost" in report.metadata["backend"]["resolver"]


def test_xgboost_manual_override():
    from sklearn.datasets import make_classification
    from xgboost import XGBClassifier

    X, y = make_classification(n_samples=10, n_features=5, n_classes=2, random_state=42)
    model = XGBClassifier(n_estimators=2, random_state=42)
    model.fit(X, y)

    custom_preds = np.ones(10, dtype=int)
    report = analyze(model, X, y, y_pred=custom_preds, verbose=False)

    # TrustReport should use the manual override
    assert np.array_equal(report.y_pred, custom_preds)
    assert report.metadata["framework"] == "xgboost"


def test_xgboost_regressor_rejection():
    from sklearn.datasets import make_classification
    from xgboost import XGBRegressor

    X, y = make_classification(n_samples=10, n_features=5, n_classes=2, random_state=42)
    model = XGBRegressor(n_estimators=2, random_state=42)
    model.fit(X, y)

    with pytest.raises(NotImplementedError, match="supports classification models only"):
        analyze(model, X, y, verbose=False)
