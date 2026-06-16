import numpy as np
import pytest

from trustlens import TrustReport, analyze
from trustlens.backends.registry import detect_framework, get_resolver

catboost = pytest.importorskip("catboost")


def test_catboost_detection():
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(verbose=False)

    assert detect_framework(model) == "catboost"


def test_catboost_resolver_basic():
    from catboost import CatBoostClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=2,
        random_state=42,
    )

    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = CatBoostClassifier(
        iterations=10,
        verbose=False,
    )

    model.fit(X_train, y_train)

    resolver = get_resolver(model)
    bundle = resolver(model, X_test)

    assert bundle.framework == "catboost"
    assert bundle.y_pred.shape == (20,)
    assert bundle.y_prob.shape == (20, 2)
    assert bundle.metadata["resolver"] == "catboost"


def test_catboost_integration_analyze():
    from catboost import CatBoostClassifier
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=2,
        random_state=42,
    )

    model = CatBoostClassifier(
        iterations=10,
        verbose=False,
    )

    model.fit(X, y)

    report = analyze(model, X, y, verbose=False)

    assert isinstance(report, TrustReport)
    assert report.metadata["framework"] == "catboost"
    assert "catboost" in report.metadata["backend"]["resolver"]


def test_catboost_manual_override():
    from catboost import CatBoostClassifier
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=20,
        n_features=5,
        n_classes=2,
        random_state=42,
    )

    model = CatBoostClassifier(
        iterations=10,
        verbose=False,
    )

    model.fit(X, y)

    custom_preds = np.ones(len(y), dtype=int)

    report = analyze(
        model,
        X,
        y,
        y_pred=custom_preds,
        verbose=False,
    )

    assert np.array_equal(report.y_pred, custom_preds)
    assert report.metadata["framework"] == "catboost"


def test_catboost_regressor_rejection():
    from catboost import CatBoostRegressor
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_classes=2,
        random_state=42,
    )

    model = CatBoostRegressor(
        iterations=10,
        verbose=False,
    )

    model.fit(X, y)

    with pytest.raises(
        NotImplementedError,
        match="supports classification models only",
    ):
        analyze(model, X, y, verbose=False)
