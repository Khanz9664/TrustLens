import numpy as np
import pytest

from trustlens import TrustReport, analyze
from trustlens.backends.registry import detect_framework, get_resolver

lightgbm = pytest.importorskip("lightgbm")


def test_lightgbm_detection():
    from lightgbm import LGBMClassifier

    model = LGBMClassifier()
    assert detect_framework(model) == "lightgbm"


def test_lightgbm_resolver_basic():
    from lightgbm import LGBMClassifier
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

    model = LGBMClassifier(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)

    resolver = get_resolver(model)
    bundle = resolver(model, X_test)

    assert bundle.framework == "lightgbm"
    assert bundle.y_pred.shape == (20,)
    assert bundle.y_prob.shape == (20, 2)
    assert bundle.metadata["resolver"] == "lightgbm"


def test_lightgbm_integration_analyze():
    from lightgbm import LGBMClassifier
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=2,
        random_state=42,
    )

    model = LGBMClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    report = analyze(model, X, y, verbose=False)

    assert isinstance(report, TrustReport)
    assert report.metadata["framework"] == "lightgbm"
    assert "lightgbm" in report.metadata["backend"]["resolver"]


def test_lightgbm_manual_override():
    from lightgbm import LGBMClassifier
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=20,
        n_features=5,
        n_classes=2,
        random_state=42,
    )

    model = LGBMClassifier(n_estimators=5, random_state=42)
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
    assert report.metadata["framework"] == "lightgbm"


def test_lightgbm_regressor_rejection():
    from lightgbm import LGBMRegressor
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_classes=2,
        random_state=42,
    )

    model = LGBMRegressor(n_estimators=5, random_state=42)
    model.fit(X, y)

    with pytest.raises(
        NotImplementedError,
        match="supports classification models only",
    ):
        analyze(model, X, y, verbose=False)
