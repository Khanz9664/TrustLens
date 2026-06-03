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


def test_raw_xgboost_booster_maps_ordinal_predictions_to_string_labels():
    rng = np.random.default_rng(42)
    class_labels = np.array(["mouse", "cat", "dog"])
    y_ordinal = np.repeat(np.arange(3), 30)
    X = np.column_stack(
        [
            y_ordinal,
            y_ordinal == 1,
            y_ordinal == 2,
            rng.normal(scale=0.01, size=len(y_ordinal)),
        ]
    ).astype(float)
    y_true = class_labels[y_ordinal]

    dtrain = xgboost.DMatrix(X, label=y_ordinal)
    model = xgboost.train(
        {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 2,
            "eta": 0.5,
            "verbosity": 0,
        },
        dtrain,
        num_boost_round=10,
    )

    report = analyze(model, X, y_true, class_labels=class_labels, verbose=False)

    assert report.metadata["framework"] == "xgboost"
    assert report.y_prob.shape == (90, 3)
    assert set(np.unique(report.y_pred)).issubset(set(class_labels))
    assert report.results["failure"]["misclassification_summary"]["__overall__"][
        "total_errors"
    ] == int(np.sum(y_true != report.y_pred))


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
