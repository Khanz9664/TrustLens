import numpy as np
import pytest
import xgboost as xgb

from trustlens.backends.types import PredictionBundle
from trustlens.backends.xgboost import resolve


def test_xgboost_multiclass_classifier():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 3, 100)

    model = xgb.XGBClassifier(objective="multi:softprob", n_estimators=2, random_state=42)
    model.fit(X, y)

    bundle = resolve(model, X)

    assert isinstance(bundle, PredictionBundle)
    assert bundle.y_pred.shape == (100,)
    assert bundle.y_prob.shape == (100, 3)
    assert bundle.framework == "xgboost"
    assert np.allclose(bundle.y_prob.sum(axis=1), 1.0)
    print("XGBClassifier Multiclass: PASSED")


def test_xgboost_binary_booster():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    dtrain = xgb.DMatrix(X, label=y)
    param = {"objective": "binary:logistic", "verbosity": 0}
    bst = xgb.train(param, dtrain, num_boost_round=2)

    bundle = resolve(bst, X)

    assert isinstance(bundle, PredictionBundle)
    assert bundle.y_pred.shape == (100,)
    assert bundle.y_prob.shape == (100, 2)
    assert bundle.framework == "xgboost"
    assert np.allclose(bundle.y_prob.sum(axis=1), 1.0)
    print("Booster Binary: PASSED")


def test_xgboost_multiclass_booster():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 3, 100)

    dtrain = xgb.DMatrix(X, label=y)
    param = {"objective": "multi:softprob", "num_class": 3, "verbosity": 0}
    bst = xgb.train(param, dtrain, num_boost_round=2)

    bundle = resolve(bst, X)

    assert isinstance(bundle, PredictionBundle)
    assert bundle.y_pred.shape == (100,)
    assert bundle.y_prob.shape == (100, 3)
    assert bundle.framework == "xgboost"
    assert np.allclose(bundle.y_prob.sum(axis=1), 1.0)
    print("Booster Multiclass: PASSED")


def test_xgboost_regression_blocking():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=2)
    model.fit(X, y)

    with pytest.raises(NotImplementedError) as excinfo:
        resolve(model, X)
    assert "classification models only" in str(excinfo.value)
    print("XGBRegressor Blocking: PASSED")


def test_xgboost_ranking_blocking():
    X = np.random.rand(100, 5)
    y = [1, 0] * 50
    group = [50, 50]

    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(group)
    param = {"objective": "rank:pairwise", "verbosity": 0}
    bst = xgb.train(param, dtrain, num_boost_round=2)

    # We need to see how resolve() handles Booster objective
    try:
        resolve(bst, X)
        print("FAILED: Booster Rank:pairwise should be blocked")
    except NotImplementedError as e:
        assert "classification models only" in str(e)
        print("Booster Ranking Blocking: PASSED")
