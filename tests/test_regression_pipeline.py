"""Tests for the regression analysis path: task routing + TrustReport integration.

Covers analyze() task detection/routing, regression metric surfacing in the
report, show() rendering, to_dict() serialization, save() round-trips, and that
classification-only features are cleanly guarded on a regression report.
"""

from __future__ import annotations

import numpy as np
import pytest

from trustlens import analyze
from trustlens.report import TrustReport


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 3))
    y_true = X @ np.array([1.5, -2.0, 0.7]) + rng.normal(scale=1.0, size=n)
    y_pred = y_true + rng.normal(scale=1.0, size=n)
    return X, y_true, y_pred


def test_explicit_regression_routes_and_skips_trust_score(regression_data):
    X, y_true, y_pred = regression_data
    rep = analyze(model=None, X=X, y_true=y_true, y_pred=y_pred,
                  task="regression", verbose=False)
    assert isinstance(rep, TrustReport)
    assert rep.task_type == "regression"
    assert rep.trust_score is None  # classification score intentionally absent
    ed = rep.results["regression"]["error_distribution"]
    for key in ("mean_absolute_error", "rmse", "median_absolute_error",
                "p90_absolute_error", "max_error"):
        assert key in ed
    assert rep.metadata["task_type"] == "regression"
    assert "n_classes" not in rep.metadata
    assert rep.metadata["n_unique_targets"] == len(np.unique(y_true))


def test_auto_detection(regression_data):
    X, y_true, y_pred = regression_data
    # Continuous target -> regression.
    rep = analyze(model=None, X=X, y_true=y_true, y_pred=y_pred,
                  task="auto", verbose=False)
    assert rep.task_type == "regression"
    # Integer label target -> classification (not routed to regression).
    y_cls = (y_true > np.median(y_true)).astype(int)
    yprob = np.column_stack([1 - y_cls, y_cls]).astype(float)
    rep_cls = analyze(model=None, X=X, y_true=y_cls, y_pred=y_cls, y_prob=yprob,
                      task="auto", verbose=False)
    assert rep_cls.task_type == "classification"


def test_show_renders_regression(regression_data, capsys):
    X, y_true, y_pred = regression_data
    rep = analyze(model=None, X=X, y_true=y_true, y_pred=y_pred,
                  task="regression", verbose=False)
    rep.show()
    out = capsys.readouterr().out
    assert "Regression Reliability Report" in out
    assert "MAE" in out and "RMSE" in out
    assert "Task      : regression" in out


def test_to_dict_serializes_regression(regression_data):
    X, y_true, y_pred = regression_data
    rep = analyze(model=None, X=X, y_true=y_true, y_pred=y_pred,
                  task="regression", verbose=False)
    d = rep.to_dict()
    assert d["task_type"] == "regression"
    assert "regression.error_distribution.rmse" in d
    assert isinstance(d["regression.error_distribution.rmse"], float)
    # No classification trust score leaks into a regression dict.
    assert "trust_score" not in d
    assert "trust_grade" not in d


def test_uncertainty_metrics_populated_when_supplied(regression_data):
    X, y_true, y_pred = regression_data
    lo, hi = y_pred - 2.0, y_pred + 2.0
    variance = np.abs(y_true - y_pred)  # perfectly tracks error
    rep = analyze(model=None, X=X, y_true=y_true, y_pred=y_pred, task="regression",
                  prediction_intervals=(lo, hi), predicted_variance=variance,
                  confidence_level=0.95, verbose=False)
    reg = rep.results["regression"]
    picp = reg["interval_coverage"]
    assert "picp" in picp and 0.0 <= picp["picp"] <= 1.0
    assert picp["target_coverage"] == 0.95
    corr = reg["error_variance_correlation"]
    assert corr["verdict"] == "informative"  # variance == error => strong corr


def test_uncertainty_metrics_skip_gracefully(regression_data):
    X, y_true, y_pred = regression_data
    rep = analyze(model=None, X=X, y_true=y_true, y_pred=y_pred,
                  task="regression", verbose=False)
    reg = rep.results["regression"]
    assert reg["interval_coverage"]["status"] == "skipped"
    assert reg["error_variance_correlation"]["status"] == "skipped"


def test_model_predict_path():
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    n = 120
    X = rng.normal(size=(n, 4))
    y = X @ np.array([1.0, 0.5, -1.0, 2.0]) + rng.normal(scale=0.5, size=n)
    model = LinearRegression().fit(X, y)
    # No y_pred given -> resolved via model.predict.
    rep = analyze(model=model, X=X, y_true=y, task="regression", verbose=False)
    assert rep.task_type == "regression"
    assert rep.results["regression"]["error_distribution"]["n_samples"] == n


def test_classification_features_guarded_on_regression(regression_data):
    X, y_true, y_pred = regression_data
    rep = analyze(model=None, X=X, y_true=y_true, y_pred=y_pred,
                  task="regression", verbose=False)
    for call in (
        lambda: rep.summary_plot(show=False),
        lambda: rep.plot(),
        lambda: rep.show_failures(),
        lambda: rep.deployment_explanation,
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_save_roundtrip_regression(regression_data, tmp_path):
    import json

    X, y_true, y_pred = regression_data
    rep = analyze(model=None, X=X, y_true=y_true, y_pred=y_pred,
                  task="regression", verbose=False)
    # JSON
    jp = rep.save(str(tmp_path / "reg.json"))
    data = json.loads(jp.read_text(encoding="utf-8"))
    assert data["task_type"] == "regression"
    assert "regression" in data["results"]
    # TXT
    tp = rep.save(str(tmp_path / "reg.txt"))
    assert "Regression Reliability Report" in tp.read_text(encoding="utf-8")


def test_invalid_task_raises(regression_data):
    X, y_true, y_pred = regression_data
    with pytest.raises(ValueError):
        analyze(model=None, X=X, y_true=y_true, y_pred=y_pred,
                task="nonsense", verbose=False)
