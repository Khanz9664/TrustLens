"""
tests/test_conformal_integration.py.
=====================================
End-to-end integration tests for conformal prediction diagnostics through the
public ``analyze()`` API (RFC #157, PR 2/2). These cover the pipeline wiring,
the ``results["calibration"]["conformal"]`` block, the report panel, coexistence
with probability-based calibration, graceful activation/degradation, and the
guarantee that the diagnostics never move the Trust Score.

The metric-level maths is covered exhaustively in ``test_conformal.py``; here we
only assert the *integration* behaviour.
"""

import io
from contextlib import redirect_stdout

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from trustlens import TrustReport, analyze


@pytest.fixture(scope="module")
def trained_multiclass_clf():
    """Train a 3-class classifier and build LAC-style prediction sets from probs."""
    X, y = make_classification(
        n_samples=400,
        n_features=12,
        n_informative=6,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=25, random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_val)
    # Threshold-based prediction sets: include every class whose probability
    # clears a fixed cut. Produces a well-formed (n, K) 0/1 membership matrix
    # with a mix of singleton and multi-class sets.
    pred_sets = (y_prob >= 0.30).astype(int)
    return clf, X_val, y_val, y_prob, pred_sets


class TestConformalActivation:
    def test_block_present_when_sets_supplied(self, trained_multiclass_clf):
        clf, X, y, prob, sets = trained_multiclass_clf
        report = analyze(
            clf, X, y, y_prob=prob, y_pred_sets=sets, confidence_level=0.9, verbose=False
        )
        conf = report.results["calibration"]["conformal"]
        # Every documented field present …
        for key in (
            "nominal",
            "marginal_coverage",
            "coverage_gap",
            "worst_class_coverage",
            "worst_class_gap",
            "ssc_violation",
            "avg_set_size",
            "singleton_rate",
            "size_efficiency",
            "empty_rate",
            "n_classes",
            "n_samples",
        ):
            assert key in conf
        # … and internally consistent.
        assert 0.0 <= conf["marginal_coverage"] <= 1.0
        assert conf["nominal"] == pytest.approx(0.9)
        assert conf["coverage_gap"] == pytest.approx(conf["marginal_coverage"] - 0.9)
        assert conf["n_samples"] == len(y)
        assert conf["n_classes"] == 3

    def test_absent_when_no_sets(self, trained_multiclass_clf):
        clf, X, y, prob, _sets = trained_multiclass_clf
        report = analyze(clf, X, y, y_prob=prob, verbose=False)
        assert "conformal" not in report.results.get("calibration", {})

    def test_coexists_with_probability_calibration(self, trained_multiclass_clf):
        clf, X, y, prob, sets = trained_multiclass_clf
        report = analyze(clf, X, y, y_prob=prob, y_pred_sets=sets, verbose=False)
        cal = report.results["calibration"]
        # Probability-based metrics still computed alongside the set-based ones.
        assert "ece" in cal
        assert "mce" in cal
        assert "conformal" in cal
        assert cal["conformal"]["marginal_coverage"] is not None

    def test_nominal_defaults_to_confidence_level(self, trained_multiclass_clf):
        clf, X, y, prob, sets = trained_multiclass_clf
        report = analyze(
            clf, X, y, y_prob=prob, y_pred_sets=sets, confidence_level=0.8, verbose=False
        )
        assert report.results["calibration"]["conformal"]["nominal"] == pytest.approx(0.8)

    def test_list_of_lists_sets_accepted(self, trained_multiclass_clf):
        clf, X, y, prob, matrix_sets = trained_multiclass_clf
        # Convert the membership matrix to ragged label-lists — the other public
        # input form — and confirm the pipeline normalises it identically.
        label_sets = [list(np.flatnonzero(row)) for row in matrix_sets]
        report = analyze(
            clf, X, y, y_prob=prob, y_pred_sets=label_sets, confidence_level=0.9, verbose=False
        )
        conf = report.results["calibration"]["conformal"]
        assert conf["n_samples"] == len(y)
        assert 0.0 <= conf["marginal_coverage"] <= 1.0


class TestConformalReportPanel:
    def test_panel_rendered_with_sets(self, trained_multiclass_clf):
        clf, X, y, prob, sets = trained_multiclass_clf
        report = analyze(
            clf, X, y, y_prob=prob, y_pred_sets=sets, confidence_level=0.9, verbose=False
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            report.show()
        out = buf.getvalue()
        assert "Conformal Prediction (prediction sets)" in out
        assert "Marginal coverage" in out
        assert "Worst-class coverage" in out
        assert "SSC violation" in out

    def test_panel_absent_without_sets(self, trained_multiclass_clf):
        clf, X, y, prob, _sets = trained_multiclass_clf
        report = analyze(clf, X, y, y_prob=prob, verbose=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            report.show()
        assert "Conformal Prediction" not in buf.getvalue()

    def test_panel_labels_over_and_under(self, trained_multiclass_clf):
        clf, X, y, prob, sets = trained_multiclass_clf
        report = analyze(
            clf, X, y, y_prob=prob, y_pred_sets=sets, confidence_level=0.9, verbose=False
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            report.show()
        out = buf.getvalue()
        # The explicit over/under annotation must appear so the two opposite-sign
        # gap fields can't be misread.
        assert ("over by" in out) or ("under by" in out) or ("on target" in out)


class TestConformalDegradation:
    def test_malformed_sets_degrade_gracefully(self, trained_multiclass_clf):
        clf, X, y, prob, sets = trained_multiclass_clf
        # Length mismatch: fewer sets than samples. Must not abort analyze().
        bad_sets = sets[:-5]
        report = analyze(
            clf, X, y, y_prob=prob, y_pred_sets=bad_sets, confidence_level=0.9, verbose=False
        )
        conf = report.results["calibration"]["conformal"]
        assert conf.get("status") == "skipped"
        assert conf.get("reason") == "invalid_prediction_sets"
        # And the rest of the report is intact.
        assert "ece" in report.results["calibration"]

    def test_conformal_without_probabilities(self, trained_multiclass_clf):
        clf, X, y, _prob, sets = trained_multiclass_clf
        # Sets present but probabilities withheld: calibration metrics skip, but
        # the conformal block is still emitted (it doesn't need y_prob).
        y_pred = clf.predict(X)
        report = analyze(
            clf, X, y, y_pred=y_pred, y_pred_sets=sets, confidence_level=0.9, verbose=False
        )
        cal = report.results["calibration"]
        assert "conformal" in cal
        assert cal["conformal"]["marginal_coverage"] is not None


class TestConformalDoesNotMoveScore:
    def test_trust_score_unchanged_by_conformal(self, trained_multiclass_clf):
        clf, X, y, prob, sets = trained_multiclass_clf
        base = analyze(clf, X, y, y_prob=prob, verbose=False)
        with_sets = analyze(clf, X, y, y_prob=prob, y_pred_sets=sets, verbose=False)
        # Phase-2 scoring is deferred: conformal diagnostics are report-only, so
        # the Trust Score and every sub-score must be identical.
        assert isinstance(with_sets, TrustReport)
        assert with_sets.trust_score.score == base.trust_score.score
        assert with_sets.trust_score.sub_scores == base.trust_score.sub_scores
