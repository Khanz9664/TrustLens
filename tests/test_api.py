"""
tests/test_api.py.
==================
Integration tests for the top-level ``analyze()`` API.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from trustlens import TrustReport, analyze


@pytest.fixture(scope="module")
def trained_binary_clf():
    """Train a simple binary classifier for integration tests."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_classes=2,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_val)
    return clf, X_val, y_val, y_prob


class TestAnalyzeAPI:
    def test_returns_trust_report(self, trained_binary_clf):
        clf, X, y, prob = trained_binary_clf
        report = analyze(clf, X, y, y_prob=prob, verbose=False)
        assert isinstance(report, TrustReport)

    def test_calibration_module_in_results(self, trained_binary_clf):
        clf, X, y, prob = trained_binary_clf
        report = analyze(clf, X, y, y_prob=prob, verbose=False)
        assert "calibration" in report.results

    def test_failure_module_in_results(self, trained_binary_clf):
        clf, X, y, prob = trained_binary_clf
        report = analyze(clf, X, y, y_prob=prob, verbose=False)
        assert "failure" in report.results

    def test_bias_module_in_results(self, trained_binary_clf):
        clf, X, y, prob = trained_binary_clf
        report = analyze(clf, X, y, y_prob=prob, verbose=False)
        assert "bias" in report.results

    def test_module_filtering(self, trained_binary_clf):
        clf, X, y, prob = trained_binary_clf
        report = analyze(
            clf,
            X,
            y,
            y_prob=prob,
            modules=["calibration"],
            verbose=False,
        )
        assert "calibration" in report.results
        assert "failure" not in report.results

    def test_no_y_prob_uses_predict_proba(self, trained_binary_clf):
        clf, X, y, _ = trained_binary_clf
        # Pass no y_prob — should auto-call predict_proba
        report = analyze(clf, X, y, verbose=False)
        assert isinstance(report, TrustReport)

    def test_no_predict_proba_without_y_prob_raises(self, trained_binary_clf):
        """Model without predict_proba and no y_prob must raise ValueError."""
        from sklearn.base import BaseEstimator, ClassifierMixin

        class NoProbaClf(BaseEstimator, ClassifierMixin):
            def fit(self, X, y):
                self.classes_ = np.unique(y)
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        clf, X, y, _ = trained_binary_clf
        bad_clf = NoProbaClf().fit(X, y)
        with pytest.raises(ValueError, match="y_prob is required"):
            analyze(bad_clf, X, y, verbose=False)

    def test_brier_score_in_calibration(self, trained_binary_clf):
        clf, X, y, prob = trained_binary_clf
        report = analyze(clf, X, y, y_prob=prob, verbose=False)
        bs = report.results["calibration"]["brier_score"]
        assert 0.0 <= bs <= 1.0

    def test_representation_module_with_embeddings(self, trained_binary_clf):
        clf, X, y, prob = trained_binary_clf
        embeddings = np.random.randn(len(y), 16)
        report = analyze(
            clf,
            X,
            y,
            y_prob=prob,
            embeddings=embeddings,
            verbose=False,
        )
        assert "representation" in report.results

    def test_representation_skipped_without_embeddings(self, trained_binary_clf):
        clf, X, y, prob = trained_binary_clf
        report = analyze(clf, X, y, y_prob=prob, verbose=False)
        assert "representation" not in report.results

    def test_sensitive_features_triggers_subgroup(self, trained_binary_clf):
        clf, X, y, prob = trained_binary_clf
        gender = np.array(["M" if i % 2 == 0 else "F" for i in range(len(y))])
        report = analyze(
            clf,
            X,
            y,
            y_prob=prob,
            sensitive_features={"gender": gender},
            verbose=False,
        )
        assert "subgroup_performance" in report.results["bias"]


class TestTrustReportInterface:
    def test_show_does_not_raise_and_covers_verbose(self, trained_binary_clf, capsys):
        clf, X, y, prob = trained_binary_clf
        report = analyze(clf, X, y, y_prob=prob, verbose=False)

        # Test default mode
        report.show(verbose=False)
        captured = capsys.readouterr()
        assert "TrustLens Analysis Report" in captured.out
        assert "Conclusion:" in captured.out

        # Test verbose mode to cover extra branching
        report.show(verbose=True)
        captured_verbose = capsys.readouterr()
        assert "Conclusion:" in captured_verbose.out

        # Test show_failures to cover massive reporting branched logic
        report.show_failures()
        captured_failures = capsys.readouterr()
        assert "CRITICAL FAILURES" in captured_failures.out

    def test_save_creates_json(self, trained_binary_clf, tmp_path):
        clf, X, y, prob = trained_binary_clf
        report = analyze(clf, X, y, y_prob=prob, verbose=False)
        out_dir = report.save(str(tmp_path / "report"))
        assert (out_dir / "report.json").exists()
        assert (out_dir / "metadata.json").exists()

    def test_metadata_contains_model_class(self, trained_binary_clf):
        clf, X, y, prob = trained_binary_clf
        report = analyze(clf, X, y, y_prob=prob, verbose=False)
        assert report.metadata["model_class"] == "RandomForestClassifier"

    def test_repr(self, trained_binary_clf):
        clf, X, y, prob = trained_binary_clf
        report = analyze(clf, X, y, y_prob=prob, verbose=False)
        assert "TrustReport" in repr(report)
