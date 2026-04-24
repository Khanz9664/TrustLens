"""
trustlens.api.
==============
Primary entry point for the TrustLens analysis pipeline.

Usage
-----
>>> from trustlens import analyze
>>> report = analyze(model, X_val, y_val, y_prob)
>>> report.show()
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from trustlens.metrics.bias import (
    class_imbalance_report,
    subgroup_performance,
)
from trustlens.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from trustlens.metrics.failure import (
    confidence_gap,
    misclassification_summary,
)
from trustlens.metrics.representation import (
    embedding_separability,
)
from trustlens.plugins.registry import PluginRegistry
from trustlens.report import TrustReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline Module Registry
#
# STABLE (allowed in the pipeline):
#   calibration, failure, bias, representation
#
# EXPERIMENTAL (do NOT import or wire into this file):
#   explainability (gradcam, faithfulness) — requires PyTorch
#   metrics.faithfulness — wrapper over explainability
#
# Before adding a new module to the pipeline:
#   1. Confirm it meets promotion criteria in docs/EXPERIMENTAL.md
#   2. Ensure it has no heavy required dependencies (use try/except)
#   3. Add it to _ALL_MODULES below and write a dispatch block
#   4. Get maintainer approval via PR review
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def quick_analyze(model=None, X=None, y=None, dataset="iris", framework="sklearn") -> TrustReport:
    """
    Zero-friction entry point for TrustLens.
    If no model/data provided, auto-loads a basic dataset to demonstrate output.
    """
    if model is None or X is None or y is None:
        logger.info(f"No model/data provided. Auto-loading {dataset} dataset for demo...")
        if dataset == "iris":
            from sklearn.datasets import load_iris
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split

            data = load_iris()
            X_all, y_all = data.data, data.target
            # Make it binary for simpler demo
            X_all, y_all = X_all[y_all != 2], y_all[y_all != 2]
            X_train, X, y_train, y = train_test_split(X_all, y_all, test_size=0.3, random_state=42)

            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
        elif dataset == "breast_cancer":
            from sklearn.datasets import load_breast_cancer
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split

            data = load_breast_cancer()
            X_all, y_all = data.data, data.target
            X_train, X, y_train, y = train_test_split(X_all, y_all, test_size=0.3, random_state=42)

            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
        else:
            raise ValueError("Supported demo datasets: 'iris', 'breast_cancer'")

    print(f"\nTrustLens Analysis: {dataset}")
    print(f"Status: Loading demo model and {dataset} validation data...")

    report = analyze(model=model, X=X, y_true=y, verbose=False)

    report.show()
    report.summary_plot()
    return report


def analyze(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray | None = None,
    *,
    embeddings: np.ndarray | None = None,
    sensitive_features: dict[str, np.ndarray] | None = None,
    modules: list[str] | None = None,
    plugins: list[str] | None = None,
    verbose: bool = True,
) -> TrustReport:
    """
    Run a full TrustLens analysis on a trained model.

    Parameters
    ----------
    model : Any
      Trained sklearn-compatible model (must expose ``predict`` and,
      optionally, ``predict_proba``).
    X : np.ndarray
      Validation feature matrix, shape (n_samples, n_features).
    y_true : np.ndarray
      Ground-truth labels, shape (n_samples,).
    y_prob : np.ndarray, optional
      Predicted class probabilities, shape (n_samples, n_classes).
      If None, TrustLens will call ``model.predict_proba(X)`` if available.
    embeddings : np.ndarray, optional
      Latent representations / embeddings for representation analysis,
      shape (n_samples, embedding_dim).
    sensitive_features : dict, optional
      Mapping of feature name → 1-D array for bias/subgroup analysis.
      Example: ``{"gender": gender_array, "age_group": age_array}``
    modules : list[str], optional
      Subset of analysis modules to run. Defaults to all available:
      ``["calibration", "failure", "bias", "representation"]``.
    plugins : list[str], optional
      Names of registered plugins to activate (see Plugin Registry).
    verbose : bool
      Print progress updates. Default True.

    Returns
    -------
    TrustReport
      Populated report object with metrics, plots, and narrative summaries.

    Examples
    --------
    >>> from trustlens import analyze
    >>> report = analyze(clf, X_val, y_val, y_prob=proba)
    >>> report.show()         # interactive view
    >>> report.save("trust_report/") # persist to disk
    """
    _log = logger.info if verbose else logger.debug

    if len(y_true) < 30:
        print("Warning: Small dataset (n < 30) detected. Calibration metrics may be unreliable.")

    # ------------------------------------------------------------------
    # 1. Resolve probability predictions
    # ------------------------------------------------------------------
    if y_prob is None:
        if hasattr(model, "predict_proba"):
            _log("Calling model.predict_proba() …")
            y_prob = model.predict_proba(X)
        else:
            raise ValueError("y_prob is required when model does not expose predict_proba().")

    y_pred = model.predict(X)

    # ------------------------------------------------------------------
    # 2. Determine which modules to run
    # ------------------------------------------------------------------
    _ALL_MODULES = ["calibration", "failure", "bias", "representation"]
    active_modules = modules or _ALL_MODULES

    results: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Progress Tracking
    # ------------------------------------------------------------------
    try:
        from tqdm import tqdm

        pbar = tqdm(active_modules, desc="Analysing Model", unit="module", leave=False)
    except ImportError:
        pbar = active_modules

    # ------------------------------------------------------------------
    # 3. Calibration module
    # ------------------------------------------------------------------
    if "calibration" in active_modules:
        print("Running calibration analysis...")
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(module="calibration")
        # For binary classification use positive-class probabilities.
        # For multi-class, compute one-vs-rest brier score (macro average).
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = y_prob  # kept as-is; metrics handle multi-class

        results["calibration"] = {
            "brier_score": brier_score(y_true, y_prob_pos),
            "ece": expected_calibration_error(y_true, y_prob_pos),
            "reliability_curve": reliability_curve(y_true, y_prob_pos),
        }

    # ------------------------------------------------------------------
    # 4. Failure analysis module
    # ------------------------------------------------------------------
    if "failure" in active_modules:
        print("Running failure analysis...")
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(module="failure")
        results["failure"] = {
            "misclassification_summary": misclassification_summary(y_true, y_pred, y_prob),
            "confidence_gap": confidence_gap(y_true, y_pred, y_prob),
        }

    # ------------------------------------------------------------------
    # 5. Bias detection module
    # ------------------------------------------------------------------
    if "bias" in active_modules:
        print("Running bias analysis...")
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(module="bias")
        results["bias"] = {
            "class_imbalance": class_imbalance_report(y_true),
        }
        if sensitive_features:
            results["bias"]["subgroup_performance"] = subgroup_performance(
                y_true, y_pred, sensitive_features
            )

    # ------------------------------------------------------------------
    # 6. Representation analysis module
    # ------------------------------------------------------------------
    if "representation" in active_modules and embeddings is not None:
        print("Running representation analysis...")
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(module="representation")
        results["representation"] = {
            "separability": embedding_separability(embeddings, y_true),
        }

    # ------------------------------------------------------------------
    # 7. Activate plugins
    # ------------------------------------------------------------------
    if plugins:
        registry = PluginRegistry()
        for plugin_name in plugins:
            _log(f"Activating plugin: {plugin_name}")
            plugin = registry.get(plugin_name)
            results[f"plugin_{plugin_name}"] = plugin.run(
                model=model,
                X=X,
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
            )

    # ------------------------------------------------------------------
    # 8. Build and return TrustReport
    # ------------------------------------------------------------------
    _log("Assembling report …")
    report = TrustReport(
        results=results,
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )
    return report
