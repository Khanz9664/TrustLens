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
from typing import Any, Optional

import numpy as np

from trustlens.backends.registry import get_resolver
from trustlens.core.pipeline import _run_analysis_pipeline
from trustlens.report import TrustReport

logger = logging.getLogger(__name__)


def quick_analyze(
    model=None, X=None, y=None, dataset="iris", framework: Optional[str] = None
) -> TrustReport:
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

    report = analyze(model=model, X=X, y_true=y, framework=framework, verbose=False)

    report.show()
    report.summary_plot()
    return report


def analyze(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
    *,
    framework: Optional[str] = None,
    embeddings: Optional[np.ndarray] = None,
    sensitive_features: Optional[dict[str, np.ndarray]] = None,
    modules: Optional[list[str]] = None,
    plugins: Optional[list[str]] = None,
    verbose: bool = True,
) -> TrustReport:
    """
    Run a full TrustLens analysis on a trained model.

    Parameters
    ----------
    model : Any, optional
      Trained machine learning model. Can be None if ``y_pred`` or ``y_prob`` are provided manually.
    X : np.ndarray
      Validation feature matrix, shape (n_samples, n_features).
    y_true : np.ndarray
      Ground-truth labels, shape (n_samples,).
    y_pred : np.ndarray, optional
      Predicted class labels, shape (n_samples,).
      If None, TrustLens will automatically resolve predictions via the backend system.
    y_prob : np.ndarray, optional
      Predicted class probabilities, shape (n_samples, n_classes).
      If None, TrustLens will automatically resolve probabilities via the backend system.
    framework : str, optional
      Explicitly specify the model framework (e.g., 'sklearn', 'xgboost').
      If None, TrustLens will attempt to auto-detect the framework.
    embeddings : np.ndarray, optional
      Latent representations / embeddings for representation analysis,
      shape (n_samples, embedding_dim).
    sensitive_features : dict, optional
      Mapping of feature name → 1-D array for bias/subgroup analysis.
    modules : list[str], optional
      Subset of analysis modules to run.
    plugins : list[str], optional
      Names of registered plugins to activate.
    verbose : bool
      Print progress updates. Default True.

    Returns
    -------
    TrustReport
      Populated report object with metrics, plots, and narrative summaries.
    """
    if len(y_true) < 30:
        logger.warning("Small dataset (n < 30) detected. Calibration metrics may be unreliable.")

    # ------------------------------------------------------------------
    # 1. Resolve predictions via Backend Registry
    # Short-circuit if both overrides are provided
    if y_pred is not None and y_prob is not None:
        framework = "manual"

    resolver = get_resolver(model, framework=framework)
    bundle = resolver(model, X, y_pred=y_pred, y_prob=y_prob)

    # ------------------------------------------------------------------
    # 2. Delegate to Core Pipeline
    # ------------------------------------------------------------------
    return _run_analysis_pipeline(
        model=model,
        X=X,
        y_true=y_true,
        y_pred=bundle.y_pred,
        y_prob=bundle.y_prob,
        framework=bundle.framework,
        backend_metadata=bundle.metadata,
        embeddings=embeddings,
        sensitive_features=sensitive_features,
        modules=modules,
        plugins=plugins,
        verbose=verbose,
    )
