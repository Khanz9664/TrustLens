"""
trustlens.api.
==============
Primary entry point for the TrustLens analysis pipeline.

Responsibilities
----------------
* Expose the core `analyze()` and `quick_analyze()` functions.
* Coordinate the translation of user inputs into the internal format via backends.
* Delegate execution to the core analysis pipeline.

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
from trustlens.core.pipeline import _run_analysis_pipeline, _run_regression_pipeline
from trustlens.report import TrustReport

logger = logging.getLogger(__name__)


def _detect_task(y_true: np.ndarray, task: str) -> str:
    """Resolve the analysis task type.

    ``task`` may be ``"classification"`` / ``"regression"`` (explicit, honored
    as-is) or ``"auto"``. Auto-detection errs toward ``"classification"`` and
    only returns ``"regression"`` when the target is clearly continuous — a
    float array that is not integer-valued, or has many distinct values — so a
    discrete label set is never mis-routed.
    """
    if task in ("classification", "regression"):
        return task
    if task != "auto":
        raise ValueError(f"Invalid task {task!r}. Use 'auto', 'classification', or 'regression'.")

    y = np.asarray(y_true)
    if y.dtype.kind == "f":
        n_unique = len(np.unique(y))
        is_integer_valued = bool(np.all(np.isfinite(y))) and bool(np.allclose(y, np.round(y)))
        # Integer-valued floats are class labels at ANY cardinality (a 25-class
        # target encoded as float must not be mistaken for regression), and a
        # small distinct-value set is also label-like. Only clearly-continuous
        # floats route to regression.
        if is_integer_valued or n_unique <= 20:
            return "classification"
        return "regression"
    # Non-float dtypes (ints, strings, bools) default to classification.
    return "classification"


def quick_analyze(
    model=None, X=None, y=None, dataset="iris", framework: Optional[str] = None
) -> TrustReport:
    """
    Zero-friction entry point for TrustLens.
    If no model/data provided, auto-loads a basic dataset to demonstrate output.

    Parameters
    ----------
    model : Any, optional
        A trained machine learning model. If None, a demo model is trained.
    X : np.ndarray, optional
        Validation feature matrix.
    y : np.ndarray, optional
        Ground-truth labels.
    dataset : str, default='iris'
        The demo dataset to load if data is not provided ('iris' or 'breast_cancer').
    framework : str, optional
        Explicitly specify the model framework (e.g., 'sklearn').

    Returns
    -------
    TrustReport
        Populated report object with metrics, plots, and narrative summaries.
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
    class_labels: Optional[np.ndarray] = None,
    task: str = "auto",
    prediction_intervals: Optional[
        tuple[np.ndarray, np.ndarray] | dict[float, tuple[np.ndarray, np.ndarray]]
    ] = None,
    predicted_variance: Optional[np.ndarray] = None,
    y_pred_sets: Optional[Any] = None,
    confidence_level: float = 0.95,
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
      Explicitly specify the model framework
      (e.g., ``'sklearn'``, ``'xgboost'``, ``'lightgbm'``, ``'catboost'``).
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
    class_labels : np.ndarray, optional
      Semantic class labels in the order corresponding to probability columns.
      Useful for raw backends such as ``xgboost.Booster`` that return ordinal
      probability columns without a ``classes_`` attribute.
    task : str, default='auto'
      Analysis task: ``'auto'`` (detect from ``y_true``), ``'classification'``,
      or ``'regression'``. Regression routes through the regression reliability
      metrics (error distribution, interval coverage, error-variance
      correlation) instead of the classification modules.
    prediction_intervals : tuple or dict, optional
      Per-sample prediction-interval bounds (regression only). Either a single
      ``(lower, upper)`` tuple — enabling single-level Prediction Interval
      Coverage (PICP) — or a mapping ``{level: (lower, upper)}`` of nominal
      coverage levels, enabling multi-level Interval Calibration Error (ICE) and
      the calibration-conditioned sharpness proxy (RFC #155). Omitted ⇒ those
      metrics are skipped.
    predicted_variance : np.ndarray, optional
      Per-sample predicted variance / uncertainty score (regression only).
      Enables the error-variance correlation metric; omitted ⇒ skipped.
    y_pred_sets : np.ndarray or sequence, optional
      Conformal **prediction sets** (classification only). Either an
      ``(n_samples, n_classes)`` 0/1 membership matrix or a ragged list of
      per-sample label lists (converted internally). When supplied, activates
      the method-agnostic conformal diagnostics sub-block under ``calibration``
      (marginal / class-conditional / size-stratified coverage + set-size
      informativeness, RFC #157). Coexists with ``y_prob`` — ECE/Brier come from
      probabilities, conformal coverage from the sets. Omitted ⇒ the block is
      simply not emitted. Diagnostic-only: it never influences the Trust Score.
    confidence_level : float, default=0.95
      Nominal coverage ``1 - α`` the supplied intervals/sets claim. Used by the
      single-tuple ``prediction_intervals`` (regression) and by ``y_pred_sets``
      (classification, as the conformal nominal target for the coverage gaps).
      Ignored when ``prediction_intervals`` is a ``{level: ...}`` mapping, where
      each level is its own nominal coverage.
    verbose : bool
      Print progress updates. Default True.

    Returns
    -------
    TrustReport
      Populated report object with metrics, plots, and narrative summaries.

    Examples
    --------
    End-to-end analysis with a RandomForest classifier:

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.model_selection import train_test_split
    >>> from trustlens import analyze
    >>>
    >>> # Create a synthetic dataset
    >>> X, y = make_classification(
    ...     n_samples=500, n_features=10, random_state=42
    ... )
    >>>
    >>> # Train / test split
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=42
    ... )
    >>>
    >>> # Train a classifier
    >>> model = RandomForestClassifier(random_state=42)
    >>> model.fit(X_train, y_train)
    >>>
    >>> # Predict probabilities
    >>> y_prob = model.predict_proba(X_test)
    >>>
    >>> # Run TrustLens analysis
    >>> report = analyze(model, X_test, y_test, y_prob=y_prob)
    >>>
    >>> # Display results
    >>> report.show()
    """
    if len(y_true) < 30:
        logger.warning("Small dataset (n < 30) detected. Metrics may be unreliable.")

    # ------------------------------------------------------------------
    # 0. Route by task. Regression skips the classification backend (which
    #    resolves class probabilities) and the classification modules.
    # ------------------------------------------------------------------
    task_type = _detect_task(y_true, task)
    if task_type == "regression":
        if y_pred is None:
            if model is None or not hasattr(model, "predict"):
                raise ValueError(
                    "Regression analysis needs point predictions: pass y_pred=..., "
                    "or a model that exposes .predict(X)."
                )
            y_pred_resolved = np.asarray(model.predict(X))
        else:
            y_pred_resolved = np.asarray(y_pred)
        return _run_regression_pipeline(
            model=model,
            X=X,
            y_true=np.asarray(y_true),
            y_pred=y_pred_resolved,
            prediction_intervals=prediction_intervals,
            predicted_variance=predicted_variance,
            confidence_level=confidence_level,
            framework=framework or ("manual" if y_pred is not None else None),
            backend_metadata={"task_type": "regression"},
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # 1. Resolve predictions via Backend Registry
    # Short-circuit if both overrides are provided
    if y_pred is not None and y_prob is not None:
        framework = "manual"

    resolver = get_resolver(model, framework=framework)
    resolved_class_labels = np.asarray(class_labels) if class_labels is not None else None
    bundle = resolver(
        model,
        X,
        y_pred=y_pred,
        y_prob=y_prob,
        class_labels=resolved_class_labels,
    )

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
        class_labels=bundle.class_labels,
        embeddings=embeddings,
        sensitive_features=sensitive_features,
        modules=modules,
        plugins=plugins,
        y_pred_sets=y_pred_sets,
        nominal_coverage=confidence_level,
        verbose=verbose,
    )
