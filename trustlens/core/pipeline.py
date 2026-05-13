"""
trustlens.core.pipeline
=======================
Internal execution engine for the TrustLens analysis pipeline.
This module is framework-agnostic and operates on standardized prediction data.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from trustlens.metrics.bias import (
    class_imbalance_report,
    equalized_odds,
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


def _run_analysis_pipeline(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    embeddings: Optional[np.ndarray] = None,
    sensitive_features: Optional[dict[str, np.ndarray]] = None,
    modules: Optional[list[str]] = None,
    plugins: Optional[list[str]] = None,
    framework: Optional[str] = None,
    backend_metadata: Optional[dict[str, Any]] = None,
    verbose: bool = True,
) -> TrustReport:
    """
    Internal orchestrator for analysis modules.

    WARNING: This function receives a model reference for plugin support and
    future XAI metrics, but it must NEVER call model methods directly (e.g. predict).
    All prediction data must be passed in via y_pred and y_prob.
    """
    _log = logger.info if verbose else logger.debug

    # ------------------------------------------------------------------
    # 1. Determine which modules to run
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
    # 2. Calibration module
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
    # 3. Failure analysis module
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
    # 4. Bias detection module
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
            # Equalized odds requires a binary target (0, 1) and features with >1 subgroup
            is_binary = set(np.unique(y_true)).issubset({0, 1})
            meaningful_features = {
                k: v for k, v in sensitive_features.items() if len(np.unique(v)) > 1
            }

            if is_binary and meaningful_features:
                try:
                    results["bias"]["equalized_odds"] = equalized_odds(
                        y_true, y_pred, meaningful_features
                    )
                except Exception as e:
                    logger.warning("Skipped equalized_odds computation: %s", e)
                    results["bias"]["equalized_odds"] = {
                        "status": "skipped",
                        "reason": "computation_error",
                        "details": str(e)[:200],
                    }
            else:
                results["bias"]["equalized_odds"] = {
                    "status": "skipped",
                    "reason": "invalid_input",
                    "details": "requires binary target and multi-group sensitive features",
                }

    # ------------------------------------------------------------------
    # 5. Representation analysis module
    # ------------------------------------------------------------------
    if "representation" in active_modules and embeddings is not None:
        print("Running representation analysis...")
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(module="representation")
        results["representation"] = {
            "separability": embedding_separability(embeddings, y_true),
        }

    # ------------------------------------------------------------------
    # 6. Activate plugins
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
    # 7. Build and return TrustReport
    # ------------------------------------------------------------------
    _log("Assembling report …")
    report = TrustReport(
        results=results,
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        embeddings=embeddings,
        framework=framework,
        backend_metadata=backend_metadata,
    )
    return report
