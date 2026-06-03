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


def _as_python_label(label: Any) -> Any:
    """Return a hashable Python scalar for NumPy scalar labels."""
    return label.item() if hasattr(label, "item") else label


def _encode_labels_for_probability_columns(
    y_true: np.ndarray,
    n_classes: int,
    class_labels: Optional[np.ndarray],
) -> np.ndarray:
    """Encode semantic labels into the class-index order used by y_prob columns."""
    if class_labels is not None:
        class_labels_array = np.asarray(class_labels)
        if len(class_labels_array) != n_classes:
            raise ValueError(
                "class_labels length "
                f"({len(class_labels_array)}) does not match probability column shape "
                f"({n_classes} columns)."
            )

        label_to_index = {
            _as_python_label(label): idx for idx, label in enumerate(class_labels_array)
        }
        try:
            encoded_labels: np.ndarray = np.asarray(
                [_as_python_label(label_to_index[_as_python_label(label)]) for label in y_true],
                dtype=int,
            )
            return encoded_labels
        except KeyError as exc:
            raise ValueError("y_true contains labels that are missing from class_labels.") from exc

    return y_true.astype(int)


def _run_analysis_pipeline(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    embeddings: Optional[np.ndarray] = None,
    sensitive_features: Optional[dict[str, np.ndarray]] = None,
    modules: Optional[list[str]] = None,
    plugins: Optional[list[str]] = None,
    framework: Optional[str] = None,
    backend_metadata: Optional[dict[str, Any]] = None,
    class_labels: Optional[np.ndarray] = None,
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
    missing_components: list[str] = []

    if y_prob is None:
        missing_components.append("probabilities")

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
        if y_prob is not None:
            print("Running calibration analysis...")
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(module="calibration")

            # Calibration logic based on task type
            if y_prob.ndim == 2 and y_prob.shape[1] > 2:
                # MULTICLASS: Top-label calibration (ECE) and Multiclass Brier Score
                n_classes = y_prob.shape[1]
                confidences = np.max(y_prob, axis=1)
                correct_mask = (y_true == y_pred).astype(float)

                # Multiclass Brier Score: 1/N * sum(sum((p_ic - o_ic)^2))
                # We can compute this efficiently
                y_true_indices = _encode_labels_for_probability_columns(
                    y_true, n_classes, class_labels
                )
                y_true_one_hot = np.eye(n_classes)[y_true_indices]
                mbrier = np.mean(np.sum((y_prob - y_true_one_hot) ** 2, axis=1))

                results["calibration"] = {
                    "brier_score": float(mbrier),
                    "ece": expected_calibration_error(correct_mask, confidences),
                    "reliability_curve": reliability_curve(correct_mask, confidences),
                }
            else:
                # BINARY or 1D probabilities
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    y_prob_pos = y_prob[:, 1]
                else:
                    y_prob_pos = y_prob

                results["calibration"] = {
                    "brier_score": brier_score(y_true, y_prob_pos),
                    "ece": expected_calibration_error(y_true, y_prob_pos),
                    "reliability_curve": reliability_curve(y_true, y_prob_pos),
                }
        else:
            logger.warning("Skipped calibration: y_prob is missing.")
            results["calibration"] = {
                "status": "skipped",
                "reason": "missing_probabilities",
                "details": "Calibration requires probabilistic predictions.",
            }
            missing_components.append("calibration_metrics")

    # ------------------------------------------------------------------
    # 3. Failure analysis module
    # ------------------------------------------------------------------
    if "failure" in active_modules:
        if y_prob is not None:
            print("Running failure analysis...")
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(module="failure")
            results["failure"] = {
                "misclassification_summary": misclassification_summary(y_true, y_pred, y_prob),
                "confidence_gap": confidence_gap(y_true, y_pred, y_prob),
            }
        else:
            logger.warning(
                "Degraded failure analysis: y_prob is missing. Confidence metrics skipped."
            )
            # Provide a minimal summary that doesn't need probabilities
            incorrect_mask = y_true != y_pred
            results["failure"] = {
                "status": "degraded",
                "reason": "missing_probabilities",
                "misclassification_summary": {
                    "__overall__": {
                        "total_errors": int(incorrect_mask.sum()),
                        "overall_error_rate": round(float(incorrect_mask.mean()), 4),
                    }
                },
                "confidence_gap": {"gap": 0.0, "status": "skipped"},
            }
            missing_components.append("failure_confidence_metrics")

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

    # Enrich metadata with degraded state information
    if backend_metadata is None:
        backend_metadata = {}

    if missing_components:
        backend_metadata["degraded_mode"] = True
        backend_metadata["missing_components"] = missing_components

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
