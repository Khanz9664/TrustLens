"""
trustlens.core.pipeline
=======================
Internal execution engine for the TrustLens analysis pipeline.
This module is framework-agnostic and operates on standardized prediction data.

Responsibilities
----------------
* Iterate through selected analysis modules and execute their respective computations.
* Handle degraded execution states gracefully (e.g., when probabilities are missing).
* Aggregate individual module results into a unified dictionary.
* Instantiate the final `TrustReport` object.

Relationship to other components
--------------------------------
Invoked by `trustlens.api.analyze()`. It relies on standardized predictions
provided by the backend resolvers and delegates the actual metric computation
to domain-specific modules (`trustlens.metrics.*`).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, cast

import numpy as np

from trustlens.metrics.bias import (
    class_imbalance_report,
    equalized_odds,
    subgroup_performance,
)
from trustlens.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_curve,
)
from trustlens.metrics.conformal import conformal_diagnostics
from trustlens.metrics.failure import (
    confidence_gap,
    misclassification_summary,
)
from trustlens.metrics.regression import (
    error_distribution,
    error_variance_correlation,
    multilevel_interval_coverage,
    prediction_interval_coverage,
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
    y_pred_sets: Optional[Any] = None,
    nominal_coverage: float = 0.95,
    verbose: bool = True,
) -> TrustReport:
    """
    Internal orchestrator for analysis modules.

    WARNING: This function receives a model reference for plugin support and
    future XAI metrics, but it must NEVER call model methods directly (e.g. predict).
    All prediction data must be passed in via y_pred and y_prob.

    ``y_pred_sets`` (optional) carries conformal prediction sets; when supplied
    it activates the method-agnostic conformal diagnostics sub-block under
    ``calibration`` (RFC #157). ``nominal_coverage`` is the target ``1 - α`` the
    sets claim. Both are diagnostic-only — they never influence the Trust Score.
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
                    "mce": maximum_calibration_error(correct_mask, confidences),
                    "reliability_curve": reliability_curve(correct_mask, confidences),
                }
            else:
                # BINARY or 1D probabilities
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    y_prob_pos = y_prob[:, 1]
                else:
                    y_prob_pos = y_prob

                y_true_encoded = _encode_labels_for_probability_columns(y_true, 2, class_labels)

                results["calibration"] = {
                    "brier_score": brier_score(y_true_encoded, y_prob_pos),
                    "ece": expected_calibration_error(y_true_encoded, y_prob_pos),
                    "mce": maximum_calibration_error(y_true_encoded, y_prob_pos),
                    "reliability_curve": reliability_curve(y_true_encoded, y_prob_pos),
                }
        else:
            logger.warning("Skipped calibration: y_prob is missing.")
            results["calibration"] = {
                "status": "skipped",
                "reason": "missing_probabilities",
                "details": "Calibration requires probabilistic predictions.",
            }
            missing_components.append("calibration_metrics")

        # Conformal prediction diagnostics (RFC #157, PR 2/2).
        # Method-agnostic: evaluates the supplied prediction SETS, independent
        # of y_prob (they coexist — ECE/Brier from probabilities, conformal
        # coverage/informativeness from sets). Emitted iff y_pred_sets is
        # supplied; strictly diagnostic-only — never wired into the Trust Score.
        if y_pred_sets is not None:
            print("Running conformal diagnostics...")
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(module="conformal")
            calibration_block = results.setdefault("calibration", {})
            try:
                calibration_block["conformal"] = conformal_diagnostics(
                    y_true, y_pred_sets, nominal_coverage=nominal_coverage
                )
            except (ValueError, TypeError) as e:
                # Malformed sets (length mismatch, all-empty, ambiguous 0/1 list,
                # out-of-contract labels) degrade gracefully to a visible skipped
                # block rather than aborting the whole analysis — the reason stays
                # surfaced. Narrow to the input-validation errors conformal_diagnostics
                # / to_membership_matrix raise, so a genuine bug still propagates.
                logger.warning("Skipped conformal diagnostics: %s", e)
                calibration_block["conformal"] = {
                    "status": "skipped",
                    "reason": "invalid_prediction_sets",
                    "details": str(e)[:200],
                }
                missing_components.append("conformal_diagnostics")

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


def _run_regression_pipeline(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prediction_intervals: Optional[
        tuple[np.ndarray, np.ndarray] | dict[float, tuple[np.ndarray, np.ndarray]]
    ] = None,
    predicted_variance: Optional[np.ndarray] = None,
    confidence_level: float = 0.95,
    framework: Optional[str] = None,
    backend_metadata: Optional[dict[str, Any]] = None,
    verbose: bool = True,
) -> TrustReport:
    """Internal orchestrator for the regression analysis path.

    Mirrors :func:`_run_analysis_pipeline` (metrics -> results dict -> report)
    but routes through the regression reliability metrics. The uncertainty
    metrics (PICP, error-variance correlation) degrade gracefully to a
    ``status="skipped"`` dict when their optional inputs are absent.

    As with the classification path, this never calls ``model.predict``; the
    point predictions (and any intervals / variance) are passed in.
    """
    _log = logger.info if verbose else logger.debug

    def _as_single_output(name: str, values: np.ndarray) -> np.ndarray:
        """Coerce to 1-D: flatten a singleton ``(n, 1)`` column; reject true
        multi-output. A single-output model emitting ``(n, 1)`` predictions is
        valid and must not crash the metrics with a shape mismatch."""
        arr = np.asarray(values)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return cast(np.ndarray, arr[:, 0])
        if arr.ndim != 1:
            raise ValueError(
                f"{name} must be 1-D for single-output regression, got shape {arr.shape}."
            )
        return cast(np.ndarray, arr)

    y_true = _as_single_output("y_true", y_true)
    y_pred = _as_single_output("y_pred", y_pred)

    # Prediction intervals may be a single ``(lower, upper)`` tuple → single-level
    # PICP, or a mapping ``{level: (lower, upper)}`` → multi-level ICE + the
    # calibration-conditioned sharpness proxy (RFC #155). Both degrade gracefully
    # to a skipped dict when omitted. ``representative_intervals`` (the outermost
    # level for the multi-level case) feeds the legacy single-interval report
    # field / calibration plot unchanged.
    representative_intervals: tuple[np.ndarray, np.ndarray] | None = None
    if isinstance(prediction_intervals, dict):
        levels: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for level, bounds in prediction_intervals.items():
            lo, hi = bounds
            lo = _as_single_output(f"prediction_intervals[{level}][0]", lo)
            hi = _as_single_output(f"prediction_intervals[{level}][1]", hi)
            levels[float(level)] = (lo, hi)
        if levels:
            interval_coverage_result = multilevel_interval_coverage(y_true, levels)
            # Representative band for the legacy single-interval report field / plot:
            # the widest by mean width (robust when interval levels cross), not just
            # the highest nominal level.
            representative_intervals = max(
                levels.values(), key=lambda b: float(np.mean(b[1] - b[0]))
            )
        else:
            interval_coverage_result = prediction_interval_coverage(y_true, None, None)
    elif prediction_intervals is not None:
        lower, upper = prediction_intervals
        lower = _as_single_output("prediction_intervals[0]", lower)
        upper = _as_single_output("prediction_intervals[1]", upper)
        interval_coverage_result = prediction_interval_coverage(
            y_true, lower, upper, confidence_level=confidence_level
        )
        representative_intervals = (lower, upper)
    else:
        interval_coverage_result = prediction_interval_coverage(y_true, None, None)

    variance = predicted_variance
    if variance is not None:
        variance = _as_single_output("predicted_variance", variance)

    _log("Running regression reliability analysis...")
    regression_results: dict[str, Any] = {
        "error_distribution": error_distribution(y_true, y_pred),
        "interval_coverage": interval_coverage_result,
        "error_variance_correlation": error_variance_correlation(y_true, y_pred, variance),
        # Persist Var(y) (population variance, ddof=0 — identical to the on-the-fly
        # computation in regression_trust_score) so the regression Trust Score can
        # be recomputed from a stored report without the original y_true (issue #150).
        # Guard the empty case: np.var([]) is NaN, which is not JSON-serializable and
        # would corrupt a downstream recompute — fall back to 0.0 (the same value the
        # read path assigns an empty y_true, routing through the constant-target branch).
        "target_variance": (
            float(np.var(np.asarray(y_true, dtype=float))) if np.asarray(y_true).size else 0.0
        ),
    }

    results: dict[str, Any] = {"regression": regression_results}

    _log("Assembling regression report …")
    if backend_metadata is None:
        backend_metadata = {}

    report = TrustReport(
        results=results,
        model=model,
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=None,
        embeddings=None,
        framework=framework,
        backend_metadata=backend_metadata,
        task_type="regression",
        prediction_intervals=representative_intervals,
        predicted_variance=variance,
    )
    return report
