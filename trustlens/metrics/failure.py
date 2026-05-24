"""
trustlens.metrics.failure.
==========================
Failure-mode analysis: where and how does a model fail?

Metrics implemented
-------------------
* ``misclassification_summary`` — per-class error rates and high-confidence
  mistakes.
* ``confidence_gap``      — distribution of confidence for correct vs.
  incorrect predictions.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "misclassification_summary",
    "confidence_gap",
]

def misclassification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Build a comprehensive misclassification summary.

    For each class, reports:
    * total support (ground truth count)
    * number of misclassified samples
    * error rate
    * average confidence of misclassified samples (overconfident mistakes)
    * indices of the *most confident* misclassifications

    Parameters
    ----------
    y_true : np.ndarray
      Ground-truth labels, shape (n_samples,).
    y_pred : np.ndarray
      Model predictions, shape (n_samples,).
    y_prob : np.ndarray
      Predicted probabilities, shape (n_samples,) for binary or
      (n_samples, n_classes) for multi-class.

    Returns
    -------
    dict
      Nested dictionary keyed by class label.

    Examples
    --------
    >>> summary = misclassification_summary(y_true, y_pred, y_prob)
    >>> print(summary[1]["error_rate"]) # error rate for class 1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    # Max probability across classes for each sample
    if y_prob.ndim == 1:
        max_conf = y_prob  # binary: confidence in positive class
    else:
        max_conf = y_prob.max(axis=1)

    incorrect_mask = y_true != y_pred
    classes = np.unique(y_true)

    summary: dict = {}
    for cls in classes:
        cls_mask = y_true == int(cls)
        cls_incorrect = cls_mask & incorrect_mask

        n_support = int(cls_mask.sum())
        n_misclassified = int(cls_incorrect.sum())
        error_rate = n_misclassified / n_support if n_support > 0 else 0.0

        miscls_confidences = max_conf[cls_incorrect]
        avg_misclassification_confidence = (
            float(miscls_confidences.mean()) if len(miscls_confidences) > 0 else 0.0
        )

        # Indices of top-5 most confident mistakes (high-confidence errors)
        if len(miscls_confidences) > 0:
            topk = min(5, len(miscls_confidences))
            local_indices = np.argsort(miscls_confidences)[-topk:][::-1]
            global_indices = np.where(cls_incorrect)[0]
            top_mistake_indices = global_indices[local_indices].tolist()
        else:
            top_mistake_indices = []

        summary[int(cls)] = {
            "support": n_support,
            "n_misclassified": n_misclassified,
            "error_rate": round(error_rate, 4),
            "avg_misclassification_confidence": round(avg_misclassification_confidence, 4),
            "top_mistake_indices": top_mistake_indices,
        }

    summary["__overall__"] = {
        "total_errors": int(incorrect_mask.sum()),
        "overall_error_rate": round(float(incorrect_mask.mean()), 4),
    }

    return summary


def confidence_gap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 20,
) -> dict:
    """
    Measure the *confidence gap* — how much more confident is the model
    on correct predictions than on incorrect ones?

    Returns
    -------
    dict with keys:
      * ``correct_confidence``  — confidence distribution for correct preds
      * ``incorrect_confidence`` — confidence distribution for incorrect preds
      * ``gap``         — mean(correct_conf) - mean(incorrect_conf)
      * ``histogram_bins``    — bin edges for the confidence histogram
      * ``correct_hist``     — histogram counts for correct predictions
      * ``incorrect_hist``    — histogram counts for incorrect predictions

    Examples
    --------
    >>> gap_data = confidence_gap(y_true, y_pred, y_prob)
    >>> print(f"Confidence gap: {gap_data['gap']:.3f}")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    if y_prob.ndim == 1:
        max_conf = y_prob
    else:
        max_conf = y_prob.max(axis=1)

    correct_mask = y_true == y_pred
    correct_conf = max_conf[correct_mask]
    incorrect_conf = max_conf[~correct_mask]

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    correct_hist, _ = np.histogram(correct_conf, bins=bins)
    incorrect_hist, _ = np.histogram(incorrect_conf, bins=bins)

    gap = float(correct_conf.mean() - incorrect_conf.mean()) if len(incorrect_conf) > 0 else 0.0

    return {
        "correct_confidence_mean": float(correct_conf.mean()) if len(correct_conf) > 0 else 0.0,
        "incorrect_confidence_mean": float(incorrect_conf.mean())
        if len(incorrect_conf) > 0
        else 0.0,
        "gap": round(gap, 4),
        "histogram_bins": bins,
        "correct_hist": correct_hist,
        "incorrect_hist": incorrect_hist,
        "n_correct": int(correct_mask.sum()),
        "n_incorrect": int((~correct_mask).sum()),
    }
