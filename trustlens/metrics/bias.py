"""
trustlens.metrics.bias.
=======================
Bias and fairness detection.

Bias in ML manifests as systematically worse performance for certain
subgroups (demographic, geographic, temporal, etc.). TrustLens surfaces
these disparities without making causal claims — the responsibility to
act lies with the practitioner.

Metrics implemented
-------------------
* ``class_imbalance_report`` — distribution statistics for label classes.
* ``subgroup_performance``  — per-subgroup accuracy/F1 breakdown.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

__all__ = [
    "class_imbalance_report",
    "subgroup_performance",
    "equalized_odds",
]

def class_imbalance_report(y_true: np.ndarray) -> dict:
    """
    Summarize the class distribution in ``y_true``.

    Reports absolute counts, relative frequencies, and an imbalance
    ratio (majority class count / minority class count).

    Parameters
    ----------
    y_true : np.ndarray
      Ground-truth labels.

    Returns
    -------
    dict with keys:
      * ``class_counts``   — dict mapping class → sample count
      * ``class_frequencies`` — dict mapping class → relative frequency
      * ``imbalance_ratio`` — max_count / min_count (1.0 = perfectly balanced)
      * ``minority_class``  — class with fewest samples
      * ``majority_class``  — class with most samples

    Examples
    --------
    >>> report = class_imbalance_report(y_true)
    >>> print(f"Imbalance ratio: {report['imbalance_ratio']:.2f}x")
    """
    y_true = np.asarray(y_true)
    classes, counts = np.unique(y_true, return_counts=True)
    n = len(y_true)

    class_counts = {int(cls): int(cnt) for cls, cnt in zip(classes, counts)}
    class_frequencies = {int(cls): round(float(cnt / n), 4) for cls, cnt in zip(classes, counts)}

    min_count = int(counts.min())
    max_count = int(counts.max())
    minority_class = int(classes[counts.argmin()])
    majority_class = int(classes[counts.argmax()])
    imbalance_ratio = round(max_count / min_count, 4) if min_count > 0 else float("inf")

    return {
        "class_counts": class_counts,
        "class_frequencies": class_frequencies,
        "imbalance_ratio": imbalance_ratio,
        "minority_class": minority_class,
        "majority_class": majority_class,
        "n_classes": int(len(classes)),
    }


def subgroup_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: dict[str, np.ndarray],
    metrics: Optional[list[str]] = None,
) -> dict:
    """
    Compute model performance broken down by sensitive subgroups.

    For each feature in ``sensitive_features``, TrustLens computes
    per-group accuracy and macro-F1 scores, then derives the
    *performance gap* between best and worst performing groups.

    Parameters
    ----------
    y_true : np.ndarray
      Ground-truth labels.
    y_pred : np.ndarray
      Model predictions.
    sensitive_features : dict
      Mapping of feature name → 1-D array of group labels.
      Example: ``{"gender": gender_array}``.
    metrics : list[str], optional
      Which metrics to compute. Supports ``"accuracy"`` and ``"f1"``.
      Default: ``["accuracy", "f1"]``.

    Returns
    -------
    dict
      Nested dict: feature → group → metric values + summary.

    Examples
    --------
    >>> results = subgroup_performance(
    ...   y_true, y_pred,
    ...   sensitive_features={"gender": gender_array},
    ... )
    >>> print(results["gender"]["performance_gap"])
    """
    if metrics is None:
        metrics = ["accuracy", "f1"]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    report: dict = {}

    for feature_name, group_array in sensitive_features.items():
        group_array = np.asarray(group_array)
        groups = np.unique(group_array)
        group_results: dict = {}

        for g in groups:
            mask = group_array == g
            y_true_g = y_true[mask]
            y_pred_g = y_pred[mask]

            group_metrics: dict = {"n_samples": int(mask.sum())}

            if "accuracy" in metrics:
                group_metrics["accuracy"] = round(float(accuracy_score(y_true_g, y_pred_g)), 4)
            if "f1" in metrics:
                group_metrics["f1"] = round(
                    float(
                        f1_score(
                            y_true_g,
                            y_pred_g,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    4,
                )

            group_results[str(g)] = group_metrics

        # Compute performance gap (accuracy-based)
        if "accuracy" in metrics and len(group_results) >= 2:
            accuracies = [v["accuracy"] for v in group_results.values()]
            gap = round(max(accuracies) - min(accuracies), 4)
            best_group = max(group_results, key=lambda g: group_results[g].get("accuracy", 0))
            worst_group = min(group_results, key=lambda g: group_results[g].get("accuracy", 0))
            group_results["__summary__"] = {
                "performance_gap": gap,
                "best_group": best_group,
                "worst_group": worst_group,
            }

        report[feature_name] = group_results

    return report


def equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: dict[str, np.ndarray],
    severe_threshold: float = 0.15,
    moderate_threshold: float = 0.05,
) -> dict:
    """
    Compute Equalized Odds fairness metrics broken down by sensitive subgroups.

    Equalized Odds requires that TPR (True Positive Rate) and FPR (False
    Positive Rate) are equal across all subgroups. Large gaps indicate that
    the model treats certain groups systematically differently.

    Reference: Hardt et al., "Equality of Opportunity in Supervised Learning",
    NeurIPS 2016.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0 or 1).
    y_pred : np.ndarray
        Model predictions (binary, 0 or 1).
    sensitive_features : dict
        Mapping of feature name → 1-D array of group labels.
        Example: ``{"gender": gender_array}``.
    severe_threshold : float, optional
        Gap above which a violation is classified as ``"severe"``.
        Default: ``0.15``.
    moderate_threshold : float, optional
        Gap above which a violation is classified as ``"moderate"``.
        Must be less than ``severe_threshold``. Default: ``0.05``.

    Returns
    -------
    dict
        Nested dict: feature → group → metric values + ``__summary__``.

        Per-group keys:
          * ``n_samples``  — number of samples in the group
          * ``tpr``        — True Positive Rate (recall)
          * ``fpr``        — False Positive Rate (FP / (FP + TN))

        Summary keys (under ``__summary__``):
          * ``tpr_gap``          — max(tpr) - min(tpr) across groups
          * ``fpr_gap``          — max(fpr) - min(fpr) across groups
          * ``tpr_violation``    — severity of TPR gap
          * ``fpr_violation``    — severity of FPR gap
          * ``best_tpr_group``   — group with highest TPR
          * ``worst_tpr_group``  — group with lowest TPR

        Violation levels:
          * ``"severe"``     — gap > severe_threshold (default 0.15)
          * ``"moderate"``   — gap between moderate_threshold and severe_threshold
          * ``"acceptable"`` — gap < moderate_threshold (default 0.05)

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_pred`` have different lengths.
    ValueError
        If any ``sensitive_features`` array has a different length than ``y_true``.
    ValueError
        If ``moderate_threshold`` >= ``severe_threshold``.
    ValueError
        If ``y_true`` or ``y_pred`` is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from trustlens.metrics.bias import equalized_odds
    >>>
    >>> y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    >>> y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    >>> gender  = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    >>>
    >>> results = equalized_odds(y_true, y_pred, {"gender": gender})
    >>> results["gender"]["0"]
    {'n_samples': 4, 'tpr': 0.5, 'fpr': 0.0}
    >>> results["gender"]["1"]
    {'n_samples': 4, 'tpr': 1.0, 'fpr': 0.5}
    >>> results["gender"]["__summary__"]
    {'tpr_gap': 0.5, 'fpr_gap': 0.5, 'tpr_violation': 'severe', 'fpr_violation': 'severe', 'best_tpr_group': '1', 'worst_tpr_group': '0'}
    """
    # --- Input validation ---
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length, got {len(y_true)} and {len(y_pred)}."
        )

    if not (0.0 < moderate_threshold < severe_threshold < 1.0):
        raise ValueError(
            f"moderate_threshold must be less than severe_threshold, "
            f"got moderate_threshold={moderate_threshold}, severe_threshold={severe_threshold}."
        )

    if not sensitive_features:
        raise ValueError("sensitive_features must not be empty.")

    for feature_name, group_array in sensitive_features.items():
        group_array = np.asarray(group_array)
        if len(group_array) != len(y_true):
            raise ValueError(
                f"sensitive_features['{feature_name}'] has length {len(group_array)}, "
                f"expected {len(y_true)}."
            )
    # --- Core computation ---
    report: dict = {}

    for feature_name, group_array in sensitive_features.items():
        group_array = np.asarray(group_array)
        groups = np.unique(group_array)
        group_results: dict = {}

        for g in groups:
            mask = group_array == g
            y_true_g = y_true[mask]
            y_pred_g = y_pred[mask]

            # TPR = TP / (TP + FN) — use recall_score for consistency
            tpr = float(recall_score(y_true_g, y_pred_g, zero_division=0))

            # FPR = FP / (FP + TN) — computed manually from confusion matrix
            tn, fp, fn, tp = 0, 0, 0, 0
            if len(y_true_g) > 0:
                cm = confusion_matrix(y_true_g, y_pred_g, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
            denominator = int(fp) + int(tn)
            fpr = float(fp / denominator) if denominator > 0 else 0.0

            group_results[str(g)] = {
                "n_samples": int(mask.sum()),
                "tpr": float(tpr),
                "fpr": float(fpr),
            }

        # Summary block
        if len(group_results) >= 2:
            tpr_values = [v["tpr"] for v in group_results.values()]
            fpr_values = [v["fpr"] for v in group_results.values()]
            tpr_gap = round(max(tpr_values) - min(tpr_values), 4)
            fpr_gap = round(max(fpr_values) - min(fpr_values), 4)
            best_tpr_group = max(group_results, key=lambda g: group_results[g]["tpr"])
            worst_tpr_group = min(group_results, key=lambda g: group_results[g]["tpr"])
        else:
            # Single subgroup — gaps are 0, no violation
            tpr_gap = 0.0
            fpr_gap = 0.0
            best_tpr_group = str(groups[0]) if len(groups) > 0 else "N/A"
            worst_tpr_group = best_tpr_group

        group_results["__summary__"] = {
            "tpr_gap": tpr_gap,
            "fpr_gap": fpr_gap,
            "tpr_violation": _violation_level(tpr_gap, severe_threshold, moderate_threshold),
            "fpr_violation": _violation_level(fpr_gap, severe_threshold, moderate_threshold),
            "best_tpr_group": best_tpr_group,
            "worst_tpr_group": worst_tpr_group,
        }

        report[feature_name] = group_results

    return report


def _violation_level(
    gap: float, severe_threshold: float = 0.15, moderate_threshold: float = 0.05
) -> str:
    """
    Classify a fairness gap into a violation severity level.

    Parameters
    ----------
    gap : float
        The fairness gap to classify (e.g. tpr_gap or fpr_gap).
    severe_threshold : float
        Gap above this value is classified as ``"severe"``. Default: 0.15.
    moderate_threshold : float
        Gap above this value is classified as ``"moderate"``. Default: 0.05.

    Returns
    -------
    str
        One of ``"severe"``, ``"moderate"``, or ``"acceptable"``.
    """
    if gap > severe_threshold:
        return "severe"
    elif gap >= moderate_threshold:
        return "moderate"
    else:
        return "acceptable"
