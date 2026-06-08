"""
trustlens.backends.sklearn
==========================
Prediction resolver for scikit-learn models.

Architecture
------------
This backend translates sklearn-compatible models into standardized `PredictionBundle`s.

Probability Extraction Strategy
-------------------------------
* Tries to call `predict_proba()` on the model.
* If probabilities are binary (n_samples,), reshapes them to (n_samples, 2) to maintain invariant shape.

Label Mapping Behavior
----------------------
* Inspects the `classes_` attribute on the model.
* Uses these classes to map probability array indices to semantic labels during `y_pred` derivation.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from trustlens.backends.types import PredictionBundle, UnsupportedModelError


def resolve(
    model: Any,
    X: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
    class_labels: Optional[np.ndarray] = None,
) -> PredictionBundle:
    """
    Resolve predictions and probabilities from a scikit-learn model.

    Handles binary probability normalization to (n, 2) and extracts class labels.
    """
    import sklearn
    from sklearn.base import is_classifier

    # 1. Classifier Validation (PR 2 Contract enforcement)
    # We recognize classifiers via sklearn's helper (if safe) or via common classification attributes
    is_clf = False
    try:
        is_clf = is_classifier(model)
    except Exception:
        pass

    if not is_clf:
        # Fallback for non-standard or mock objects
        is_clf = (
            getattr(model, "_estimator_type", None) == "classifier"
            or hasattr(model, "classes_")
            or hasattr(model, "predict_proba")
        )

    if not is_clf:
        raise NotImplementedError(
            f"TrustLens currently supports classification models only. "
            f"Model type '{type(model).__name__}' is not a recognized classifier."
        )

    # 2. Resolve Probabilities
    if y_prob is None:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)
        else:
            raise ValueError("y_prob is required when model does not expose predict_proba().")

    # 3. Normalize Probabilities (Contract Enforcement)
    # Binary: (n,) or (n, 1) -> (n, 2)
    if y_prob is not None:
        if y_prob.ndim == 1:
            y_prob = np.stack([1 - y_prob, y_prob], axis=1)
        elif y_prob.ndim == 2 and y_prob.shape[1] == 1:
            y_prob = np.hstack([1 - y_prob, y_prob])

    # 4. Resolve Predictions
    if y_pred is None:
        if y_prob is not None:
            # Derive from normalized probabilities
            y_pred_indices = np.argmax(y_prob, axis=1)

            # Defensive label mapping
            classes = getattr(model, "classes_", None)
            if classes is None:
                classes = class_labels
            if classes is not None:
                classes_arr = np.asarray(classes)
                if len(classes_arr) == y_prob.shape[1]:
                    y_pred = classes_arr[y_pred_indices]
                else:
                    y_pred = y_pred_indices
            else:
                y_pred = y_pred_indices
        elif hasattr(model, "predict"):
            y_pred = model.predict(X)
        else:
            raise UnsupportedModelError(
                model_type=type(model).__name__,
                supported_frameworks=["sklearn"],
            )

    # 5. Metadata & Labels
    model_class_labels = getattr(model, "classes_", None)
    if model_class_labels is not None:
        resolved_class_labels = np.asarray(model_class_labels)
    elif class_labels is not None:
        resolved_class_labels = np.asarray(class_labels)
    else:
        resolved_class_labels = None

    metadata = {
        "resolver": "sklearn",
        "detection_method": "manual",  # Default value. Registry may override.
        "framework_version": getattr(sklearn, "__version__", "unknown"),
    }

    return PredictionBundle(
        y_pred=y_pred,
        y_prob=y_prob,
        framework="sklearn",
        class_labels=resolved_class_labels,
        metadata=metadata,
    )
