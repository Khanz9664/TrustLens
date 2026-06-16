"""
trustlens.backends.catboost
===========================
Prediction resolver for CatBoost models.

Architecture
------------
Handles prediction extraction from:

* CatBoostClassifier
* catboost.Pool inputs

Probability Extraction Strategy
-------------------------------
* Uses predict_proba().
* Supports binary and multiclass classification.
* Binary probabilities are normalized to shape (n_samples, 2).

Label Mapping Behavior
----------------------
* Uses classes_ when available.
* Falls back to provided class_labels.
* Otherwise uses raw class indices.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from trustlens.backends.types import PredictionBundle

logger = logging.getLogger(__name__)


def resolve(
    model: Any,
    X: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
    class_labels: Optional[np.ndarray] = None,
) -> PredictionBundle:
    """
    Prediction resolver for CatBoost models.

    Supports:
    - CatBoostClassifier
    - catboost.Pool inputs
    """
    try:
        import catboost
        from catboost import CatBoostClassifier
    except ImportError as e:
        raise ImportError("CatBoost support requires the 'catboost' package.") from e

    # 1. Classification validation
    if not isinstance(model, CatBoostClassifier):
        raise NotImplementedError(
            f"TrustLens currently supports classification models only. "
            f"Model type '{type(model).__name__}' is not supported."
        )

    # 2. Resolve probabilities
    if y_prob is None:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)

        elif hasattr(model, "predict"):
            y_prob = model.predict(X)

        else:
            raise ValueError(
                "Could not resolve probabilities for CatBoost model. "
                "Ensure the model has 'predict_proba()' or 'predict()'."
            )

    # 3. Normalize probabilities
    y_prob = np.asarray(y_prob)

    if y_prob.ndim == 1:
        y_prob = np.column_stack([1 - y_prob, y_prob])

    elif y_prob.ndim == 2 and y_prob.shape[1] == 1:
        y_prob_flat = y_prob.flatten()
        y_prob = np.column_stack([1 - y_prob_flat, y_prob_flat])

    model_class_labels = getattr(model, "classes_", None)

    if model_class_labels is not None:
        resolved_class_labels = np.asarray(model_class_labels)
    elif class_labels is not None:
        resolved_class_labels = np.asarray(class_labels)
    else:
        resolved_class_labels = None

    # 4. Resolve class predictions
    if y_pred is None:
        y_pred_indices = np.argmax(y_prob, axis=1)

        if resolved_class_labels is not None:
            if len(resolved_class_labels) == y_prob.shape[1]:
                y_pred = resolved_class_labels[y_pred_indices]
            else:
                y_pred = y_pred_indices
        else:
            y_pred = y_pred_indices

    # 5. Metadata
    metadata = {
        "resolver": "catboost",
        "framework_version": getattr(catboost, "__version__", "unknown"),
        "model_type": type(model).__name__,
    }

    return PredictionBundle(
        y_pred=np.asarray(y_pred),
        y_prob=y_prob,
        framework="catboost",
        class_labels=resolved_class_labels,
        metadata=metadata,
    )
