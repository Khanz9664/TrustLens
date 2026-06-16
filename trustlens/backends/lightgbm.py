"""
trustlens.backends.lightgbm
===========================
Prediction resolver for LightGBM models.

Architecture
------------
Handles prediction extraction from both:

* lightgbm.LGBMClassifier
* lightgbm.Booster

Probability Extraction Strategy
-------------------------------
* For LGBMClassifier, uses predict_proba().
* For Booster, uses predict().
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
    Prediction resolver for LightGBM models.

    Supports ``lightgbm.LGBMClassifier`` and raw ``lightgbm.Booster``.
    Regression objectives are explicitly blocked.

    Parameters
    ----------
    model : Any
        A fitted LightGBM model (``LGBMClassifier`` or ``Booster``).
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    y_pred : np.ndarray, optional
        Predicted class labels. Derived from ``y_prob`` when not provided.
    y_prob : np.ndarray, optional
        Predicted class probabilities. Computed from the model when not provided.
        Binary outputs are normalized to shape ``(n_samples, 2)``.
    class_labels : np.ndarray, optional
        Semantic class labels. Falls back to ``model.classes_`` when available.

    Returns
    -------
    PredictionBundle
        Standardized prediction container with ``y_pred``, ``y_prob``,
        ``framework='lightgbm'``, and resolver metadata.

    Raises
    ------
    ImportError
        If the ``lightgbm`` package is not installed.
    NotImplementedError
        If the model is a regressor or uses a regression objective.
    ValueError
        If probabilities cannot be resolved from the model.
    """
    try:
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
    except ImportError as e:
        raise ImportError("LightGBM support requires the 'lightgbm' package.") from e

    # 1. Objective detection / regression blocking
    # Explicit estimator-type rejection (most reliable)
    if isinstance(model, LGBMRegressor):
        raise NotImplementedError(
            "TrustLens currently supports classification models only. "
            "LightGBM regressors are not supported."
        )

    objective = ""

    # Fitted LightGBM models sometimes expose objective_
    if hasattr(model, "objective_"):
        objective = str(model.objective_ or "")

    # sklearn wrapper params
    if not objective and hasattr(model, "get_params"):
        objective = str(model.get_params().get("objective") or "")

    # native Booster
    if not objective and isinstance(model, lgb.Booster):
        try:
            objective = str(model.params.get("objective") or "")
        except Exception:
            objective = ""

    objective = objective.lower()

    regression_objectives = {
        "regression",
        "regression_l1",
        "huber",
        "fair",
        "poisson",
        "quantile",
        "mape",
        "gamma",
        "tweedie",
    }

    if objective in regression_objectives:
        raise NotImplementedError(
            "TrustLens currently supports classification models only. "
            f"LightGBM model with objective '{objective}' is not supported."
        )

    # 2. Resolve probabilities
    if y_prob is None:
        if isinstance(model, lgb.Booster):
            y_prob = np.asarray(model.predict(X), dtype=np.float64)

        elif hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)

        elif hasattr(model, "predict"):
            y_prob = np.asarray(model.predict(X), dtype=np.float64)

        else:
            raise ValueError(
                "Could not resolve probabilities for LightGBM model. "
                "Ensure the model has 'predict_proba()' or 'predict()'."
            )

    # 3. Normalize probabilities
    y_prob = np.asarray(y_prob, dtype=np.float64)

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
        # 1. Prefer derived labels from probabilities (IMPORTANT FIX)
        if y_prob is not None:
            y_prob_arr = np.asarray(y_prob)
            y_pred_indices = np.argmax(y_prob_arr, axis=1)

            if resolved_class_labels is not None:
                y_pred = resolved_class_labels[y_pred_indices]
            else:
                y_pred = y_pred_indices

        # 2. Fallback only if probabilities are NOT provided
        else:
            if hasattr(model, "predict"):
                y_pred = np.asarray(model.predict(X)).reshape(-1)
            else:
                raise ValueError("Cannot resolve y_pred")

    # 5. Metadata
    metadata = {
        "resolver": "lightgbm",
        "framework_version": getattr(lgb, "__version__", "unknown"),
        "model_type": (type(model).__name__ if not isinstance(model, lgb.Booster) else "Booster"),
        "objective": objective,
    }

    return PredictionBundle(
        y_pred=np.asarray(y_pred),
        y_prob=y_prob,
        framework="lightgbm",
        class_labels=resolved_class_labels,
        metadata=metadata,
    )
