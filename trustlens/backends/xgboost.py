"""
trustlens.backends.xgboost
==========================
Prediction resolver for XGBoost models.

Architecture
------------
Handles the extraction of predictions from both the scikit-learn API (`XGBClassifier`)
and the native `xgboost.Booster` API.

Probability Extraction Strategy
-------------------------------
* For `XGBClassifier`, uses `predict_proba()`.
* For `Booster`, automatically wraps the input `X` in an `xgb.DMatrix` and calls `predict()`.
* Normalizes binary probability outputs to the standardized (n_samples, 2) shape.

Label Mapping Behavior
----------------------
* Automatically handles label mapping if `classes_` is available (e.g., from `XGBClassifier`).
* For native `Booster`, falls back to provided `class_labels` or raw integer indices.
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
    Prediction resolver for XGBoost models.
    Supports XGBClassifier and raw Booster objects.
    """
    import json

    import xgboost as xgb

    # 1. Objective Detection & Regression blocking
    objective = ""
    if hasattr(model, "objective"):
        # XGBClassifier path
        objective = str(model.objective)
    elif hasattr(model, "get_params"):
        # Scikit-learn wrapper path
        objective = str(model.get_params().get("objective", ""))
    elif isinstance(model, xgb.Booster):
        # Raw Booster path: extract from config
        try:
            config = json.loads(model.save_config())
            # Path in JSON config: learner -> objective
            obj_data = config.get("learner", {}).get("objective")
            if isinstance(obj_data, dict):
                objective = obj_data.get("name", "")
            else:
                objective = str(obj_data or "")

            if not objective:
                # Fallback for older versions
                objective = str(config.get("objective", ""))
        except Exception:
            objective = ""

    objective = str(objective)

    if objective.startswith(("reg:", "rank:")):
        raise NotImplementedError(
            f"TrustLens currently supports classification models only. "
            f"XGBoost model with objective '{objective}' is not supported."
        )

    # 2. Resolve Probabilities
    if y_prob is None:
        if isinstance(model, xgb.Booster):
            # Raw Booster requires DMatrix
            dmat = xgb.DMatrix(X)
            y_prob = model.predict(dmat)
        elif hasattr(model, "predict_proba"):
            # XGBClassifier path
            y_prob = model.predict_proba(X)
        elif hasattr(model, "predict"):
            # Generic predict fallback
            y_prob = model.predict(X)
        else:
            raise ValueError(
                "Could not resolve probabilities for XGBoost model. "
                "Ensure the model has 'predict_proba()' or 'predict()'."
            )

    # 3. Normalize Probabilities
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:
        # Binary case: [p] -> [1-p, p]
        y_prob = np.column_stack([1 - y_prob, y_prob])
    elif y_prob.ndim == 2 and y_prob.shape[1] == 1:
        # Binary case: [[p]] -> [1-p, p]
        y_prob_flat = y_prob.flatten()
        y_prob = np.column_stack([1 - y_prob_flat, y_prob_flat])

    model_class_labels = getattr(model, "classes_", None)
    if model_class_labels is not None:
        resolved_class_labels = np.asarray(model_class_labels)
    elif class_labels is not None:
        resolved_class_labels = np.asarray(class_labels)
    else:
        resolved_class_labels = None

    # 4. Resolve Class Predictions
    if y_pred is None:
        if hasattr(model, "predict") and not isinstance(model, xgb.Booster):
            # XGBClassifier.predict() handles label mapping (classes_)
            y_pred = model.predict(X)
        else:
            # Fallback to argmax from probabilities
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
        "resolver": "xgboost",
        "framework_version": getattr(xgb, "__version__", "unknown"),
        "model_type": type(model).__name__ if not isinstance(model, xgb.Booster) else "Booster",
        "objective": objective,
    }

    return PredictionBundle(
        y_pred=np.asarray(y_pred),
        y_prob=y_prob,
        framework="xgboost",
        class_labels=resolved_class_labels,
        metadata=metadata,
    )
