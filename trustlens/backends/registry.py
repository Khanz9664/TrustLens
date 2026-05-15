"""
trustlens.backends.registry
===========================
Registry and detection logic for framework-specific prediction resolvers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Optional

import numpy as np

from trustlens.backends.types import PredictionBundle, UnsupportedModelError

logger = logging.getLogger(__name__)


# Map of module prefix -> framework identifier
FRAMEWORK_MAPPING = {
    "sklearn": "sklearn",
    "xgboost": "xgboost",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "torch": "pytorch",
    "catboost": "catboost",
    "manual": "manual",
}

# Frameworks we can theoretically detect/support
SUPPORTED_FRAMEWORKS = tuple(sorted(set(FRAMEWORK_MAPPING.values())))

# Frameworks with concrete resolver implementations
IMPLEMENTED_RESOLVERS = tuple(sorted({"sklearn", "xgboost", "manual"}))


def manual_resolve(
    model: Any,
    X: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
) -> PredictionBundle:
    """
    Passthrough resolver for manual overrides.
    """
    if y_pred is None:
        if y_prob is not None:
            # Derive y_pred from y_prob if missing
            y_pred = np.argmax(np.asarray(y_prob), axis=1)
        else:
            raise ValueError("Manual override requires either y_pred or y_prob.")

    metadata = {
        "resolver": "manual_override",
        "model_type": type(model).__name__ if model is not None else "None",
    }

    # At this point y_pred is guaranteed to be a numpy array
    return PredictionBundle(
        y_pred=np.asarray(y_pred),
        y_prob=np.asarray(y_prob) if y_prob is not None else None,
        framework="manual",
        metadata=metadata,
    )


def detect_framework(model: Any, framework: Optional[str] = None) -> str:
    """
    Detect the ML framework for a given model using deterministic priority.
    """
    # 0. Handle None model (Manual Override path)
    if model is None:
        if framework is not None:
            return framework.lower()
        return "manual"

    # 1. Explicit override
    if framework is not None:
        normalized = framework.lower()
        # Check if the user provided one of our internal identifiers (e.g., 'pytorch')
        if normalized in SUPPORTED_FRAMEWORKS:
            return normalized
        # Check if the user provided a framework name we can map (e.g., 'torch')
        if normalized in FRAMEWORK_MAPPING:
            return FRAMEWORK_MAPPING[normalized]

        raise UnsupportedModelError(
            model_type=f"{type(model).__module__}.{type(model).__name__}",
            supported_frameworks=list(IMPLEMENTED_RESOLVERS),
        )

    # 2. Module-name inspection
    module_name = getattr(type(model), "__module__", "")
    if module_name:
        for prefix, identifier in FRAMEWORK_MAPPING.items():
            if module_name.startswith(prefix):
                return identifier

    # 3. Capability fallback (conservative)
    if hasattr(model, "predict") or hasattr(model, "predict_proba"):
        logger.debug("Detected sklearn-like model via capability")
        return "sklearn"

    # 4. Fail clearly
    raise UnsupportedModelError(
        model_type=f"{type(model).__module__}.{type(model).__name__}",
        supported_frameworks=list(IMPLEMENTED_RESOLVERS),
    )


def get_resolver(model: Any, framework: Optional[str] = None) -> Callable[..., PredictionBundle]:
    """
    Detect the framework and return the corresponding resolver function.
    """
    detected = detect_framework(model, framework=framework)

    if detected == "sklearn":
        from trustlens.backends import sklearn

        return sklearn.resolve

    if detected == "xgboost":
        from trustlens.backends import xgboost

        return xgboost.resolve

    if detected == "manual":
        return manual_resolve

    # Note: Future backends will be added here

    raise UnsupportedModelError(
        model_type=f"{type(model).__module__}.{type(model).__name__}",
        supported_frameworks=list(IMPLEMENTED_RESOLVERS),
    )


def get_supported_frameworks() -> list[str]:
    """Return a list of frameworks with implemented resolvers."""
    return list(IMPLEMENTED_RESOLVERS)
