"""
trustlens.backends.registry
===========================
Registry and detection logic for framework-specific prediction resolvers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Optional

from trustlens.backends.types import UnsupportedModelError

logger = logging.getLogger(__name__)


# Map of module prefix -> framework identifier
FRAMEWORK_MAPPING = {
    "sklearn": "sklearn",
    "xgboost": "xgboost",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "torch": "pytorch",
    "catboost": "catboost",
}

# Frameworks we can theoretically detect/support
SUPPORTED_FRAMEWORKS = tuple(sorted(set(FRAMEWORK_MAPPING.values())))

# Frameworks with concrete resolver implementations
IMPLEMENTED_RESOLVERS = tuple(sorted({"sklearn"}))


def detect_framework(model: Any, framework: Optional[str] = None) -> str:
    """
    Detect the ML framework for a given model using deterministic priority.

    Priority:
    1. Explicit override (validated)
    2. Module-name inspection (prefix-based, no eager imports)
    3. Capability fallback (predict, predict_proba)
    """
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
    has_predict = hasattr(model, "predict")
    has_proba = hasattr(model, "predict_proba")

    if has_predict and has_proba:
        logger.debug("Detected sklearn-like model via capability (predict + predict_proba)")
        return "sklearn"

    # 4. Fail clearly
    raise UnsupportedModelError(
        model_type=f"{type(model).__module__}.{type(model).__name__}",
        supported_frameworks=list(IMPLEMENTED_RESOLVERS),
    )


def get_resolver(model: Any, framework: Optional[str] = None) -> Callable:
    """
    Detect the framework and return the corresponding resolver function.
    """
    detected = detect_framework(model, framework=framework)

    if detected == "sklearn":
        from trustlens.backends import sklearn

        return sklearn.resolve

    # Note: Future backends will be added here
    # elif detected == "xgboost":
    #     from trustlens.backends import xgboost
    #     return xgboost.resolve

    raise UnsupportedModelError(
        model_type=f"{type(model).__module__}.{type(model).__name__}",
        supported_frameworks=list(IMPLEMENTED_RESOLVERS),
    )


def get_supported_frameworks() -> list[str]:
    """Return a list of frameworks with implemented resolvers."""
    return list(IMPLEMENTED_RESOLVERS)
