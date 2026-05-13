from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class PredictionBundle:
    """
    Standardized container for model predictions and framework metadata.

    Attributes
    ----------
    y_pred : np.ndarray
        Predicted class labels, shape (n_samples,).
    y_prob : np.ndarray | None
        Predicted class probabilities, shape (n_samples, n_classes).
        Can be None for models that do not provide probability scores.
    framework : str
        Identifier for the source framework (e.g., 'sklearn', 'xgboost').
    class_labels : np.ndarray | None
        Array of semantic class labels in the order corresponding to y_prob columns.
    metadata : dict[str, Any]
        Optional dictionary for resolver-specific telemetry and debugging
        (e.g., resolver name, detection method, framework version).
    """

    y_pred: np.ndarray
    y_prob: Optional[np.ndarray]
    framework: str
    class_labels: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Enforce strict shape invariants according to the prediction contract."""
        if self.y_pred.ndim != 1:
            raise ValueError(
                f"y_pred must be a 1D array. Found shape: {self.y_pred.shape}. "
                "Consult docs/internal/prediction_contract.md for details."
            )

        if self.y_prob is not None:
            if self.y_prob.ndim != 2:
                raise ValueError(
                    f"y_prob must be a 2D array. Found shape: {self.y_prob.shape}. "
                    "Consult docs/internal/prediction_contract.md for details."
                )

            if len(self.y_pred) != len(self.y_prob):
                raise ValueError(
                    f"Sample count mismatch: y_pred has {len(self.y_pred)} samples, "
                    f"but y_prob has {len(self.y_prob)} samples."
                )


class UnsupportedModelError(Exception):
    """
    Raised when TrustLens cannot determine the framework or resolve
    predictions for a given model type.
    """

    def __init__(
        self,
        model_type: str,
        supported_frameworks: Optional[Sequence[str]] = None,
    ) -> None:
        self.model_type = model_type
        self.supported_frameworks = supported_frameworks or []

        supported = ", ".join(self.supported_frameworks)
        message = (
            f"Unsupported model type: {model_type}. "
            f"Supported frameworks: {supported or 'none'}. "
            "You may provide y_pred and y_prob manually to bypass detection."
        )
        super().__init__(message)
