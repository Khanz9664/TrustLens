"""
trustlens.plugins.base.
=======================
Abstract base class for all TrustLens plugins.

Every plugin must subclass ``BasePlugin`` and implement ``run()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BasePlugin(ABC):
    """
    Abstract base class for TrustLens plugins.

    Subclass this and implement ``run()`` to create a custom analysis plugin.

    Class Attributes
    ----------------
    name : str
      Unique identifier used to register and look up the plugin.
      Must be a valid Python identifier (no spaces).
    description : str
      Human-readable description displayed in the plugin registry listing.
    version : str
      Plugin version string. Default ``"0.2.0"``.

    Examples
    --------
    >>> class MyPlugin(BasePlugin):
    ...   name = "my_plugin"
    ...   description = "Computes a custom metric."
    ...
    ...   def run(self, model, X, y_true, y_pred, y_prob, **kwargs):
    ...     return {"custom_result": 1.0}
    """

    name: str = ""
    description: str = ""
    version: str = "0.2.0"

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(
        self,
        model: Any,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute the plugin analysis.

        Parameters
        ----------
        model : Any
          Trained model (same object passed to ``analyze()``).
        X : np.ndarray
          Feature matrix.
        y_true : np.ndarray
          Ground-truth labels.
        y_pred : np.ndarray
          Model predictions.
        y_prob : np.ndarray
          Predicted probabilities.
        **kwargs : Any
          Plugin-specific keyword arguments.

        Returns
        -------
        dict
          Plugin results — must be JSON-serializable (no raw tensors).
        """
        ...

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """
        Optional validation hook. Called before ``run()``.

        Override to check dependencies, configuration, etc.
        Return ``False`` to skip plugin execution with a warning.
        """
        return True

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, version={self.version!r})"
