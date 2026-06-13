"""
trustlens.utils.
================
Shared utility functions used across the TrustLens library.

Kept intentionally small — module-specific helpers live in their own module.
"""

from __future__ import annotations

import numbers
from typing import Any, cast

import numpy as np


def validate_array(arr: Any, name: str, ndim: int | None = None) -> np.ndarray:
    """
    Convert ``arr`` to a numpy array and validate its shape.

    Parameters
    ----------
    arr : Any
      Input data to validate.
    name : str
      Variable name for error messages.
    ndim : int, optional
      Expected number of dimensions.

    Returns
    -------
    np.ndarray

    Raises
    ------
    ValueError
      If ``arr`` is None, empty, or does not match the expected ``ndim``.
    """
    if arr is None:
        raise ValueError(f"'{name}' cannot be None")

    arr = np.asarray(arr)

    if arr.size == 0:
        raise ValueError(f"'{name}' cannot be empty")

    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"Expected '{name}' to have {ndim} dimensions, got {arr.ndim}.")
    return cast(np.ndarray, arr)


def check_consistent_length(*arrays: np.ndarray) -> None:
    """
    Verify that all arrays have the same first-dimension length.

    Raises
    ------
    ValueError
      If no arrays are provided, any array is None, or lengths differ.
    """
    if not arrays:
        raise ValueError("At least one array must be provided")

    for i, a in enumerate(arrays):
        if a is None:
            raise ValueError(f"Array at index {i} is None")

    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Inconsistent array lengths: {lengths}. "
            "All arrays must have the same number of samples."
        )


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Return numerator / denominator, or ``default`` if denominator is zero.

    Raises
    ------
    TypeError
      If numerator or denominator are not numeric.
    """
    if not isinstance(numerator, numbers.Number) or not isinstance(denominator, numbers.Number):
        raise TypeError(
            "Both numerator and denominator must be numeric (int, float, or numpy scalar)"
        )

    return float(numerator) / float(denominator) if abs(denominator) > 1e-12 else default


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """
    Flatten a nested dictionary using dot notation.

    Parameters
    ----------
    d : dict
      Nested input dictionary.
    parent_key : str
      Prefix for the current recursion level.
    sep : str
      Separator between parent and child keys. Default ``"."``.

    Returns
    -------
    dict
      Flattened dictionary.

    Raises
    ------
    TypeError
      If ``d`` is not a dictionary.

    Examples
    --------
    >>> flatten_dict({"a": {"b": 1, "c": {"d": 2}}})
    {'a.b': 1, 'a.c.d': 2}
    """
    if not isinstance(d, dict):
        raise TypeError("Input 'd' must be a dictionary")

    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def describe_array(arr: np.ndarray, name: str = "array") -> str:
    """Return a one-line descriptive string for an array (shape, dtype, stats)."""
    arr = np.asarray(arr)
    if arr.size == 0:
        return f"{name}: empty array, shape={arr.shape}, dtype={arr.dtype}"

    return (
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={arr.min():.4g}, max={arr.max():.4g}, mean={arr.mean():.4g}"
    )
