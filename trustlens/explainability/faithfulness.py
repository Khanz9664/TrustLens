# NOTE:
# This module is under active development and is not part of the public API.
# Do not import into production pipelines until stabilized.

"""
trustlens.explainability.faithfulness.
======================================
Faithfulness evaluation via pixel deletion and insertion tests.

Faithfulness measures whether an explanation method correctly identifies
the input features that most influence the model's prediction.

Tests
-----
* **Pixel Deletion** (ROAR — Remove And Retrain light variant):
 Sequentially remove the most important pixels (set to mean/zero)
 and measure how quickly model confidence drops. A faithful explanation
 should cause a steep degradation.

* **Pixel Insertion**:
 Start from a blurred/masked image and progressively insert the most
 important pixels. A faithful explanation should cause a fast rise
 in model confidence.

The *Area Under the Perturbation Curve* (AUPC) summarizes both tests.

References
----------
* Samek, W., et al. (2017). Evaluating the visualization of what a deep
 neural network has learned. IEEE TNNLS.
* Hooker, S., et al. (2019). A benchmark for interpretability methods in
 deep neural networks. NeurIPS.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def pixel_deletion_test(
    image: np.ndarray,
    saliency_map: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    target_class: int,
    n_steps: int = 20,
    baseline: str = "mean",
) -> dict:
    """
    Pixel deletion faithfulness test.

    Progressively replaces the most salient pixels with a baseline value
    and records the model's confidence at each step.

    Parameters
    ----------
    image : np.ndarray
      Input image, shape (H, W, C) or (H, W).
    saliency_map : np.ndarray
      Saliency / attribution map, shape (H, W). Higher = more important.
    predict_fn : callable
      Function that accepts a single image array and returns class
      probabilities, shape (n_classes,).
    target_class : int
      Class index to track confidence for.
    n_steps : int
      Number of deletion steps. Default 20.
    baseline : str
      Pixel fill value: ``"mean"`` (image mean) or ``"zero"``. Default ``"mean"``.

    Returns
    -------
    dict with keys:
      * ``confidences``  — confidence at each step (list of floats)
      * ``step_fractions`` — fraction of pixels deleted at each step
      * ``aupc``      — Area Under the Perturbation Curve (lower = more faithful)
      * ``initial_confidence`` — confidence before any deletion
      * ``final_confidence`` — confidence after deleting top n_steps fraction

    Examples
    --------
    >>> result = pixel_deletion_test(img, saliency, predict_fn, target_class=0)
    >>> print(f"AUPC (deletion): {result['aupc']:.4f}")
    """
    image = np.asarray(image, dtype=float)
    saliency = np.asarray(saliency_map, dtype=float)

    h, w = saliency.shape
    n_pixels = h * w

    # Sort pixels by descending saliency
    flat_saliency = saliency.flatten()
    sorted_indices = np.argsort(flat_saliency)[::-1]

    fill_value = image.mean() if baseline == "mean" else 0.0

    confidences = []
    step_fractions = []

    for step in range(n_steps + 1):
        frac = step / n_steps
        n_del = int(frac * n_pixels)

        perturbed = image.copy()
        if n_del > 0:
            del_idx = sorted_indices[:n_del]
            row_idx = del_idx // w
            col_idx = del_idx % w
            if perturbed.ndim == 3:
                perturbed[row_idx, col_idx, :] = fill_value
            else:
                perturbed[row_idx, col_idx] = fill_value

        prob = predict_fn(perturbed)
        conf = float(prob[target_class])
        confidences.append(conf)
        step_fractions.append(frac)

    if hasattr(np, "trapezoid"):
        aupc = float(np.trapezoid(confidences, step_fractions))  # type: ignore
    else:
        aupc = float(np.trapz(confidences, step_fractions))  # type: ignore

    return {
        "confidences": confidences,
        "step_fractions": step_fractions,
        "aupc": round(aupc, 4),
        "initial_confidence": confidences[0],
        "final_confidence": confidences[-1],
        "test": "deletion",
    }


def pixel_insertion_test(
    image: np.ndarray,
    saliency_map: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    target_class: int,
    n_steps: int = 20,
    blur_sigma: float = 10.0,
) -> dict:
    """
    Pixel insertion faithfulness test.

    Starts from a blurred baseline and progressively inserts the most
    salient pixels from the original image, tracking confidence rise.

    Parameters
    ----------
    image : np.ndarray
      Input image, shape (H, W, C) or (H, W).
    saliency_map : np.ndarray
      Saliency map, shape (H, W).
    predict_fn : callable
      Returns class probabilities for a single image.
    target_class : int
      Class to track.
    n_steps : int
      Number of insertion steps. Default 20.
    blur_sigma : float
      Gaussian blur sigma for the starting baseline image. Default 10.

    Returns
    -------
    dict with keys:
      * ``confidences``  — confidence at each step
      * ``step_fractions`` — fraction of pixels inserted at each step
      * ``aupc``      — Area Under the Perturbation Curve (higher = more faithful)
      * ``initial_confidence`` — confidence before insertion (blurred image)
      * ``final_confidence`` — confidence after full insertion

    Examples
    --------
    >>> result = pixel_insertion_test(img, saliency, predict_fn, target_class=0)
    >>> print(f"AUPC (insertion): {result['aupc']:.4f}")
    """
    image = np.asarray(image, dtype=float)
    saliency = np.asarray(saliency_map, dtype=float)

    h, w = saliency.shape
    n_pixels = h * w

    # Create blurred baseline
    try:
        from scipy.ndimage import gaussian_filter

        if image.ndim == 3:
            baseline = np.stack(
                [gaussian_filter(image[..., c], sigma=blur_sigma) for c in range(image.shape[2])],
                axis=-1,
            )
        else:
            baseline = gaussian_filter(image, sigma=blur_sigma)
    except ImportError:
        # Fallback: use mean image as baseline
        baseline = np.full_like(image, image.mean())

    flat_saliency = saliency.flatten()
    sorted_indices = np.argsort(flat_saliency)[::-1]  # most important first

    confidences = []
    step_fractions = []

    for step in range(n_steps + 1):
        frac = step / n_steps
        n_ins = int(frac * n_pixels)

        perturbed = baseline.copy()
        if n_ins > 0:
            ins_idx = sorted_indices[:n_ins]
            row_idx = ins_idx // w
            col_idx = ins_idx % w
            if perturbed.ndim == 3:
                perturbed[row_idx, col_idx, :] = image[row_idx, col_idx, :]
            else:
                perturbed[row_idx, col_idx] = image[row_idx, col_idx]

        prob = predict_fn(perturbed)
        conf = float(prob[target_class])
        confidences.append(conf)
        step_fractions.append(frac)

    if hasattr(np, "trapezoid"):
        aupc = float(np.trapezoid(confidences, step_fractions))  # type: ignore
    else:
        aupc = float(np.trapz(confidences, step_fractions))  # type: ignore

    return {
        "confidences": confidences,
        "step_fractions": step_fractions,
        "aupc": round(aupc, 4),
        "initial_confidence": confidences[0],
        "final_confidence": confidences[-1],
        "test": "insertion",
    }
