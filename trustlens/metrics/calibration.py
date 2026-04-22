"""
trustlens.metrics.calibration.
==============================
Calibration metrics for probabilistic classifiers.

Calibration measures how well a model's predicted probabilities reflect
the true likelihood of outcomes. A perfectly calibrated model that predicts
80% confidence for a set of samples should be correct ~80% of the time.

Metrics implemented
-------------------
* ``brier_score``       — proper scoring rule for probabilistic forecasts
* ``expected_calibration_error`` — binned confidence vs accuracy gap
* ``reliability_curve``    — data for reliability (calibration) diagrams

References
----------
* Brier, G. W. (1950). Verification of forecasts expressed in terms of
  probability. Monthly Weather Review, 78(1), 1–3.
* Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities
  with supervised learning. ICML.
* Guo, C., et al. (2017). On calibration of modern neural networks. ICML.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Brier Score
# ---------------------------------------------------------------------------


def brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    r"""
    Compute the Brier Score for a binary probabilistic classifier.

    The Brier Score is the mean squared difference between predicted
    probabilities and actual outcomes. Lower is better; a perfect
    forecaster scores 0.0, a random coin-flip scores ~0.25.

    .. math::
      \\text{BS} = \\frac{1}{N} \\sum_{i=1}^{N}
             \\bigl(\\hat{p}_i - y_i\\bigr)^2

    Parameters
    ----------
    y_true : np.ndarray
      Binary ground-truth labels (0 or 1), shape (n_samples,).
    y_prob : np.ndarray
      Predicted probabilities for the positive class, shape (n_samples,).

    Returns
    -------
    float
      Brier Score in [0, 1].

    Raises
    ------
    ValueError
      If ``y_true`` and ``y_prob`` have different lengths, or if
      ``y_true`` contains values outside {0, 1}.

    Examples
    --------
    >>> import numpy as np
    >>> from trustlens.metrics.calibration import brier_score
    >>> y_true = np.array([1, 0, 1, 1, 0])
    >>> y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.3])
    >>> brier_score(y_true, y_prob)
    0.036
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}.")

    unique_labels = np.unique(y_true)
    if not set(unique_labels.tolist()).issubset({0.0, 1.0}):
        raise ValueError(
            f"brier_score expects binary labels (0/1). Got unique values: {unique_labels}."
        )

    return float(np.mean((y_prob - y_true) ** 2))


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    r"""
    Compute the Expected Calibration Error (ECE).

    ECE measures the weighted average absolute difference between
    predicted confidence and actual accuracy across probability bins.

    .. math::
      \\text{ECE} = \\sum_{b=1}^{B}
             \\frac{|\\mathcal{B}_b|}{N}
             \\left|\\text{acc}(\\mathcal{B}_b) -
                 \\text{conf}(\\mathcal{B}_b)\\right|

    Parameters
    ----------
    y_true : np.ndarray
      Binary ground-truth labels (0 or 1), shape (n_samples,).
    y_prob : np.ndarray
      Predicted probabilities for the positive class, shape (n_samples,).
    n_bins : int
      Number of confidence bins. Default 10.
    strategy : str
      Binning strategy — ``"uniform"`` (equal-width) or ``"quantile"``
      (equal-frequency). Default ``"uniform"``.

    Returns
    -------
    float
      ECE value in [0, 1]. Lower is better.

    Examples
    --------
    >>> from trustlens.metrics.calibration import expected_calibration_error
    >>> ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1))
        bin_edges = np.unique(bin_edges)  # remove duplicates at extremes
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'uniform' or 'quantile'.")

    ece = 0.0
    n = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Include the right edge in the last bin
        if hi == bin_edges[-1]:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        accuracy = y_true[mask].mean()
        confidence = y_prob[mask].mean()
        ece += (n_bin / n) * abs(accuracy - confidence)

    return float(ece)


# ---------------------------------------------------------------------------
# Reliability Curve
# ---------------------------------------------------------------------------


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the reliability (calibration) curve data.

    Returns the mean predicted probability, fraction of positives,
    and bin counts for each confidence bin. Use this data with
    ``trustlens.visualization.plot_reliability_diagram`` to render
    a calibration plot.

    Parameters
    ----------
    y_true : np.ndarray
      Binary ground-truth labels (0 or 1).
    y_prob : np.ndarray
      Predicted probabilities for the positive class.
    n_bins : int
      Number of confidence bins. Default 10.
    strategy : str
      ``"uniform"`` or ``"quantile"``. Default ``"uniform"``.

    Returns
    -------
    fraction_of_positives : np.ndarray
      Actual fraction of positive samples in each bin.
    mean_predicted_value : np.ndarray
      Mean predicted probability in each bin.
    bin_counts : np.ndarray
      Number of samples in each bin.

    Examples
    --------
    >>> frac_pos, mean_pred, counts = reliability_curve(y_true, y_prob)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1))
        # Collapse duplicate edges that arise when many samples share the
        # same value (like a majority-class predictor at 0.0 or 1.0).
        bin_edges = np.unique(bin_edges)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'uniform' or 'quantile'.")

    n_bins_actual = len(bin_edges) - 1
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins_actual - 1)

    counts = np.bincount(bin_idx, minlength=n_bins_actual)
    prob_sum = np.bincount(bin_idx, weights=y_prob, minlength=n_bins_actual)
    true_sum = np.bincount(bin_idx, weights=y_true, minlength=n_bins_actual)

    active = counts > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_pos = np.where(active, true_sum / counts, 0.0)
        mean_pred = np.where(active, prob_sum / counts, 0.0)

    return (
        frac_pos[active],
        mean_pred[active],
        counts[active].astype(int),
    )
