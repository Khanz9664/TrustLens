"""
trustlens.metrics.regression.
=============================
Reliability diagnostics for regression models: error distribution and
uncertainty calibration (beyond R²).

Metrics implemented
-------------------
* ``error_distribution`` — absolute-error (EPE) summary: MedAE, 90th-percentile
  error, max error, MAE, RMSE, plus histogram data for plotting.
* ``prediction_interval_coverage`` — PICP: does a model's prediction intervals
  actually contain the realised values at the stated confidence level?
* ``error_variance_correlation`` — does the model's predicted uncertainty track
  the magnitude of its actual errors?

The latter two degrade gracefully (returning a ``status="skipped"`` dict) when
the optional uncertainty inputs (intervals / predicted variance) are not
provided, mirroring the skip pattern used elsewhere in TrustLens.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "error_distribution",
    "prediction_interval_coverage",
    "error_variance_correlation",
]


def error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 20,
) -> dict:
    """
    Summarise the distribution of absolute errors (Expected Prediction Error).

    What it measures
    ----------------
    The spread of ``|y_true - y_pred|``, reported through robust summary
    statistics rather than a single mean.

    Why it matters
    --------------
    ``R²`` and MSE hide the *shape* of the error distribution. A model can have a
    respectable mean error while occasionally being catastrophically wrong; the
    median and the 90th-percentile error expose that tail.

    Limitations
    -----------
    Operates on point predictions only — it says nothing about whether the model
    *knew* it was uncertain (see :func:`error_variance_correlation`).

    Interpretation guidance
    -----------------------
    Lower is better. A large gap between the median and the 90th-percentile error
    signals a heavy tail of large mistakes worth investigating.

    Parameters
    ----------
    y_true : np.ndarray
      Ground-truth continuous targets, shape ``(n_samples,)``.
    y_pred : np.ndarray
      Model point predictions, shape ``(n_samples,)``.
    n_bins : int, default=20
      Number of bins for the returned absolute-error histogram.

    Returns
    -------
    dict with keys:
      * ``median_absolute_error`` — MedAE, robust central error.
      * ``p90_absolute_error``   — 90th-percentile absolute error (tail).
      * ``max_error``        — worst single absolute error.
      * ``mean_absolute_error``  — MAE.
      * ``rmse``           — root-mean-square error.
      * ``histogram_bins``     — bin edges for the absolute-error histogram.
      * ``error_hist``       — histogram counts (for plotting).
      * ``n_samples``        — number of samples scored.

    Raises
    ------
    ValueError
      If ``y_true`` and ``y_pred`` have mismatched shapes or are empty.

    Examples
    --------
    >>> dist = error_distribution(y_true, y_pred)
    >>> print(f"MedAE: {dist['median_absolute_error']}, p90: {dist['p90_absolute_error']}")
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}."
        )
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty.")

    abs_err = np.abs(y_true - y_pred)
    upper = float(abs_err.max())
    bins = np.linspace(0.0, upper if upper > 0 else 1.0, n_bins + 1)
    error_hist, _ = np.histogram(abs_err, bins=bins)

    return {
        "median_absolute_error": round(float(np.median(abs_err)), 4),
        "p90_absolute_error": round(float(np.percentile(abs_err, 90)), 4),
        "max_error": round(upper, 4),
        "mean_absolute_error": round(float(abs_err.mean()), 4),
        "rmse": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
        "histogram_bins": bins,
        "error_hist": error_hist,
        "n_samples": int(abs_err.size),
    }


def prediction_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    confidence_level: float = 0.95,
    tolerance: float = 0.05,
) -> dict:
    """
    Prediction Interval Coverage Probability (PICP).

    What it measures
    ----------------
    The fraction of realised values that fall inside the model's predicted
    ``[lower, upper]`` intervals, compared against the stated confidence level.

    Why it matters
    --------------
    An interval is only trustworthy if it covers what it claims to. A nominal 95%
    interval that actually covers 80% of points is over-confident — a silent risk
    in any decision that consumes the interval rather than the point estimate.

    Limitations
    -----------
    PICP is a *marginal* coverage measure: it can be satisfied on average while
    being badly wrong in specific regions of the input space. It also requires
    intervals; if none are supplied this metric is skipped.

    Interpretation guidance
    -----------------------
    ``picp ≈ confidence_level`` is the goal. ``picp`` well below the target ⇒
    intervals too narrow (over-confident); well above ⇒ too wide
    (under-confident, wasting precision).

    Parameters
    ----------
    y_true : np.ndarray
      Ground-truth continuous targets, shape ``(n_samples,)``.
    lower, upper : np.ndarray or None
      Per-sample lower / upper interval bounds, shape ``(n_samples,)``. If either
      is ``None`` the metric degrades gracefully (no intervals available).
    confidence_level : float, default=0.95
      The nominal coverage the intervals claim to provide.
    tolerance : float, default=0.05
      Absolute coverage gap within which coverage is deemed "well-calibrated".

    Returns
    -------
    dict
      When intervals are supplied: ``picp``, ``target_coverage``,
      ``calibration_error`` (``picp - target``), ``mean_interval_width`` and a
      ``verdict`` in {"over-confident", "under-confident", "well-calibrated"}.
      When intervals are missing: ``{"status": "skipped", "reason":
      "missing_intervals", "details": ...}``.

    Raises
    ------
    ValueError
      If supplied arrays have mismatched shapes, or any ``lower > upper``.

    Examples
    --------
    >>> prediction_interval_coverage(y_true, lo, hi, confidence_level=0.9)["picp"]
    """
    if lower is None or upper is None:
        return {
            "status": "skipped",
            "reason": "missing_intervals",
            "details": (
                "PICP requires per-sample prediction intervals (lower, upper). "
                "Provide them from a quantile/interval model to enable this metric."
            ),
        }

    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (y_true.shape == lower.shape == upper.shape):
        raise ValueError(
            "y_true, lower and upper must share the same shape, got "
            f"{y_true.shape}, {lower.shape}, {upper.shape}."
        )
    if y_true.size == 0:
        raise ValueError("y_true must be non-empty.")
    if np.any(lower > upper):
        raise ValueError("Each lower bound must be <= its corresponding upper bound.")

    covered = (y_true >= lower) & (y_true <= upper)
    picp = float(covered.mean())
    calibration_error = picp - confidence_level
    if calibration_error < -tolerance:
        verdict = "over-confident"
    elif calibration_error > tolerance:
        verdict = "under-confident"
    else:
        verdict = "well-calibrated"

    return {
        "picp": round(picp, 4),
        "target_coverage": confidence_level,
        "calibration_error": round(calibration_error, 4),
        "mean_interval_width": round(float(np.mean(upper - lower)), 4),
        "verdict": verdict,
        "n_samples": int(y_true.size),
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that returns 0.0 for degenerate (constant/short) inputs."""
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def error_variance_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    predicted_variance: np.ndarray | None = None,
) -> dict:
    """
    Correlation between the model's predicted uncertainty and its actual error.

    What it measures
    ----------------
    How well the model's per-sample predicted variance (or any uncertainty score)
    ranks alongside the realised absolute error ``|y_true - y_pred|`` — via both
    Pearson (linear) and Spearman (rank/monotonic) correlation.

    Why it matters
    --------------
    A trustworthy probabilistic model should be *more* uncertain exactly where it
    is *more* wrong. If predicted variance is uncorrelated with error, the
    uncertainty estimates are decorative and unsafe to gate decisions on.

    Limitations
    -----------
    Correlation captures monotonic association, not calibrated magnitude — pair it
    with :func:`prediction_interval_coverage` for an absolute check. Requires a
    predicted-variance input; skipped if none is supplied.

    Interpretation guidance
    -----------------------
    Higher positive correlation is better (uncertainty tracks error). Near-zero or
    negative correlation means the uncertainty signal is not informative.

    Parameters
    ----------
    y_true : np.ndarray
      Ground-truth continuous targets, shape ``(n_samples,)``.
    y_pred : np.ndarray
      Model point predictions, shape ``(n_samples,)``.
    predicted_variance : np.ndarray or None
      Per-sample predicted variance or uncertainty score, shape ``(n_samples,)``.
      If ``None`` the metric degrades gracefully.

    Returns
    -------
    dict
      When variance is supplied: ``pearson``, ``spearman``, ``verdict`` in
      {"informative", "weak", "uninformative"} and ``n_samples``. When missing:
      ``{"status": "skipped", "reason": "missing_variance", "details": ...}``.

    Raises
    ------
    ValueError
      If supplied arrays have mismatched shapes or are empty.

    Examples
    --------
    >>> error_variance_correlation(y_true, y_pred, variance)["spearman"]
    """
    if predicted_variance is None:
        return {
            "status": "skipped",
            "reason": "missing_variance",
            "details": (
                "Requires a per-sample predicted variance / uncertainty score "
                "(e.g. from a Gaussian process, ensemble spread, or NGBoost)."
            ),
        }

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    predicted_variance = np.asarray(predicted_variance, dtype=float)
    if not (y_true.shape == y_pred.shape == predicted_variance.shape):
        raise ValueError(
            "y_true, y_pred and predicted_variance must share the same shape, got "
            f"{y_true.shape}, {y_pred.shape}, {predicted_variance.shape}."
        )
    if y_true.size == 0:
        raise ValueError("Inputs must be non-empty.")

    abs_err = np.abs(y_true - y_pred)
    pearson = _safe_corr(predicted_variance, abs_err)
    # Spearman = Pearson on average ranks (ties handled by averaging).
    var_ranks = _average_ranks(predicted_variance)
    err_ranks = _average_ranks(abs_err)
    spearman = _safe_corr(var_ranks, err_ranks)

    strongest = max(pearson, spearman)
    if strongest >= 0.5:
        verdict = "informative"
    elif strongest >= 0.2:
        verdict = "weak"
    else:
        verdict = "uninformative"

    return {
        "pearson": round(pearson, 4),
        "spearman": round(spearman, 4),
        "verdict": verdict,
        "n_samples": int(y_true.size),
    }


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks of ``x`` (ties share the mean of their positions)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    # Resolve ties to their average rank.
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size, dtype=float)
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]
