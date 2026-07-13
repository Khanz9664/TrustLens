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
    "multilevel_interval_coverage",
    "crps_from_intervals",
    "crps_decomposition",
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
    if n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins}.")

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

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}.")
    if not 0.0 <= tolerance < 1.0:
        raise ValueError(f"tolerance must be in [0, 1), got {tolerance}.")

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


def multilevel_interval_coverage(
    y_true: np.ndarray,
    intervals: dict[float, tuple[np.ndarray, np.ndarray]] | None = None,
    tolerance: float = 0.05,
) -> dict:
    """
    Multi-level interval calibration (RFC #155): ICE + a calibration-conditioned
    sharpness proxy.

    What it measures
    ----------------
    Where :func:`prediction_interval_coverage` checks a single nominal level,
    this evaluates a *set* of prediction-interval levels at once and summarises
    them with two complementary numbers:

    * **ICE** (Interval Calibration Error) — the mean absolute coverage gap
      across the supplied levels, ``mean_tau |emp(tau) - tau|``. The multi-level
      analog of ``|PICP calibration_error|`` (and of ECE for classification):
      one continuous calibration signal that summarises the whole reliability
      curve rather than a single point on it.
    * **sharpness_skill** — a *calibration-conditioned* sharpness proxy in the
      spirit of the CRPS Resolution component. Among only the levels that
      actually pass calibration (``|emp(tau) - tau| <= tolerance``), it compares
      the model's mean interval width against the climatology interval at the
      same level: ``1 - mean(model_width / climatology_width)``. Higher is
      better (intervals sharper than the marginal baseline while staying
      honest). Restricting to calibrated levels is the point: intervals that
      look "sharp" only because they are over-confident fail the calibration
      gate and are excluded, so they cannot inflate the score.

    Why two numbers
    ---------------
    They isolate distinct properties: ICE answers "are the stated probabilities
    honest?" while sharpness_skill answers "given honest probabilities, how
    discriminative is the uncertainty?". Raw CRPS conflates the two (plus
    accuracy), which is exactly what we avoid by reporting them separately (see
    RFC #155 — the full CRPS Reliability/Resolution decomposition can later
    replace this proxy in place, as it measures the same property).

    Parameters
    ----------
    y_true : np.ndarray
      Ground-truth continuous targets, shape ``(n_samples,)``.
    intervals : dict[float, tuple[np.ndarray, np.ndarray]] or None
      Maps each nominal coverage level ``tau in (0, 1)`` to its per-sample
      ``(lower, upper)`` bounds, each shape ``(n_samples,)``. ``None`` or an
      empty mapping degrades gracefully (returns a ``status="skipped"`` dict).
      A single-level mapping is valid and additionally emits the back-compatible
      single-PICP fields (``picp``, ``target_coverage``, ``calibration_error``).
    tolerance : float, default=0.05
      Absolute coverage gap within which a level is deemed calibrated — used
      both for the verdict and as the gate for the sharpness proxy.

    Returns
    -------
    dict
      When intervals are supplied: ``ice``, ``sharpness_skill`` (``None`` if no
      level passes the calibration gate), ``n_levels``, ``n_calibrated_levels``,
      ``worst_calibration_error`` (most negative ``emp - tau``; drives the
      over-confidence blocker downstream), ``mean_interval_width``, a
      ``per_level`` table, a ``verdict`` and ``n_samples``. A single-level call
      also includes ``picp`` / ``target_coverage`` / ``calibration_error``.
      When intervals are missing: ``{"status": "skipped", "reason":
      "missing_intervals", "details": ...}``.

    Raises
    ------
    ValueError
      If arrays mismatch ``y_true``'s shape, any ``lower > upper``, any level is
      outside ``(0, 1)``, ``y_true`` is empty, or ``tolerance`` is out of range.

    Examples
    --------
    >>> ivs = {0.5: (lo50, hi50), 0.9: (lo90, hi90)}
    >>> multilevel_interval_coverage(y_true, ivs)["ice"]
    """
    if not intervals:
        return {
            "status": "skipped",
            "reason": "missing_intervals",
            "details": (
                "Multi-level interval calibration requires a mapping of nominal "
                "levels to per-sample (lower, upper) bounds. Provide them from a "
                "quantile/interval model to enable ICE and the sharpness proxy."
            ),
        }

    if not 0.0 <= tolerance < 1.0:
        raise ValueError(f"tolerance must be in [0, 1), got {tolerance}.")

    y_true = np.asarray(y_true, dtype=float)
    if y_true.size == 0:
        raise ValueError("y_true must be non-empty.")

    levels = sorted(float(t) for t in intervals)
    for tau in levels:
        if not 0.0 < tau < 1.0:
            raise ValueError(f"each interval level must be in (0, 1), got {tau}.")

    # Climatology reference widths: the marginal central interval of y at each
    # level, computed once from the raw quantiles (vectorized over levels).
    lo_q = np.clip([0.5 - t / 2.0 for t in levels], 0.0, 1.0)
    hi_q = np.clip([0.5 + t / 2.0 for t in levels], 0.0, 1.0)
    ref_widths = np.asarray(np.quantile(y_true, hi_q)) - np.asarray(np.quantile(y_true, lo_q))

    per_level: list[dict] = []
    abs_errors: list[float] = []
    widths: list[float] = []
    # Raw (unrounded) model_width / climatology_width over calibrated levels — kept
    # separate from the rounded per_level report values so display rounding never
    # biases the sharpness proxy.
    ratios: list[float] = []
    worst_cal_err = np.inf

    for i, tau in enumerate(levels):
        lower, upper = intervals[tau]
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if not (y_true.shape == lower.shape == upper.shape):
            raise ValueError(
                "y_true and each (lower, upper) pair must share the same shape; "
                f"level {tau} got {lower.shape}, {upper.shape} vs {y_true.shape}."
            )
        if np.any(lower > upper):
            raise ValueError(
                f"each lower bound must be <= its upper bound (violated at level {tau})."
            )

        emp = float(((y_true >= lower) & (y_true <= upper)).mean())
        cal_err = emp - tau
        width = float(np.mean(upper - lower))
        ref_width = float(ref_widths[i])
        calibrated = abs(cal_err) <= tolerance

        abs_errors.append(abs(cal_err))
        widths.append(width)
        worst_cal_err = min(worst_cal_err, cal_err)
        if calibrated and ref_width > 0.0:
            ratios.append(width / ref_width)
        per_level.append(
            {
                "level": round(tau, 4),
                "emp_coverage": round(emp, 4),
                "calibration_error": round(cal_err, 4),
                "mean_interval_width": round(width, 4),
                "ref_width": round(ref_width, 4),
                "calibrated": calibrated,
            }
        )

    ice = float(np.mean(abs_errors))
    sharpness_skill = round(1.0 - float(np.mean(ratios)), 4) if ratios else None

    if ice <= tolerance:
        verdict = "well-calibrated"
    elif worst_cal_err < -tolerance:
        verdict = "over-confident"
    else:
        verdict = "under-confident"

    result = {
        "ice": round(ice, 4),
        "sharpness_skill": sharpness_skill,
        "n_levels": len(levels),
        "n_calibrated_levels": len(ratios),
        "worst_calibration_error": round(float(worst_cal_err), 4),
        "mean_interval_width": round(float(np.mean(widths)), 4),
        "per_level": per_level,
        "verdict": verdict,
        "n_samples": int(y_true.size),
    }

    # Back-compatible single-PICP fields when exactly one level was supplied, so
    # existing single-level consumers keep working unchanged.
    if len(levels) == 1:
        tau = levels[0]
        result["picp"] = per_level[0]["emp_coverage"]
        result["target_coverage"] = round(tau, 4)
        result["calibration_error"] = per_level[0]["calibration_error"]

    return result


def _interval_quantile_grid(
    y_true: np.ndarray,
    intervals: dict[float, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[float], np.ndarray, np.ndarray]:
    """Validate a multi-level interval mapping and build its shared quantile grid.

    Each central level ``tau`` contributes two quantiles of ``F`` — the lower bound
    at ``(1 - tau)/2`` and the upper bound at ``(1 + tau)/2``. Returns
    ``(y_true, levels, alphas, quantiles)`` where ``quantiles[j, i]`` is the
    ``alphas[j]``-quantile for sample ``i``, sorted ascending per sample so the
    implied inverse CDF is non-decreasing (repairing any quantile crossing).

    Raises ``ValueError`` on empty ``y_true``, a level outside ``(0, 1)``, a shape
    mismatch, or an inverted bound. Callers handle the empty-mapping and
    fewer-than-two-levels skip paths.
    """
    y_true = np.asarray(y_true, dtype=float)
    if y_true.size == 0:
        raise ValueError("y_true must be non-empty.")

    levels = sorted(float(t) for t in intervals)
    for tau in levels:
        if not 0.0 < tau < 1.0:
            raise ValueError(f"each interval level must be in (0, 1), got {tau}.")

    # Each central level tau contributes two quantiles of F: the lower bound is
    # the (1 - tau)/2 quantile and the upper bound is the (1 + tau)/2 quantile.
    alpha_values: dict[float, np.ndarray] = {}
    for tau in levels:
        lower, upper = intervals[tau]
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if not (y_true.shape == lower.shape == upper.shape):
            raise ValueError(
                "y_true and each (lower, upper) pair must share the same shape; "
                f"level {tau} got {lower.shape}, {upper.shape} vs {y_true.shape}."
            )
        if np.any(lower > upper):
            raise ValueError(
                f"each lower bound must be <= its upper bound (violated at level {tau})."
            )
        # Distinct central levels never collide on a quantile key: (1 - tau)/2 is
        # strictly below 0.5 and strictly decreasing in tau, (1 + tau)/2 strictly
        # above 0.5 and strictly increasing, so the rounding only stabilises the
        # float key and cannot silently overwrite a different level's quantiles.
        alpha_values[round((1.0 - tau) / 2.0, 10)] = lower
        alpha_values[round((1.0 + tau) / 2.0, 10)] = upper

    alphas = np.array(sorted(alpha_values), dtype=float)
    # Quantile matrix Q[j, i] = the alphas[j]-quantile for sample i; sorting each
    # column ascending repairs any quantile crossing so F^{-1} is non-decreasing.
    quantiles = np.sort(np.vstack([alpha_values[a] for a in alphas]), axis=0)
    return y_true, levels, alphas, quantiles


def crps_from_intervals(
    y_true: np.ndarray,
    intervals: dict[float, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    """
    Continuous Ranked Probability Score (CRPS) from multi-level intervals (RFC #155).

    What it measures
    ----------------
    A single proper score for the *whole* predictive distribution, estimated from
    the same multi-level central intervals :func:`multilevel_interval_coverage`
    consumes. CRPS rewards forecasts that are both calibrated and sharp, so it
    complements ICE (calibration only) and the sharpness proxy (calibration-
    conditioned sharpness only) with one aggregate that a lower value always
    improves.

    How it is estimated
    -------------------
    Via the quantile-loss identity ``CRPS(F, y) = 2 * integral_0^1
    pinball_alpha(F^{-1}(alpha), y) d alpha``. Each central level ``tau`` supplies
    two quantiles of ``F`` — ``F^{-1}((1 - tau) / 2) = lower`` and
    ``F^{-1}((1 + tau) / 2) = upper`` — so a set of interval levels yields a grid
    of quantile levels. The pinball loss is evaluated per sample at each grid
    point and integrated over the quantile level by the trapezoidal rule, then
    averaged across samples.

    Limitations
    -----------
    The estimate is *grid-dependent*: it integrates only over the quantile levels
    the intervals span, so a coarse or narrow grid biases CRPS upward (the tails
    beyond the outermost levels are truncated). Against a closed-form Gaussian a
    3-level grid runs ~17% high while a 19-level grid is <1%. ``n_quantile_levels``
    and ``quantile_level_span`` are returned so the density is visible; prefer a
    dense grid (>= ~9 interval levels) for a trustworthy value. Quantile crossing
    (non-nested intervals) is repaired by sorting each sample's quantiles ascending
    before integration.

    Interpretation guidance
    -----------------------
    Lower is better, in the units of ``y``. CRPS is only comparable across models
    scored on the *same* quantile grid and observations.

    Parameters
    ----------
    y_true : np.ndarray
      Ground-truth continuous targets, shape ``(n_samples,)``.
    intervals : dict[float, tuple[np.ndarray, np.ndarray]] or None
      Maps each nominal central level ``tau in (0, 1)`` to its per-sample
      ``(lower, upper)`` bounds, each shape ``(n_samples,)`` — the same input
      :func:`multilevel_interval_coverage` takes. ``None``/empty, or fewer than
      two distinct quantile levels, degrades gracefully to a ``status="skipped"``
      dict.

    Returns
    -------
    dict
      When intervals are supplied: ``mean_crps``, ``median_crps`` (robust central
      CRPS across samples), ``n_quantile_levels``, ``quantile_level_span``
      (``[alpha_min, alpha_max]`` actually integrated), ``n_levels`` and
      ``n_samples``. When missing/degenerate: a ``status="skipped"`` dict.

    Raises
    ------
    ValueError
      If arrays mismatch ``y_true``'s shape, any ``lower > upper``, any level is
      outside ``(0, 1)``, or ``y_true`` is empty.

    Examples
    --------
    >>> ivs = {0.5: (lo50, hi50), 0.8: (lo80, hi80), 0.95: (lo95, hi95)}
    >>> crps_from_intervals(y_true, ivs)["mean_crps"]
    """
    if not intervals:
        return {
            "status": "skipped",
            "reason": "missing_intervals",
            "details": (
                "CRPS requires a mapping of nominal central levels to per-sample "
                "(lower, upper) bounds. Provide them from a quantile/interval model "
                "to enable the score."
            ),
        }

    y_true, levels, alphas, quantiles = _interval_quantile_grid(y_true, intervals)
    if alphas.size < 2:
        return {
            "status": "skipped",
            "reason": "insufficient_levels",
            "details": (
                "CRPS needs at least two distinct quantile levels to integrate; "
                "supply more interval levels."
            ),
        }

    y = y_true[np.newaxis, :]
    a = alphas[:, np.newaxis]
    # Pinball loss at each (alpha_j, sample_i): (y - q) * (alpha - 1{y < q}).
    pinball = (y - quantiles) * (a - (y < quantiles).astype(float))

    # CRPS_i = 2 * integral_alpha pinball, trapezoidal rule (kept explicit so the
    # estimator is identical on numpy 1.23 and 2.x, where np.trapz was renamed).
    dalpha = np.diff(alphas)[:, np.newaxis]
    segment_means = 0.5 * (pinball[1:] + pinball[:-1])
    crps_per_sample = 2.0 * np.sum(dalpha * segment_means, axis=0)

    return {
        "mean_crps": round(float(np.mean(crps_per_sample)), 4),
        "median_crps": round(float(np.median(crps_per_sample)), 4),
        "n_quantile_levels": int(alphas.size),
        "quantile_level_span": [round(float(alphas[0]), 4), round(float(alphas[-1]), 4)],
        "n_levels": len(levels),
        "n_samples": int(y_true.size),
    }


def _hersbach_terms(
    y_true: np.ndarray, alphas: np.ndarray, quantiles: np.ndarray
) -> tuple[float, float, float]:
    """Hersbach (2000) reliability and potential CRPS on a shared quantile grid.

    For each interval between adjacent quantiles (forecast CDF level ``p = alpha_i``)
    this accumulates, across samples, the mean widths lying below and above the
    observation (``abar``, ``bbar``). With ``g = abar + bbar`` and the observed
    above-frequency ``o = bbar / g``:

      * ``reliability     = sum_i g_i (o_i - p_i)^2``  (calibration miss, >= 0)
      * ``crps_potential  = sum_i g_i o_i (1 - o_i)``  (best CRPS at this resolution)
      * ``crps_recon      = sum_i (abar_i p_i^2 + bbar_i (1 - p_i)^2)``

    Returns ``(reliability, crps_potential, crps_recon)`` with the exact algebraic
    identity ``crps_recon == reliability + crps_potential``.
    """
    p = alphas[:-1]
    qi = quantiles[:-1]
    qip1 = quantiles[1:]
    y = y_true[np.newaxis, :]
    below = np.where(y >= qip1, qip1 - qi, np.where(y <= qi, 0.0, y - qi))
    above = np.where(y >= qip1, 0.0, np.where(y <= qi, qip1 - qi, qip1 - y))
    abar = below.mean(axis=1)
    bbar = above.mean(axis=1)
    g = abar + bbar
    o = np.divide(bbar, g, out=np.zeros_like(g), where=g > 0.0)
    reliability = float(np.sum(g * (o - p) ** 2))
    crps_potential = float(np.sum(g * o * (1.0 - o)))
    pc = p[:, np.newaxis]
    crps_recon = float(np.mean(np.sum(below * pc**2 + above * (1.0 - pc) ** 2, axis=0)))
    return reliability, crps_potential, crps_recon


def _empirical_climatology_quantiles(y_true: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Marginal empirical quantiles of ``y_true`` at each alpha, broadcast to (m, n)."""
    q = np.quantile(y_true, alphas)
    grid: np.ndarray = np.repeat(q[:, np.newaxis], y_true.size, axis=1)
    return grid


def crps_decomposition(
    y_true: np.ndarray,
    intervals: dict[float, tuple[np.ndarray, np.ndarray]] | None = None,
    climatology: dict[float, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    """CRPS reliability / resolution / uncertainty decomposition (Hersbach 2000, RFC #155).

    What it measures
    ----------------
    Splits the aggregate CRPS from :func:`crps_from_intervals` into the three
    interpretable terms of the Hersbach decomposition, so a change in CRPS can be
    attributed to *why* it changed:

    ``CRPS = Reliability - Resolution + Uncertainty``

    * **Reliability** (>= 0) — the calibration miss: how far the forecast's stated
      exceedance probabilities are from the observed frequencies. Lower is better;
      0 means perfectly calibrated on this grid.
    * **Resolution** (higher is better) — how much the forecast improves on
      climatology by varying with the case. ``Uncertainty - crps_potential``.
    * **Uncertainty** — the CRPS of the climatological (marginal) forecast; a
      property of the data, not the model, and the ceiling a zero-resolution
      forecast pays.

    How it is estimated
    -------------------
    On the same multi-level interval grid as :func:`crps_from_intervals`. Adjacent
    quantiles define intervals with forecast CDF level ``p = alpha``; averaging the
    below/above widths across samples yields ``reliability`` and ``crps_potential``
    with the exact identity ``crps == reliability + crps_potential``. ``uncertainty``
    is the same construction applied to the climatological forecast (by default the
    marginal empirical quantiles of ``y_true``), and ``resolution = uncertainty -
    crps_potential``.

    Limitations
    -----------
    ``crps`` here is the decomposition-consistent (step-CDF) reconstruction; it
    agrees with :func:`crps_from_intervals`'s trapezoidal ``mean_crps`` to within
    the grid-interpolation gap (a few percent at ~19 levels, tightening as the grid
    densifies). Reliability and resolution are *dataset-level* properties — they
    need the joint distribution over many forecasts and are not per-sample
    averageable. ``resolution`` can go slightly negative for a forecast that is
    worse than climatology.

    Parameters
    ----------
    y_true : np.ndarray
      Ground-truth continuous targets, shape ``(n_samples,)``.
    intervals : dict[float, tuple[np.ndarray, np.ndarray]] or None
      Nominal central levels mapped to per-sample ``(lower, upper)`` bounds — the
      same input :func:`crps_from_intervals` takes. ``None``/empty or fewer than
      two distinct quantile levels degrades to a ``status="skipped"`` dict.
    climatology : dict[...] or None
      Optional reference forecast on the **same** interval levels; when ``None`` the
      marginal empirical quantiles of ``y_true`` are used.

    Returns
    -------
    dict
      When intervals are supplied: ``reliability``, ``resolution``, ``uncertainty``,
      ``crps``, ``crps_potential``, ``n_quantile_levels``, ``n_levels`` and
      ``n_samples``. When missing/degenerate: a ``status="skipped"`` dict.

    Raises
    ------
    ValueError
      If arrays mismatch ``y_true``'s shape, any ``lower > upper``, any level is
      outside ``(0, 1)``, ``y_true`` is empty, or ``climatology`` is supplied on
      different levels than ``intervals``.

    Examples
    --------
    >>> ivs = {0.5: (lo50, hi50), 0.8: (lo80, hi80), 0.95: (lo95, hi95)}
    >>> d = crps_decomposition(y_true, ivs)
    >>> d["reliability"], d["resolution"], d["uncertainty"]
    """
    if not intervals:
        return {
            "status": "skipped",
            "reason": "missing_intervals",
            "details": (
                "CRPS decomposition requires a mapping of nominal central levels to "
                "per-sample (lower, upper) bounds. Provide them from a "
                "quantile/interval model to enable it."
            ),
        }

    y_true, levels, alphas, quantiles = _interval_quantile_grid(y_true, intervals)
    if alphas.size < 2:
        return {
            "status": "skipped",
            "reason": "insufficient_levels",
            "details": (
                "CRPS decomposition needs at least two distinct quantile levels; "
                "supply more interval levels."
            ),
        }

    reliability, crps_potential, crps_recon = _hersbach_terms(y_true, alphas, quantiles)

    # Uncertainty = CRPS of the climatological forecast on the same grid. Default
    # climatology = the marginal empirical quantiles of y_true (constant across
    # samples); callers may pass an explicit reference distribution.
    if climatology is None:
        clim_quantiles = _empirical_climatology_quantiles(y_true, alphas)
    else:
        _, _, clim_alphas, clim_quantiles = _interval_quantile_grid(y_true, climatology)
        if not np.array_equal(np.round(clim_alphas, 10), np.round(alphas, 10)):
            raise ValueError(
                "climatology must be supplied on the same interval levels as intervals."
            )
    _, _, uncertainty = _hersbach_terms(y_true, alphas, clim_quantiles)
    resolution = uncertainty - crps_potential

    return {
        "reliability": round(reliability, 4),
        "resolution": round(resolution, 4),
        "uncertainty": round(uncertainty, 4),
        "crps": round(crps_recon, 4),
        "crps_potential": round(crps_potential, 4),
        "n_quantile_levels": int(alphas.size),
        "n_levels": len(levels),
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
    ranks: np.ndarray = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    # Resolve ties to their average rank.
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size, dtype=float)
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    averaged: np.ndarray = avg[inv]
    return averaged
