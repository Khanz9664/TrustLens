"""
trustlens.visualization.regression_plots.
=========================================
Diagnostic plots for the regression reliability path.

Two complementary views of a regression model's errors:

* :func:`plot_residuals` — residuals (``y_true - y_pred``) against the predicted
  value. This is the canonical way to spot **heteroscedasticity** (a fan or curve
  in the residual cloud) and **systematic bias** (the cloud drifting off the zero
  line). An optional prediction-interval band can be overlaid.
* :func:`plot_error_distribution` — a histogram of the signed errors against a
  fitted-normal reference, so **skew** and **heavy tails** are visible at a glance.

Both mirror the conventions of the classification plot modules: brand styling via
:func:`trustlens.visualization.style.apply_style`, an explicit ``save_path`` /
``show`` contract, and a returned :class:`matplotlib.figure.Figure`.
"""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from trustlens.visualization.style import apply_style


def _as_1d(name: str, values: np.ndarray) -> np.ndarray:
    """Coerce to a 1-D float array, flattening a singleton ``(n, 1)`` column and
    rejecting a true multi-output shape with a clear error."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}.")
    return cast(np.ndarray, arr)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prediction_intervals: tuple[np.ndarray, np.ndarray] | None = None,
    title: str = "Residuals vs. Predicted",
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot residuals (``y_true - y_pred``) against the predicted value.

    The plot shows:

    * **Residual scatter** — one point per observation; a horizontal, evenly
      spread band around zero indicates well-behaved errors.
    * **Zero-error reference** — a dashed line at ``residual = 0``.
    * **Optional prediction-interval band** — when ``prediction_intervals`` is
      supplied, the interval width is shown in residual space (``lower - pred`` to
      ``upper - pred``), so you can read off whether the band actually brackets the
      residual cloud.
    * **Diagnostics annotation** — mean residual (bias) and the correlation between
      ``|residual|`` and the prediction (a positive value flags heteroscedasticity).

    Parameters
    ----------
    y_true : np.ndarray
        Observed target values, shape ``(n,)`` (a singleton ``(n, 1)`` is accepted).
    y_pred : np.ndarray
        Point predictions, same shape as ``y_true``.
    prediction_intervals : tuple[np.ndarray, np.ndarray], optional
        ``(lower, upper)`` per-observation interval bounds. When given, the band is
        overlaid in residual space.
    title : str
        Figure title.
    save_path : str, optional
        If given, the figure is written to this path.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        When ``y_true``/``y_pred`` are empty or their shapes disagree.
    """
    yt = _as_1d("y_true", y_true)
    yp = _as_1d("y_pred", y_pred)
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {yt.shape} vs {yp.shape}."
        )
    if yt.size == 0:
        raise ValueError("y_true and y_pred must be non-empty.")

    residuals = yt - yp
    bias = float(np.mean(residuals))
    # Heteroscedasticity signal: correlation of |residual| with the prediction.
    if yt.size >= 2 and np.std(yp) > 0 and np.std(np.abs(residuals)) > 0:
        hetero = float(np.corrcoef(yp, np.abs(residuals))[0, 1])
    else:
        hetero = float("nan")

    with apply_style() as theme:
        blue = theme.brand["blue"]
        gray = theme.brand["muted_gray"]
        neutral = theme.semantic["neutral"]

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

        if prediction_intervals is not None:
            try:
                lower_raw, upper_raw = prediction_intervals
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "prediction_intervals must be a (lower, upper) pair of arrays."
                ) from exc
            lower = _as_1d("prediction_intervals[0]", lower_raw)
            upper = _as_1d("prediction_intervals[1]", upper_raw)
            if lower.shape != yp.shape or upper.shape != yp.shape:
                raise ValueError(
                    "prediction_intervals bounds must match y_pred's shape, got "
                    f"{lower.shape} / {upper.shape} vs {yp.shape}."
                )
            order = np.argsort(yp)
            ax.fill_between(
                yp[order],
                (lower - yp)[order],
                (upper - yp)[order],
                alpha=0.15,
                color=blue,
                label="Prediction interval",
            )

        ax.scatter(
            yp,
            residuals,
            s=18,
            alpha=0.6,
            color=blue,
            edgecolor=neutral["edge"],
            linewidth=0.3,
            label="Residuals",
        )
        ax.axhline(0, color=gray, lw=1.5, linestyle="--", label="Zero error")

        annotation_lines = [f"bias  = {bias:.4g}"]
        if not np.isnan(hetero):
            annotation_lines.append(f"corr(|r|, ŷ) = {hetero:+.3f}")
        ax.text(
            0.04,
            0.96,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=neutral["annotation_face"],
                edgecolor=neutral["annotation_edge"],
                alpha=0.9,
            ),
            fontfamily="monospace",
        )

        ax.set_xlabel("Predicted value", fontsize=12)
        ax.set_ylabel("Residual (true − predicted)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=theme.grid["alpha"])

        if save_path:
            fig.savefig(save_path, dpi=theme.fig_defaults["savefig_dpi"], bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)
        return fig


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    bins: int = 30,
    title: str = "Error Distribution",
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a histogram of signed errors (``y_true - y_pred``) against a fitted normal.

    The plot shows:

    * **Error histogram** — density of the signed errors.
    * **Fitted-normal overlay** — a Gaussian with the errors' own mean and standard
      deviation, so departures (skew, heavy tails, bimodality) are obvious.
    * **Zero-error reference** — a dashed vertical line.
    * **Metric annotation** — MAE and RMSE.

    Parameters
    ----------
    y_true : np.ndarray
        Observed target values, shape ``(n,)`` (a singleton ``(n, 1)`` is accepted).
    y_pred : np.ndarray
        Point predictions, same shape as ``y_true``.
    bins : int
        Number of histogram bins (must be >= 1).
    title : str
        Figure title.
    save_path : str, optional
        If given, the figure is written to this path.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        When inputs are empty, shapes disagree, or ``bins < 1``.
    """
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")

    yt = _as_1d("y_true", y_true)
    yp = _as_1d("y_pred", y_pred)
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {yt.shape} vs {yp.shape}."
        )
    if yt.size == 0:
        raise ValueError("y_true and y_pred must be non-empty.")

    errors = yt - yp
    mu = float(np.mean(errors))
    sigma = float(np.std(errors))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    with apply_style() as theme:
        blue = theme.brand["blue"]
        orange = theme.brand["orange"]
        gray = theme.brand["muted_gray"]
        neutral = theme.semantic["neutral"]

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

        ax.hist(
            errors,
            bins=bins,
            density=True,
            color=blue,
            alpha=0.7,
            edgecolor=neutral["edge"],
            linewidth=0.5,
            label="Errors",
        )

        if sigma > 0:
            xs = np.linspace(float(errors.min()), float(errors.max()), 200)
            pdf = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
            ax.plot(
                xs,
                pdf,
                color=orange,
                lw=2.0,
                label=f"Normal fit (μ={mu:.3g}, σ={sigma:.3g})",
            )

        ax.axvline(0, color=gray, lw=1.5, linestyle="--", label="Zero error")

        ax.text(
            0.04,
            0.96,
            f"MAE  = {mae:.4g}\nRMSE = {rmse:.4g}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=neutral["annotation_face"],
                edgecolor=neutral["annotation_edge"],
                alpha=0.9,
            ),
            fontfamily="monospace",
        )

        ax.set_xlabel("Error (true − predicted)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=theme.grid["alpha"])

        if save_path:
            fig.savefig(save_path, dpi=theme.fig_defaults["savefig_dpi"], bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)
        return fig
