"""Multi-model radar comparison plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from trustlens.visualization.style import apply_style, get_categorical_colors


def plot_radar_comparison(
    metrics_dict: dict[str, dict[str, float]],
    title: str = "Model Comparison",
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a radar chart comparing dimensions across multiple models.

    Each model is drawn as a filled polygon on a polar axis. Axes correspond
    to metric names (e.g., calibration, failure, bias) and values are scores.

    Parameters
    ----------
    metrics_dict : dict[str, dict[str, float]]
      Mapping of model name to metric scores, e.g.
      ``{"Random Forest": {"calibration": 82.4, "failure": 76.1}}``.
      All models should share the same metric keys; axis labels are taken
      from the first model entry.
    title : str
      Plot title.
    save_path : str, optional
      If provided, saves the figure to this path.
    show : bool
      If True, calls ``plt.show()`` on interactive backends.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        When ``metrics_dict`` is empty.
    """
    if not metrics_dict:
        raise ValueError("metrics_dict must not be empty")

    labels = list(next(iter(metrics_dict.values())).keys())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    with apply_style() as theme:
        fig, ax = plt.subplots(
            figsize=(7, 7),
            subplot_kw={"projection": "polar"},
            constrained_layout=True,
        )

        colors = get_categorical_colors(len(metrics_dict), theme=theme)

        for (model_name, scores), color in zip(metrics_dict.items(), colors):
            values = [scores[label] for label in labels]
            values = values + values[:1]

            ax.plot(angles, values, color=color, label=model_name, linewidth=2)
            ax.fill(angles, values, color=color, alpha=0.2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=theme.grid["alpha"])
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

        if save_path:
            fig.savefig(save_path, dpi=theme.fig_defaults["savefig_dpi"], bbox_inches="tight")
        if show:
            plt.show()

        plt.close(fig)
        return fig
