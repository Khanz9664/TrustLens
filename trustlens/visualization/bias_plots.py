"""
trustlens.visualization.bias_plots.
====================================
Visualizations for bias and fairness analysis.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_class_distribution(
    imbalance_data: dict,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Bar chart of class frequency distribution.

    Highlights the majority and minority class and annotates the imbalance
    ratio so practitioners can immediately assess dataset balance.

    Parameters
    ----------
    imbalance_data : dict
      Output from ``class_imbalance_report()``.
    save_path : str, optional
      If provided, saves figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    PALETTE = [
        "#4B8BF5",
        "#F5784B",
        "#34C759",
        "#AF52DE",
        "#FF9F0A",
        "#FF2D55",
        "#5AC8FA",
        "#FF6B35",
    ]

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    class_counts = imbalance_data["class_counts"]
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    total = sum(counts)

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(classes))]
    bars = ax.bar(
        [str(c) for c in classes],
        counts,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        alpha=0.85,
    )

    # Annotate bars with percentage
    for bar, count in zip(bars, counts):
        pct = 100 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Imbalance ratio annotation
    ratio = imbalance_data["imbalance_ratio"]
    ax.text(
        0.97,
        0.97,
        f"Imbalance ratio = {ratio:.2f}×",
        transform=ax.transAxes,
        fontsize=11,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
        fontfamily="monospace",
    )

    ax.set_xlabel("Class Label", fontsize=12)
    ax.set_ylabel("Sample Count", fontsize=12)
    ax.set_title("Class Distribution", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.35)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        if "agg" not in plt.get_backend().lower():
            plt.show()

    plt.close(fig)
    return fig
