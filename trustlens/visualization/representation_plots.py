"""
trustlens.visualization.representation_plots.
=============================================
Visualizations for representation / embedding analysis.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_embedding_separability(
    sep_data: dict,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Render a compact metric card for embedding separability results.

    Displays silhouette score, within/between class distances, and the
    separability ratio as a visual scorecard — optimized for quick
    interpretability in reports.

    Parameters
    ----------
    sep_data : dict
      Output from ``embedding_separability()``.
    save_path : str, optional
      If provided, saves figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    sil = sep_data.get("silhouette_score", float("nan"))
    within = sep_data.get("within_class_distance", 0.0)
    between = sep_data.get("between_class_distance", 0.0)
    ratio = sep_data.get("separability_ratio", 0.0)
    n_used = sep_data.get("n_samples_used", 0)
    emb_dim = sep_data.get("embedding_dim", 0)

    # Color-code silhouette score
    if sil >= 0.5:
        sil_color = "#34C759"
    elif sil >= 0.25:
        sil_color = "#FF9F0A"
    else:
        sil_color = "#FF3B30"

    # Title
    ax.text(
        5, 9.3, "Embedding Separability", ha="center", va="center", fontsize=14, fontweight="bold"
    )
    ax.text(
        5,
        8.6,
        f"n_samples={n_used:,} | dim={emb_dim}",
        ha="center",
        va="center",
        fontsize=10,
        color="#666666",
    )

    # Metric cards
    def draw_card(x, y, label, value, color, val_fmt="{:.4f}"):
        rect = plt.Rectangle(
            (x - 1.8, y - 1), 3.6, 2, facecolor=color, alpha=0.12, edgecolor=color, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.45, label, ha="center", va="center", fontsize=9, color="#444444")
        ax.text(
            x,
            y - 0.35,
            val_fmt.format(value),
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=color,
        )

    draw_card(2.0, 6.5, "Silhouette Score", sil, sil_color)
    draw_card(5.0, 6.5, "Within-Class Dist.", within, "#4B8BF5")
    draw_card(8.0, 6.5, "Between-Class Dist.", between, "#AF52DE")

    # Separability ratio
    ratio_color = "#34C759" if ratio >= 1.5 else "#FF9F0A" if ratio >= 1.0 else "#FF3B30"
    ax.text(
        5,
        4.5,
        "Separability Ratio (Between / Within)",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )
    ax.text(
        5,
        3.5,
        f"{ratio:.3f}×",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=ratio_color,
    )

    guidance = "> 1.5: Well separated  |  1.0–1.5: Moderate  |  < 1.0: Poor"
    ax.text(5, 2.5, guidance, ha="center", va="center", fontsize=9, color="#888888", style="italic")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        if "agg" not in plt.get_backend().lower():
            plt.show()

    plt.close(fig)
    return fig
