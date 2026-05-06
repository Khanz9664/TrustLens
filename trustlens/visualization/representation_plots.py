"""
trustlens.visualization.representation_plots.
=============================================
Visualizations for representation / embedding analysis.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


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


def plot_embedding_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    silhouette_score: float | None = None,
    method: str = "umap",
    n_max: int = 5000,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Project high-dimensional embeddings to 2D and render a class-colored scatter plot.

    Projection strategy supports optional dependency fallback:
    ``umap`` -> ``tsne`` -> ``pca``.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels)

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array with shape (n_samples, n_features).")
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if len(embeddings) != len(labels):
        raise ValueError("embeddings and labels must have the same number of samples.")
    if n_max < 2:
        raise ValueError("n_max must be >= 2.")

    method = method.lower()
    allowed = {"umap", "tsne", "pca"}
    if method not in allowed:
        raise ValueError(f"Unknown method '{method}'. Use 'umap', 'tsne', or 'pca'.")

    emb_used = embeddings
    labels_used = labels
    if len(embeddings) > n_max:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embeddings), n_max, replace=False)
        emb_used = embeddings[idx]
        labels_used = labels[idx]
        print(
            f"[TrustLens] Subsampled embeddings from {len(embeddings)} to {n_max} for 2D plotting."
        )

    methods_to_try = {
        "umap": ["umap", "tsne", "pca"],
        "tsne": ["tsne", "pca"],
        "pca": ["pca"],
    }[method]

    coords_2d = None
    used_method = None
    last_error = None

    for m in methods_to_try:
        try:
            if m == "umap":
                import umap  # type: ignore

                reducer = umap.UMAP(n_components=2, random_state=42)
                coords_2d = reducer.fit_transform(emb_used)
            elif m == "tsne":
                from sklearn.manifold import TSNE

                reducer = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
                coords_2d = reducer.fit_transform(emb_used)
            else:
                from sklearn.decomposition import PCA

                reducer = PCA(n_components=2, random_state=42)
                coords_2d = reducer.fit_transform(emb_used)

            used_method = m
            break
        except Exception as exc:  # pragma: no cover - branch depends on optional deps/runtime.
            last_error = exc

    if coords_2d is None or used_method is None:
        raise RuntimeError(
            f"Failed to compute 2D projection with method '{method}'."
        ) from last_error

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    unique_labels = np.unique(labels_used)
    cmap = plt.get_cmap("tab10")
    for i, cls in enumerate(unique_labels):
        mask = labels_used == cls
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            s=24,
            alpha=0.8,
            color=cmap(i % 10),
            label=str(cls),
            edgecolors="none",
        )

    title = f"Embedding 2D Projection ({used_method.upper()})"
    if silhouette_score is not None and np.isfinite(silhouette_score):
        title += f" | Silhouette={silhouette_score:.4f}"

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(title="Class", loc="best", frameon=True)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show and "agg" not in plt.get_backend().lower():
        plt.show()

    plt.close(fig)
    return fig
