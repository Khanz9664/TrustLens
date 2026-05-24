"""
trustlens.metrics.representation.
=================================
Representation space analysis.

Probes the geometry of learned embedding spaces to understand:
* Whether classes are well-separated
* How similar two representation layers are (CKA)
* Whether cluster structure aligns with ground-truth labels

Metrics implemented
-------------------
* ``embedding_separability``  — silhouette score + within/between class distance
* ``centered_kernel_alignment`` — measures representational similarity between
  two sets of embeddings (e.g., two layers)

References
----------
* Kornblith, S., et al. (2019). Similarity of Neural Network Representations
  Revisited. ICML.
* Rousseeuw, P. (1987). Silhouettes: A graphical aid to the interpretation
  and validation of cluster analysis. Journal of Computational and Applied
  Mathematics.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from sklearn.metrics import silhouette_score

__all__ = [
    "embedding_separability",
    "centered_kernel_alignment",
]

def embedding_separability(
    embeddings: np.ndarray,
    y_true: np.ndarray,
    metric: str = "euclidean",
    sample_limit: int = 5000,
) -> dict:
    """
    Measure how well class embeddings are separated in latent space.

    Uses the silhouette score as the primary separability measure, augmented
    with within-class and between-class mean distances.

    Parameters
    ----------
    embeddings : np.ndarray
      Latent representations, shape (n_samples, embedding_dim).
    y_true : np.ndarray
      Ground-truth labels, shape (n_samples,).
    metric : str
      Distance metric passed to ``silhouette_score``. Default ``"euclidean"``.
    sample_limit : int
      Maximum samples used for silhouette computation (avoids O(n²) cost).
      A random subsample is drawn when ``len(embeddings) > sample_limit``.

    Returns
    -------
    dict with keys:
      * ``silhouette_score``    — in [-1, 1]; 1.0 = perfect separation
      * ``within_class_distance``  — mean pairwise distance within classes
      * ``between_class_distance`` — mean pairwise distance across classes
      * ``separability_ratio``   — between / within (> 1 preferred)

    Examples
    --------
    >>> sep = embedding_separability(embeddings, y_true)
    >>> print(f"Silhouette: {sep['silhouette_score']:.3f}")
    """
    embeddings = np.asarray(embeddings, dtype=float)
    y_true = np.asarray(y_true)

    n = len(embeddings)

    # Subsample for large datasets
    if n > sample_limit:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, sample_limit, replace=False)
        embeddings_ss = embeddings[idx]
        y_true_ss = y_true[idx]
    else:
        embeddings_ss = embeddings
        y_true_ss = y_true

    # Silhouette score requires at least 2 distinct labels
    n_classes = len(np.unique(y_true_ss))
    if n_classes < 2:
        sil = float("nan")
    else:
        sil = float(silhouette_score(embeddings_ss, y_true_ss, metric=metric))

    # Within-class and between-class distances (sampled)
    within_dists: list = []
    between_dists: list = []
    classes = np.unique(y_true_ss)

    # Limit pair-wise computation to a smaller random subset for speed
    max_pairs = 200
    rng = np.random.default_rng(0)

    for cls in classes:
        in_cls = embeddings_ss[y_true_ss == cls]
        out_cls = embeddings_ss[y_true_ss != cls]

        if len(in_cls) >= 2:
            pairs = min(max_pairs, len(in_cls) * (len(in_cls) - 1) // 2)
            idx_a = rng.integers(0, len(in_cls), pairs)
            idx_b = rng.integers(0, len(in_cls), pairs)
            diff = in_cls[idx_a] - in_cls[idx_b]
            within_dists.extend(np.linalg.norm(diff, axis=1).tolist())

        if len(in_cls) >= 1 and len(out_cls) >= 1:
            pairs = min(max_pairs, len(in_cls) * len(out_cls))
            idx_a = rng.integers(0, len(in_cls), pairs)
            idx_b = rng.integers(0, len(out_cls), pairs)
            diff = in_cls[idx_a] - out_cls[idx_b]
            between_dists.extend(np.linalg.norm(diff, axis=1).tolist())

    within_mean = float(np.mean(within_dists)) if within_dists else 0.0
    between_mean = float(np.mean(between_dists)) if between_dists else 0.0
    sep_ratio = round(between_mean / within_mean, 4) if within_mean > 0 else float("inf")

    return {
        "silhouette_score": round(sil, 4),
        "within_class_distance": round(within_mean, 4),
        "between_class_distance": round(between_mean, 4),
        "separability_ratio": sep_ratio,
        "n_samples_used": len(embeddings_ss),
        "embedding_dim": embeddings.shape[1],
    }


def centered_kernel_alignment(
    X: np.ndarray,
    Y: np.ndarray,
) -> float:
    r"""
    Compute Centered Kernel Alignment (CKA) between two representation matrices.

    CKA is a representational similarity metric that is invariant to
    orthogonal transformations and isotropic scaling, making it suitable
    for comparing representations across architectures and layers.

    .. math::
      \\text{CKA}(K, L) =
      \\frac{\\text{HSIC}(K, L)}{
        \\sqrt{\\text{HSIC}(K, K) \\cdot \\text{HSIC}(L, L)}}

    Parameters
    ----------
    X : np.ndarray
      First representation matrix, shape (n_samples, d1).
    Y : np.ndarray
      Second representation matrix, shape (n_samples, d2).

    Returns
    -------
    float
      CKA similarity score in [0, 1]. Higher → more similar representations.

    Raises
    ------
    ValueError
      If ``X`` and ``Y`` have different numbers of samples.

    Examples
    --------
    >>> cka = centered_kernel_alignment(layer1_embeddings, layer2_embeddings)
    >>> print(f"CKA similarity: {cka:.3f}")
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples, got {X.shape[0]} and {Y.shape[0]}."
        )

    # Linear kernel matrices
    K = X @ X.T
    L = Y @ Y.T

    # Center the kernel matrices
    K = _center_kernel(K)
    L = _center_kernel(L)

    hsic_kl = _hsic(K, L)
    hsic_kk = _hsic(K, K)
    hsic_ll = _hsic(L, L)

    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-12:
        return 0.0

    return float(np.clip(hsic_kl / denom, 0.0, 1.0))


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Double-center a kernel matrix."""
    n = K.shape[0]
    H = cast(np.ndarray, np.eye(n) - np.ones((n, n)) / n)
    return cast(np.ndarray, H @ K @ H)


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Biased HSIC estimator."""
    n = K.shape[0]
    return float(cast(float, np.trace(K @ L)) / ((n - 1) ** 2))
