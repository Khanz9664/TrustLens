"""
tests/test_representation.py.
=============================
Unit tests for trustlens.metrics.representation.
"""

import numpy as np
import pytest

from trustlens.metrics.representation import (
    centered_kernel_alignment,
    embedding_separability,
)
from trustlens.visualization.representation_plots import plot_embedding_2d


class TestEmbeddingSeparability:
    def test_basic_output_keys(self):
        embeddings = np.random.randn(50, 10)
        y_true = np.array([0] * 25 + [1] * 25)
        result = embedding_separability(embeddings, y_true)

        expected_keys = {
            "silhouette_score",
            "within_class_distance",
            "between_class_distance",
            "separability_ratio",
            "n_samples_used",
            "embedding_dim",
        }
        assert expected_keys.issubset(result.keys())

    def test_well_separated_has_high_silhouette(self):
        """Classes far apart in embedding space → high silhouette."""
        class_a = np.random.randn(50, 5) + np.array([10, 0, 0, 0, 0])
        class_b = np.random.randn(50, 5) - np.array([10, 0, 0, 0, 0])
        embeddings = np.vstack([class_a, class_b])
        y_true = np.array([0] * 50 + [1] * 50)
        result = embedding_separability(embeddings, y_true)
        assert result["silhouette_score"] > 0.5

    def test_embedding_dim_correct(self):
        embeddings = np.random.randn(30, 16)
        y_true = np.array([0] * 15 + [1] * 15)
        result = embedding_separability(embeddings, y_true)
        assert result["embedding_dim"] == 16

    def test_separability_ratio_positive(self):
        embeddings = np.random.randn(40, 8)
        y_true = np.array([0] * 20 + [1] * 20)
        result = embedding_separability(embeddings, y_true)
        # Ratio can be 'inf' for very tight clusters
        assert result["separability_ratio"] >= 0


class TestCenteredKernelAlignment:
    def test_self_similarity_is_one(self):
        """CKA(X, X) should be exactly 1.0."""
        X = np.random.randn(20, 5)
        cka = centered_kernel_alignment(X, X)
        assert cka == pytest.approx(1.0, rel=1e-5)

    def test_range_zero_to_one(self):
        X = np.random.randn(20, 5)
        Y = np.random.randn(20, 8)
        cka = centered_kernel_alignment(X, Y)
        assert 0.0 <= cka <= 1.0

    def test_shape_mismatch_raises(self):
        X = np.random.randn(20, 5)
        Y = np.random.randn(25, 5)
        with pytest.raises(ValueError, match="same number of samples"):
            centered_kernel_alignment(X, Y)

    def test_orthogonal_representations_low_cka(self):
        """Near-orthogonal representations should have low CKA."""
        rng = np.random.default_rng(99)
        X = rng.standard_normal((50, 10))
        # Make Y completely independent of X
        Y = rng.standard_normal((50, 10))
        cka = centered_kernel_alignment(X, Y)
        # We can't guarantee exactly 0, but it should be < 0.9
        assert cka < 0.9


class TestPlotEmbedding2D:
    def test_plot_embedding_2d_returns_figure(self):
        rng = np.random.default_rng(0)
        embeddings = rng.standard_normal((120, 8))
        labels = np.array([0] * 60 + [1] * 60)
        fig = plot_embedding_2d(embeddings, labels, method="pca", show=False)
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_plot_embedding_2d_method_fallback_to_pca(self, monkeypatch):
        rng = np.random.default_rng(1)
        embeddings = rng.standard_normal((90, 6))
        labels = np.array([0] * 45 + [1] * 45)

        from sklearn.manifold import TSNE

        def _raise_on_tsne(self, X):
            raise RuntimeError("forced tsne failure")

        monkeypatch.setattr(TSNE, "fit_transform", _raise_on_tsne)
        fig = plot_embedding_2d(embeddings, labels, method="tsne", show=False)
        assert fig is not None

    def test_plot_embedding_2d_subsampling_respects_n_max(self, capsys):
        rng = np.random.default_rng(2)
        embeddings = rng.standard_normal((250, 12))
        labels = np.array([0] * 125 + [1] * 125)
        fig = plot_embedding_2d(embeddings, labels, method="pca", n_max=80, show=False)
        captured = capsys.readouterr()
        assert "Subsampled embeddings" in captured.out
        assert fig is not None

    def test_plot_embedding_2d_save_path(self, tmp_path):
        rng = np.random.default_rng(3)
        embeddings = rng.standard_normal((80, 10))
        labels = np.array([0] * 40 + [1] * 40)
        save_path = tmp_path / "embedding_2d.png"
        _ = plot_embedding_2d(
            embeddings, labels, method="pca", save_path=str(save_path), show=False
        )
        assert save_path.exists()

    def test_plot_embedding_2d_show_false(self):
        rng = np.random.default_rng(4)
        embeddings = rng.standard_normal((60, 7))
        labels = np.array([0] * 30 + [1] * 30)
        fig = plot_embedding_2d(embeddings, labels, method="pca", show=False)
        assert fig is not None

    def test_plot_embedding_2d_invalid_method_raises(self):
        rng = np.random.default_rng(5)
        embeddings = rng.standard_normal((40, 5))
        labels = np.array([0] * 20 + [1] * 20)
        with pytest.raises(ValueError, match="Unknown method"):
            plot_embedding_2d(embeddings, labels, method="invalid", show=False)
