"""Tests for the noise injection engine."""

import numpy as np
import pytest

from topowarp.generators import concentric_annuli
from topowarp.noise import SUPPORTED_DISTRIBUTIONS, apply_feature_noise, apply_label_noise


# ---------------------------------------------------------------------------
# Feature noise
# ---------------------------------------------------------------------------


class TestFeatureNoise:
    def setup_method(self):
        self.X, self.y = concentric_annuli(200, 5, seed=0)

    def test_output_shape_unchanged(self):
        X_noisy = apply_feature_noise(self.X, "gaussian", scale=0.5, seed=0)
        assert X_noisy.shape == self.X.shape

    def test_zero_scale_returns_copy(self):
        X_noisy = apply_feature_noise(self.X, "gaussian", scale=0.0, seed=0)
        np.testing.assert_array_equal(X_noisy, self.X)
        assert X_noisy is not self.X

    def test_nonzero_scale_modifies_data(self):
        X_noisy = apply_feature_noise(self.X, "gaussian", scale=1.0, seed=0)
        assert not np.array_equal(X_noisy, self.X)

    @pytest.mark.parametrize("dist", SUPPORTED_DISTRIBUTIONS)
    def test_all_distributions_run(self, dist):
        X_noisy = apply_feature_noise(self.X, dist, scale=0.5, seed=0)
        assert X_noisy.shape == self.X.shape

    def test_sparsity_affects_subset(self):
        X_noisy = apply_feature_noise(self.X, "gaussian", scale=1.0, sparsity=0.4, seed=0)
        diffs = np.abs(X_noisy - self.X).sum(axis=0)
        n_affected = np.sum(diffs > 0)
        # 5 dims * 0.4 = 2 columns
        assert n_affected == 2

    def test_full_sparsity_affects_all(self):
        X_noisy = apply_feature_noise(self.X, "gaussian", scale=1.0, sparsity=1.0, seed=0)
        diffs = np.abs(X_noisy - self.X).sum(axis=0)
        assert np.all(diffs > 0)

    def test_determinism(self):
        a = apply_feature_noise(self.X, "laplacian", scale=0.3, seed=99)
        b = apply_feature_noise(self.X, "laplacian", scale=0.3, seed=99)
        np.testing.assert_array_equal(a, b)

    def test_invalid_distribution(self):
        with pytest.raises(ValueError, match="distribution"):
            apply_feature_noise(self.X, "poisson", scale=0.1)

    def test_exponential_mean_centered(self):
        X_noisy = apply_feature_noise(self.X, "exponential", scale=0.5, seed=0)
        noise = X_noisy - self.X
        # Each affected column should have near-zero mean
        np.testing.assert_allclose(noise.mean(axis=0), 0.0, atol=0.1)

    def test_erlang_mean_centered(self):
        X_noisy = apply_feature_noise(self.X, "erlang", scale=0.5, seed=0)
        noise = X_noisy - self.X
        np.testing.assert_allclose(noise.mean(axis=0), 0.0, atol=0.1)

    def test_exponential_noise_is_not_symmetric(self):
        # After mean-centering, exponential noise should still be right-skewed
        X_noisy = apply_feature_noise(self.X, "exponential", scale=1.0, seed=0)
        noise = (X_noisy - self.X)[:, 0]
        from scipy.stats import skew
        assert skew(noise) > 0.5, "Exponential noise should retain positive skew after centering"

    def test_negative_scale(self):
        with pytest.raises(ValueError, match="scale"):
            apply_feature_noise(self.X, "gaussian", scale=-1)

    def test_bad_sparsity(self):
        with pytest.raises(ValueError, match="sparsity"):
            apply_feature_noise(self.X, "gaussian", scale=0.1, sparsity=0)
        with pytest.raises(ValueError, match="sparsity"):
            apply_feature_noise(self.X, "gaussian", scale=0.1, sparsity=1.5)


# ---------------------------------------------------------------------------
# Label noise
# ---------------------------------------------------------------------------


class TestLabelNoise:
    def setup_method(self):
        self.X, self.y = concentric_annuli(200, 4, n_rings=3, seed=0)

    def test_zero_flip_returns_copy(self):
        y_noisy, mask = apply_label_noise(self.y, flip_prob=0.0, seed=0)
        np.testing.assert_array_equal(y_noisy, self.y)
        assert not mask.any()

    def test_flip_count_matches_prob(self):
        y_noisy, mask = apply_label_noise(self.y, flip_prob=0.1, seed=0)
        expected = int(round(200 * 0.1))
        assert mask.sum() == expected

    def test_flipped_labels_differ(self):
        y_noisy, mask = apply_label_noise(self.y, flip_prob=0.15, seed=0)
        # Every flipped index should have a different label
        assert np.all(y_noisy[mask] != self.y[mask])

    def test_unflipped_labels_unchanged(self):
        y_noisy, mask = apply_label_noise(self.y, flip_prob=0.15, seed=0)
        np.testing.assert_array_equal(y_noisy[~mask], self.y[~mask])

    def test_boundary_targeting_requires_X(self):
        with pytest.raises(ValueError, match="X must be provided"):
            apply_label_noise(self.y, flip_prob=0.1, targeting="boundary")

    def test_boundary_targeting_runs(self):
        y_noisy, mask = apply_label_noise(
            self.y, flip_prob=0.1, targeting="boundary", X=self.X, seed=0
        )
        assert mask.sum() == int(round(200 * 0.1))
        assert np.all(y_noisy[mask] != self.y[mask])

    def test_boundary_targets_near_boundary(self):
        """Boundary-targeted flips should cluster near class boundaries."""
        from scipy.spatial import KDTree

        X, y = concentric_annuli(500, 2, n_rings=2, thickness=0.3, margin=1.0, seed=0)
        _, mask = apply_label_noise(y, flip_prob=0.1, targeting="boundary", X=X, seed=0)

        # Compute cross-class distances for all points
        classes = np.unique(y)
        boundary_dists = np.full(len(y), np.inf)
        for cls in classes:
            other = X[y != cls]
            tree = KDTree(other)
            d, _ = tree.query(X[y == cls])
            boundary_dists[y == cls] = d

        # Flipped points should have smaller median boundary distance
        flipped_median = np.median(boundary_dists[mask])
        all_median = np.median(boundary_dists)
        assert flipped_median < all_median

    def test_determinism(self):
        a, ma = apply_label_noise(self.y, flip_prob=0.1, seed=42)
        b, mb = apply_label_noise(self.y, flip_prob=0.1, seed=42)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(ma, mb)

    def test_invalid_flip_prob(self):
        with pytest.raises(ValueError, match="flip_prob"):
            apply_label_noise(self.y, flip_prob=-0.1)
        with pytest.raises(ValueError, match="flip_prob"):
            apply_label_noise(self.y, flip_prob=0.6)

    def test_invalid_targeting(self):
        with pytest.raises(ValueError, match="targeting"):
            apply_label_noise(self.y, flip_prob=0.1, targeting="random")
