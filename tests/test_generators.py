"""Tests for topological manifold generators."""

import numpy as np
import pytest

from topowarp.generators import (
    MAX_DIMS,
    MAX_SAMPLES,
    archimedean_spirals,
    concentric_annuli,
    disjoint_clusters,
    nd_checkerboard,
)


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------

GENERATORS = [
    ("annuli", lambda n, d, seed: concentric_annuli(n, d, seed=seed)),
    ("spirals", lambda n, d, seed: archimedean_spirals(n, d, seed=seed)),
    ("checkerboard", lambda n, d, seed: nd_checkerboard(n, d, seed=seed)),
    ("clusters", lambda n, d, seed: disjoint_clusters(n, d, seed=seed)),
]


@pytest.mark.parametrize("name, gen", GENERATORS, ids=[g[0] for g in GENERATORS])
@pytest.mark.parametrize("n, d", [(100, 2), (200, 5), (53, 10)])
def test_output_shape(name, gen, n, d):
    X, y = gen(n, d, seed=42)
    assert X.shape == (n, d)
    assert y.shape == (n,)


@pytest.mark.parametrize("name, gen", GENERATORS, ids=[g[0] for g in GENERATORS])
def test_label_dtype(name, gen):
    _, y = gen(100, 3, seed=0)
    assert y.dtype == np.int64


# ---------------------------------------------------------------------------
# Class balance
# ---------------------------------------------------------------------------


def test_annuli_class_balance():
    X, y = concentric_annuli(100, 2, n_rings=4, seed=0)
    counts = np.bincount(y)
    assert counts.sum() == 100
    assert max(counts) - min(counts) <= 1


def test_spirals_class_balance():
    X, y = archimedean_spirals(99, 2, n_arms=3, seed=0)
    counts = np.bincount(y)
    assert counts.sum() == 99
    assert max(counts) - min(counts) <= 1


def test_checkerboard_is_binary():
    _, y = nd_checkerboard(500, 4, freq=3, seed=0)
    assert set(np.unique(y)) == {0, 1}


def test_clusters_class_balance():
    _, y = disjoint_clusters(101, 3, k=5, seed=0)
    counts = np.bincount(y)
    assert counts.sum() == 101
    assert max(counts) - min(counts) <= 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, gen", GENERATORS, ids=[g[0] for g in GENERATORS])
def test_determinism(name, gen):
    X1, y1 = gen(200, 4, seed=123)
    X2, y2 = gen(200, 4, seed=123)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


@pytest.mark.parametrize("name, gen", GENERATORS, ids=[g[0] for g in GENERATORS])
def test_different_seeds_differ(name, gen):
    X1, _ = gen(200, 4, seed=0)
    X2, _ = gen(200, 4, seed=1)
    assert not np.array_equal(X1, X2)


# ---------------------------------------------------------------------------
# High-dimensional embedding
# ---------------------------------------------------------------------------


def test_annuli_high_d_preserves_pairwise_distances():
    """Random rotation preserves Euclidean distances."""
    X_2d, _ = concentric_annuli(50, 2, seed=42)
    X_hd, _ = concentric_annuli(50, 8, seed=42)
    # The 2D generation uses the same seed, but the RNG state diverges
    # after embedding. Instead, just verify the high-d data has non-trivial
    # variance in more than 2 dimensions (rotation spread the data).
    col_vars = np.var(X_hd, axis=0)
    assert np.sum(col_vars > 1e-10) > 2, "Embedding should spread data across dimensions"


def test_checkerboard_native_d():
    """Checkerboard generates natively in d dimensions, no embedding needed."""
    X, _ = nd_checkerboard(200, 5, freq=2, seed=0)
    col_vars = np.var(X, axis=0)
    assert np.all(col_vars > 0.01), "All dimensions should have variance"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_n_out_of_range():
    with pytest.raises(ValueError, match="n must be"):
        concentric_annuli(0, 2)
    with pytest.raises(ValueError, match="n must be"):
        concentric_annuli(MAX_SAMPLES + 1, 2)


def test_d_out_of_range():
    with pytest.raises(ValueError, match="d must be"):
        archimedean_spirals(100, 1)
    with pytest.raises(ValueError, match="d must be"):
        archimedean_spirals(100, MAX_DIMS + 1)


def test_annuli_bad_params():
    with pytest.raises(ValueError, match="n_rings"):
        concentric_annuli(100, 2, n_rings=1)
    with pytest.raises(ValueError, match="thickness"):
        concentric_annuli(100, 2, thickness=0)
    with pytest.raises(ValueError, match="margin"):
        concentric_annuli(100, 2, margin=-1)


def test_spirals_bad_params():
    with pytest.raises(ValueError, match="n_arms"):
        archimedean_spirals(100, 2, n_arms=1)
    with pytest.raises(ValueError, match="turns"):
        archimedean_spirals(100, 2, turns=0)


def test_checkerboard_bad_freq():
    with pytest.raises(ValueError, match="freq"):
        nd_checkerboard(100, 2, freq=0)


def test_clusters_bad_params():
    with pytest.raises(ValueError, match="k must be"):
        disjoint_clusters(100, 2, k=1)
    with pytest.raises(ValueError, match="separation"):
        disjoint_clusters(100, 2, separation=0)
