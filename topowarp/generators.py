"""Topological manifold generators.

Each factory function returns (X, y) where X has shape (n, d) and y has shape (n,).
Labels are integer class indices starting at 0. All randomness flows through
np.random.Generator for reproducibility and thread safety.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ortho_group


MAX_SAMPLES = 50_000
MAX_DIMS = 30


def _validate_common(n: int, d: int) -> None:
    """Validate parameters shared across all generators."""
    if not 1 <= n <= MAX_SAMPLES:
        raise ValueError(f"n must be in [1, {MAX_SAMPLES}], got {n}")
    if not 2 <= d <= MAX_DIMS:
        raise ValueError(f"d must be in [2, {MAX_DIMS}], got {d}")


def _embed(X_2d: np.ndarray, d: int, rng: np.random.Generator) -> np.ndarray:
    """Embed 2D points into d dimensions via random orthogonal rotation.

    Places the 2D data into the first two coordinates of a d-dimensional
    space, then applies a random rotation in SO(d) so the structure is
    not axis-aligned.
    """
    if d == 2:
        return X_2d
    n = X_2d.shape[0]
    X_full = np.zeros((n, d), dtype=X_2d.dtype)
    X_full[:, :2] = X_2d
    rotation = ortho_group.rvs(d, random_state=rng)
    return X_full @ rotation.T


def _split_counts(n: int, k: int) -> list[int]:
    """Split n samples across k classes as evenly as possible."""
    base, remainder = divmod(n, k)
    return [base + (1 if i < remainder else 0) for i in range(k)]


def concentric_annuli(
    n: int,
    d: int,
    n_rings: int = 3,
    thickness: float = 0.3,
    margin: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate concentric ring-shaped clusters in d dimensions."""
    _validate_common(n, d)
    if n_rings < 2:
        raise ValueError(f"n_rings must be >= 2, got {n_rings}")
    if thickness <= 0:
        raise ValueError(f"thickness must be > 0, got {thickness}")
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")

    rng = np.random.default_rng(seed)
    counts = _split_counts(n, n_rings)

    points, labels = [], []
    for ring_idx, count in enumerate(counts):
        inner_radius = ring_idx * (thickness + margin)
        angles = rng.uniform(0, 2 * np.pi, size=count)
        radii = rng.uniform(inner_radius, inner_radius + thickness, size=count)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        points.append(np.column_stack([x, y]))
        labels.append(np.full(count, ring_idx, dtype=np.int64))

    X_2d = np.vstack(points)
    y = np.concatenate(labels)
    return _embed(X_2d, d, rng), y


def archimedean_spirals(
    n: int,
    d: int,
    n_arms: int = 2,
    turns: float = 2.0,
    margin: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate interleaved Archimedean spiral arms in d dimensions."""
    _validate_common(n, d)
    if n_arms < 2:
        raise ValueError(f"n_arms must be >= 2, got {n_arms}")
    if turns <= 0:
        raise ValueError(f"turns must be > 0, got {turns}")

    rng = np.random.default_rng(seed)
    counts = _split_counts(n, n_arms)

    points, labels = [], []
    angular_offset = 2 * np.pi / n_arms
    for arm_idx, count in enumerate(counts):
        t = np.linspace(0, turns * 2 * np.pi, count)
        # Small radial jitter so points aren't perfectly on the curve
        noise = rng.normal(0, margin, size=count)
        r = t + noise
        base_angle = arm_idx * angular_offset
        x = r * np.cos(t + base_angle)
        y = r * np.sin(t + base_angle)
        points.append(np.column_stack([x, y]))
        labels.append(np.full(count, arm_idx, dtype=np.int64))

    X_2d = np.vstack(points)
    y = np.concatenate(labels)
    return _embed(X_2d, d, rng), y


def nd_checkerboard(
    n: int,
    d: int,
    freq: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a d-dimensional checkerboard pattern.

    Points are sampled uniformly in [0, freq]^d. The label is determined
    by the parity of the sum of integer cell coordinates, producing a
    binary checkerboard in arbitrary dimensions.
    """
    _validate_common(n, d)
    if freq < 1:
        raise ValueError(f"freq must be >= 1, got {freq}")

    rng = np.random.default_rng(seed)
    X = rng.uniform(0, freq, size=(n, d))
    cell_coords = np.floor(X).astype(np.int64)
    y = (cell_coords.sum(axis=1) % 2).astype(np.int64)
    return X, y


def disjoint_clusters(
    n: int,
    d: int,
    k: int = 4,
    separation: float = 5.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate k isotropic Gaussian clusters with controlled separation."""
    _validate_common(n, d)
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if separation <= 0:
        raise ValueError(f"separation must be > 0, got {separation}")

    rng = np.random.default_rng(seed)
    counts = _split_counts(n, k)

    # Place centroids on a regular simplex-like arrangement in d dimensions.
    # For k <= d+1 this gives exact equidistant centroids. For k > d+1 we
    # fall back to random unit vectors scaled by separation, which still
    # gives good spacing in high-d.
    if k <= d + 1:
        # Standard simplex vertices in d dimensions, scaled
        raw = np.zeros((k, d))
        for i in range(k):
            if i < d:
                raw[i, i] = 1.0
            else:
                raw[i, :] = -1.0 / d
        # Center at origin and scale
        raw -= raw.mean(axis=0)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        centroids = raw / norms * separation
    else:
        directions = rng.standard_normal((k, d))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        centroids = directions / norms * separation

    points, labels = [], []
    for cluster_idx, count in enumerate(counts):
        cluster_points = rng.standard_normal((count, d)) + centroids[cluster_idx]
        points.append(cluster_points)
        labels.append(np.full(count, cluster_idx, dtype=np.int64))

    X = np.vstack(points)
    y = np.concatenate(labels)
    return X, y
