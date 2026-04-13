"""Noise injection for features and labels.

Feature noise is additive. Label noise flips a fraction of labels, either
uniformly at random or targeting points near class boundaries.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


SUPPORTED_DISTRIBUTIONS = ("gaussian", "uniform", "laplacian", "exponential", "erlang")


def apply_feature_noise(
    X: np.ndarray,
    distribution: str = "gaussian",
    scale: float = 0.1,
    sparsity: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Add noise to feature columns.

    Returns a new array; the input is not modified.
    """
    if distribution not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(
            f"distribution must be one of {SUPPORTED_DISTRIBUTIONS}, got '{distribution}'"
        )
    if scale < 0:
        raise ValueError(f"scale must be >= 0, got {scale}")
    if not 0 < sparsity <= 1.0:
        raise ValueError(f"sparsity must be in (0, 1], got {sparsity}")
    if scale == 0:
        return X.copy()

    rng = np.random.default_rng(seed)
    n, d = X.shape

    # Select which columns get noise
    n_noisy_cols = max(1, int(round(d * sparsity)))
    noisy_cols = rng.choice(d, size=n_noisy_cols, replace=False)
    noisy_cols.sort()

    # Generate noise only for selected columns
    noise = np.zeros_like(X)
    shape = (n, n_noisy_cols)

    if distribution == "gaussian":
        noise[:, noisy_cols] = rng.normal(0, scale, size=shape)
    elif distribution == "uniform":
        noise[:, noisy_cols] = rng.uniform(-scale, scale, size=shape)
    elif distribution == "laplacian":
        noise[:, noisy_cols] = rng.laplace(0, scale, size=shape)
    elif distribution == "exponential":
        # Exponential is non-negative; mean-center so noise has zero mean.
        raw = rng.exponential(scale, size=shape)
        noise[:, noisy_cols] = raw - raw.mean(axis=0)
    elif distribution == "erlang":
        # Erlang(k=2) gives a right-skewed distribution with heavier tail
        # than exponential. Mean-centered for the same reason as exponential.
        raw = rng.gamma(shape=2, scale=scale / 2, size=shape)
        noise[:, noisy_cols] = raw - raw.mean(axis=0)

    return X + noise


def apply_label_noise(
    y: np.ndarray,
    flip_prob: float = 0.05,
    targeting: str = "uniform",
    X: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip a fraction of labels.

    Returns (y_noisy, flip_mask) where flip_mask is a boolean array
    indicating which labels were changed.
    """
    if not 0 <= flip_prob <= 0.5:
        raise ValueError(f"flip_prob must be in [0, 0.5], got {flip_prob}")
    if targeting not in ("uniform", "boundary"):
        raise ValueError(f"targeting must be 'uniform' or 'boundary', got '{targeting}'")
    if targeting == "boundary" and X is None:
        raise ValueError("X must be provided when targeting='boundary'")

    n = len(y)
    n_flips = int(round(n * flip_prob))
    flip_mask = np.zeros(n, dtype=bool)

    if n_flips == 0:
        return y.copy(), flip_mask

    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    if targeting == "uniform":
        flip_indices = rng.choice(n, size=n_flips, replace=False)
    else:
        # Boundary targeting: flip points closest to a different class.
        # For each point, find distance to nearest point of a different class.
        tree = KDTree(X)
        boundary_distances = np.full(n, np.inf)
        for cls in classes:
            mask = y == cls
            other_mask = ~mask
            if not np.any(other_mask):
                continue
            other_tree = KDTree(X[other_mask])
            dists, _ = other_tree.query(X[mask], k=1)
            boundary_distances[mask] = dists

        # Pick the n_flips points with smallest boundary distance
        flip_indices = np.argsort(boundary_distances)[:n_flips]

    flip_mask[flip_indices] = True
    y_noisy = y.copy()

    # Flip each selected label to a different class
    for idx in flip_indices:
        other_classes = classes[classes != y[idx]]
        y_noisy[idx] = rng.choice(other_classes)

    return y_noisy, flip_mask
