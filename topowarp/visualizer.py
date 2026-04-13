"""Visualization module for topological datasets.

Each plot function accepts arrays and returns a matplotlib Figure.
The save_plots utility handles export to png/svg. The module is also
runnable as a CLI: python -m topowarp.visualizer --input dataset.npz
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.decomposition import PCA


# -- Color palette ------------------------------------------------------------

CLEAN_COLOR = "#2563eb"
NOISY_COLOR = "#dc2626"
FLIP_COLOR = "#f59e0b"
CLASS_CMAP = "tab10"


# -- Projection helper -------------------------------------------------------


def _project_2d(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Reduce to 2D via PCA if needed. Returns (X_2d, axis_labels)."""
    if X.shape[1] == 2:
        return X, ["x1", "x2"]
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    return X_2d, ["PC1", "PC2"]


def _project_3d(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Reduce to 3D via PCA if needed. Returns (X_3d, axis_labels)."""
    if X.shape[1] == 3:
        return X, ["x1", "x2", "x3"]
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X)
    return X_3d, ["PC1", "PC2", "PC3"]


# -- Plot functions -----------------------------------------------------------


def plot_scatter(
    X_clean: np.ndarray,
    y_clean: np.ndarray,
    X_noisy: np.ndarray,
    y_noisy: np.ndarray,
) -> Figure:
    """Side-by-side scatter of clean vs noisy data, color-coded by class."""
    d = X_clean.shape[1]
    use_3d = d == 3

    if use_3d:
        X_c, labels_c = _project_3d(X_clean)
        X_n, labels_n = _project_3d(X_noisy)
    else:
        X_c, labels_c = _project_2d(X_clean)
        X_n, labels_n = _project_2d(X_noisy)

    if use_3d:
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")

        ax1.scatter(X_c[:, 0], X_c[:, 1], X_c[:, 2], c=y_clean, cmap=CLASS_CMAP, s=8, alpha=0.7)
        ax1.set_xlabel(labels_c[0])
        ax1.set_ylabel(labels_c[1])
        ax1.set_zlabel(labels_c[2])
        ax1.set_title("Clean")

        ax2.scatter(X_n[:, 0], X_n[:, 1], X_n[:, 2], c=y_noisy, cmap=CLASS_CMAP, s=8, alpha=0.7)
        ax2.set_xlabel(labels_n[0])
        ax2.set_ylabel(labels_n[1])
        ax2.set_zlabel(labels_n[2])
        ax2.set_title("Noisy")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.scatter(X_c[:, 0], X_c[:, 1], c=y_clean, cmap=CLASS_CMAP, s=8, alpha=0.7)
        ax1.set_xlabel(labels_c[0])
        ax1.set_ylabel(labels_c[1])
        ax1.set_title("Clean")
        ax1.set_aspect("equal", adjustable="datalim")

        ax2.scatter(X_n[:, 0], X_n[:, 1], c=y_noisy, cmap=CLASS_CMAP, s=8, alpha=0.7)
        ax2.set_xlabel(labels_n[0])
        ax2.set_ylabel(labels_n[1])
        ax2.set_title("Noisy")
        ax2.set_aspect("equal", adjustable="datalim")

    fig.suptitle("Clean vs Noisy", fontsize=14)
    fig.tight_layout()
    return fig


def plot_feature_histograms(
    X_clean: np.ndarray,
    X_noisy: np.ndarray,
    max_features: int = 6,
) -> Figure:
    """Per-feature density overlay of clean vs noisy distributions."""
    d = X_clean.shape[1]

    if d <= max_features:
        indices = list(range(d))
    else:
        # Pick features with highest total noise delta
        deltas = np.abs(X_noisy - X_clean).sum(axis=0)
        indices = list(np.argsort(deltas)[-max_features:])
        indices.sort()

    n_plots = len(indices)
    cols = min(n_plots, 3)
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax_idx, feat_idx in enumerate(indices):
        ax = axes[ax_idx]
        ax.hist(X_clean[:, feat_idx], bins=40, alpha=0.6, color=CLEAN_COLOR, label="Clean", density=True)
        ax.hist(X_noisy[:, feat_idx], bins=40, alpha=0.6, color=NOISY_COLOR, label="Noisy", density=True)
        ax.set_title(f"Feature {feat_idx}")
        ax.legend(fontsize=8)

    for ax_idx in range(n_plots, len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=14)
    fig.tight_layout()
    return fig


def plot_noise_delta_heatmap(
    X_clean: np.ndarray,
    X_noisy: np.ndarray,
    max_samples: int = 500,
) -> Figure:
    """Heatmap of |X_noisy - X_clean| showing sparsity and outlier patterns."""
    delta = np.abs(X_noisy - X_clean)

    # Subsample rows for readability if dataset is large
    if delta.shape[0] > max_samples:
        step = delta.shape[0] // max_samples
        delta = delta[::step][:max_samples]

    fig, ax = plt.subplots(figsize=(max(6, delta.shape[1] * 0.5), max(4, delta.shape[0] * 0.01)))
    im = ax.imshow(delta, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Sample")
    ax.set_title("Noise Delta |X_noisy - X_clean|")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_label_flip_map(
    X_clean: np.ndarray,
    y_clean: np.ndarray,
    flip_mask: np.ndarray,
) -> Figure:
    """Scatter highlighting which points had labels flipped."""
    X_2d, labels = _project_2d(X_clean)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        X_2d[~flip_mask, 0], X_2d[~flip_mask, 1],
        c=y_clean[~flip_mask], cmap=CLASS_CMAP, s=8, alpha=0.4, label="Unchanged",
    )
    ax.scatter(
        X_2d[flip_mask, 0], X_2d[flip_mask, 1],
        c=FLIP_COLOR, s=30, alpha=0.9, marker="x", linewidths=1.5, label="Flipped",
    )
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title("Label Flip Map")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig


def plot_class_balance(
    y_clean: np.ndarray,
    y_noisy: np.ndarray,
) -> Figure:
    """Side-by-side bar chart of class distributions."""
    classes = np.union1d(np.unique(y_clean), np.unique(y_noisy))
    clean_counts = np.array([np.sum(y_clean == c) for c in classes])
    noisy_counts = np.array([np.sum(y_noisy == c) for c in classes])

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.2), 5))
    ax.bar(x - width / 2, clean_counts, width, label="Clean", color=CLEAN_COLOR)
    ax.bar(x + width / 2, noisy_counts, width, label="Noisy", color=NOISY_COLOR)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Balance")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes])
    ax.legend()
    fig.tight_layout()
    return fig


def plot_robustness_curve(
    results: dict[float, float],
    metric_name: str = "Accuracy",
    param_name: str = "Noise Scale",
) -> Figure:
    """Plot noise parameter vs model performance curve."""
    params = sorted(results.keys())
    values = [results[p] for p in params]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(params, values, "o-", color=CLEAN_COLOR, linewidth=2, markersize=6)
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs {param_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# -- Save utility -------------------------------------------------------------


def save_plots(
    figures: dict[str, Figure],
    output_dir: str,
    dataset_name: str = "dataset",
    formats: tuple[str, ...] = ("png", "svg"),
    dpi: int = 300,
) -> list[str]:
    """Save a dict of {plot_type: Figure} to files. Returns paths written."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for plot_type, fig in figures.items():
        for fmt in formats:
            path = out / f"{dataset_name}_{plot_type}.{fmt}"
            fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
            written.append(str(path))
        plt.close(fig)
    return written


# -- Full suite runner --------------------------------------------------------


def render_all(
    X_clean: np.ndarray,
    y_clean: np.ndarray,
    X_noisy: np.ndarray,
    y_noisy: np.ndarray,
    flip_mask: np.ndarray | None = None,
    output_dir: str | None = None,
    dataset_name: str = "dataset",
) -> dict[str, Figure]:
    """Generate the full plot suite. Optionally save to disk."""
    figures: dict[str, Figure] = {}

    figures["scatter"] = plot_scatter(X_clean, y_clean, X_noisy, y_noisy)
    figures["feature_hist"] = plot_feature_histograms(X_clean, X_noisy)
    figures["noise_delta"] = plot_noise_delta_heatmap(X_clean, X_noisy)
    figures["class_balance"] = plot_class_balance(y_clean, y_noisy)

    if flip_mask is not None and flip_mask.any():
        figures["label_flip"] = plot_label_flip_map(X_clean, y_clean, flip_mask)

    if output_dir is not None:
        save_plots(figures, output_dir, dataset_name)

    return figures


# -- CLI entry point ----------------------------------------------------------


def main() -> None:
    """CLI: python -m topowarp.visualizer --input dataset.npz --output plots/"""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize a Topowarp dataset")
    parser.add_argument("--input", required=True, help="Path to .npz dataset file")
    parser.add_argument("--output", default="plots", help="Output directory for plots")
    parser.add_argument("--name", default=None, help="Base name for plot files")
    args = parser.parse_args()

    data = np.load(args.input)
    X_clean = data["X_clean"]
    y_clean = data["y_clean"]
    X_noisy = data["X_noisy"]
    y_noisy = data["y_noisy"]
    flip_mask = data.get("flip_mask")

    dataset_name = args.name or Path(args.input).stem

    render_all(X_clean, y_clean, X_noisy, y_noisy, flip_mask, args.output, dataset_name)
    print(f"Plots saved to {args.output}/")


if __name__ == "__main__":
    main()
