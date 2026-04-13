"""Smoke tests for the visualization module.

Each test verifies the plot function runs and returns a Figure.
No pixel-level assertions -- visual correctness is verified manually.
"""

import numpy as np
import pytest
from matplotlib.figure import Figure

from topowarp.generators import (
    archimedean_spirals,
    concentric_annuli,
    disjoint_clusters,
    nd_checkerboard,
)
from topowarp.noise import apply_feature_noise, apply_label_noise
from topowarp.visualizer import (
    plot_class_balance,
    plot_feature_histograms,
    plot_label_flip_map,
    plot_noise_delta_heatmap,
    plot_robustness_curve,
    plot_scatter,
    render_all,
    save_plots,
)


@pytest.fixture
def dataset_2d():
    X, y = concentric_annuli(200, 2, seed=0)
    X_n = apply_feature_noise(X, "gaussian", scale=0.3, seed=0)
    y_n, mask = apply_label_noise(y, flip_prob=0.1, targeting="uniform", seed=0)
    return X, y, X_n, y_n, mask


@pytest.fixture
def dataset_3d():
    X, y = archimedean_spirals(200, 3, seed=0)
    X_n = apply_feature_noise(X, "gaussian", scale=0.3, seed=0)
    y_n, mask = apply_label_noise(y, flip_prob=0.1, seed=0)
    return X, y, X_n, y_n, mask


@pytest.fixture
def dataset_high_d():
    X, y = nd_checkerboard(300, 10, seed=0)
    X_n = apply_feature_noise(X, "laplacian", scale=0.2, sparsity=0.5, seed=0)
    y_n, mask = apply_label_noise(y, flip_prob=0.05, seed=0)
    return X, y, X_n, y_n, mask


# -- Individual plot functions ------------------------------------------------


class TestScatter:
    def test_2d(self, dataset_2d):
        X_c, y_c, X_n, y_n, _ = dataset_2d
        fig = plot_scatter(X_c, y_c, X_n, y_n)
        assert isinstance(fig, Figure)

    def test_3d(self, dataset_3d):
        X_c, y_c, X_n, y_n, _ = dataset_3d
        fig = plot_scatter(X_c, y_c, X_n, y_n)
        assert isinstance(fig, Figure)

    def test_high_d_pca(self, dataset_high_d):
        X_c, y_c, X_n, y_n, _ = dataset_high_d
        fig = plot_scatter(X_c, y_c, X_n, y_n)
        assert isinstance(fig, Figure)


class TestFeatureHistograms:
    def test_few_features(self, dataset_2d):
        X_c, _, X_n, _, _ = dataset_2d
        fig = plot_feature_histograms(X_c, X_n)
        assert isinstance(fig, Figure)

    def test_many_features_selects_top(self, dataset_high_d):
        X_c, _, X_n, _, _ = dataset_high_d
        fig = plot_feature_histograms(X_c, X_n, max_features=4)
        assert isinstance(fig, Figure)


class TestNoiseDeltaHeatmap:
    def test_basic(self, dataset_2d):
        X_c, _, X_n, _, _ = dataset_2d
        fig = plot_noise_delta_heatmap(X_c, X_n)
        assert isinstance(fig, Figure)

    def test_subsampling(self):
        X_c, _ = disjoint_clusters(2000, 5, seed=0)
        X_n = apply_feature_noise(X_c, "gaussian", scale=0.5, seed=0)
        fig = plot_noise_delta_heatmap(X_c, X_n, max_samples=100)
        assert isinstance(fig, Figure)


class TestLabelFlipMap:
    def test_basic(self, dataset_2d):
        X_c, y_c, _, _, mask = dataset_2d
        fig = plot_label_flip_map(X_c, y_c, mask)
        assert isinstance(fig, Figure)

    def test_high_d(self, dataset_high_d):
        X_c, y_c, _, _, mask = dataset_high_d
        fig = plot_label_flip_map(X_c, y_c, mask)
        assert isinstance(fig, Figure)


class TestClassBalance:
    def test_basic(self, dataset_2d):
        _, y_c, _, y_n, _ = dataset_2d
        fig = plot_class_balance(y_c, y_n)
        assert isinstance(fig, Figure)


class TestRobustnessCurve:
    def test_basic(self):
        results = {0.0: 0.95, 0.1: 0.92, 0.5: 0.80, 1.0: 0.65, 1.5: 0.50}
        fig = plot_robustness_curve(results)
        assert isinstance(fig, Figure)

    def test_custom_labels(self):
        results = {0.0: 0.9, 0.3: 0.7}
        fig = plot_robustness_curve(results, metric_name="F1", param_name="Flip Probability")
        assert isinstance(fig, Figure)


# -- Full suite and export ----------------------------------------------------


class TestRenderAll:
    def test_returns_figures(self, dataset_2d):
        X_c, y_c, X_n, y_n, mask = dataset_2d
        figs = render_all(X_c, y_c, X_n, y_n, mask)
        assert "scatter" in figs
        assert "feature_hist" in figs
        assert "noise_delta" in figs
        assert "class_balance" in figs
        assert "label_flip" in figs
        assert all(isinstance(f, Figure) for f in figs.values())

    def test_no_flip_mask(self, dataset_2d):
        X_c, y_c, X_n, y_n, _ = dataset_2d
        figs = render_all(X_c, y_c, X_n, y_n, flip_mask=None)
        assert "label_flip" not in figs

    def test_save_to_dir(self, dataset_2d, tmp_path):
        X_c, y_c, X_n, y_n, mask = dataset_2d
        figs = render_all(X_c, y_c, X_n, y_n, mask, output_dir=str(tmp_path), dataset_name="test")
        # Figures are closed after save, check files exist
        png_files = list(tmp_path.glob("*.png"))
        svg_files = list(tmp_path.glob("*.svg"))
        assert len(png_files) == 5
        assert len(svg_files) == 5


class TestSavePlots:
    def test_formats(self, dataset_2d, tmp_path):
        X_c, y_c, X_n, y_n, _ = dataset_2d
        fig = plot_scatter(X_c, y_c, X_n, y_n)
        paths = save_plots({"scatter": fig}, str(tmp_path), "test", ("png",))
        assert len(paths) == 1
        assert paths[0].endswith(".png")
