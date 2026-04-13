"""Tests for dataset export pipeline."""

import json

import numpy as np
import pytest
import torch

from topowarp.export import export_dataset


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(0)
    n, d = 50, 4
    X_clean = rng.standard_normal((n, d))
    y_clean = np.repeat(np.arange(2), n // 2)
    X_noisy = X_clean + rng.normal(0, 0.1, size=(n, d))
    y_noisy = y_clean.copy()
    y_noisy[0] = 1 - y_noisy[0]
    flip_mask = np.zeros(n, dtype=bool)
    flip_mask[0] = True
    metadata = {"topology": "test", "seed": 0}
    return X_clean, y_clean, X_noisy, y_noisy, flip_mask, metadata


class TestNpzExport:
    def test_roundtrip(self, sample_data, tmp_path):
        X_c, y_c, X_n, y_n, mask, meta = sample_data
        paths = export_dataset(X_c, y_c, X_n, y_n, mask, meta, str(tmp_path), ["npz"], "test")

        npz_paths = [p for p in paths if p.endswith(".npz")]
        assert len(npz_paths) == 1

        loaded = np.load(npz_paths[0])
        np.testing.assert_array_equal(loaded["X_clean"], X_c)
        np.testing.assert_array_equal(loaded["y_clean"], y_c)
        np.testing.assert_array_equal(loaded["X_noisy"], X_n)
        np.testing.assert_array_equal(loaded["y_noisy"], y_n)
        np.testing.assert_array_equal(loaded["flip_mask"], mask)


class TestCsvExport:
    def test_roundtrip(self, sample_data, tmp_path):
        X_c, y_c, X_n, y_n, mask, meta = sample_data
        paths = export_dataset(X_c, y_c, X_n, y_n, mask, meta, str(tmp_path), ["csv"], "test")

        csv_paths = [p for p in paths if p.endswith(".csv")]
        assert len(csv_paths) == 5  # X_clean, y_clean, X_noisy, y_noisy, flip_mask

        X_loaded = np.loadtxt([p for p in csv_paths if "X_clean" in p][0], delimiter=",")
        np.testing.assert_array_almost_equal(X_loaded, X_c)


class TestPtExport:
    def test_roundtrip(self, sample_data, tmp_path):
        X_c, y_c, X_n, y_n, mask, meta = sample_data
        paths = export_dataset(X_c, y_c, X_n, y_n, mask, meta, str(tmp_path), ["pt"], "test")

        pt_paths = [p for p in paths if p.endswith(".pt")]
        assert len(pt_paths) == 1

        loaded = torch.load(pt_paths[0], weights_only=True)
        np.testing.assert_array_equal(loaded["X_clean"].numpy(), X_c)
        np.testing.assert_array_equal(loaded["y_noisy"].numpy(), y_n)


class TestMetadataSidecar:
    def test_json_written(self, sample_data, tmp_path):
        X_c, y_c, X_n, y_n, mask, meta = sample_data
        paths = export_dataset(X_c, y_c, X_n, y_n, mask, meta, str(tmp_path), ["npz"], "test")

        json_paths = [p for p in paths if p.endswith(".json")]
        assert len(json_paths) == 1

        with open(json_paths[0]) as f:
            sidecar = json.load(f)

        assert sidecar["topology"] == "test"
        assert sidecar["seed"] == 0
        assert "timestamp" in sidecar
        assert sidecar["arrays"]["X_clean"] == [50, 4]

    def test_none_flip_mask(self, sample_data, tmp_path):
        X_c, y_c, X_n, y_n, _, meta = sample_data
        paths = export_dataset(X_c, y_c, X_n, y_n, None, meta, str(tmp_path), ["npz"], "test")

        with open([p for p in paths if p.endswith(".json")][0]) as f:
            sidecar = json.load(f)
        assert sidecar["arrays"]["flip_mask"] is None


class TestMultiFormat:
    def test_all_formats(self, sample_data, tmp_path):
        X_c, y_c, X_n, y_n, mask, meta = sample_data
        paths = export_dataset(
            X_c, y_c, X_n, y_n, mask, meta, str(tmp_path), ["npz", "csv", "pt"], "test"
        )
        extensions = {p.rsplit(".", 1)[-1] for p in paths}
        assert {"json", "npz", "csv", "pt"} == extensions


class TestValidation:
    def test_invalid_format(self, sample_data, tmp_path):
        X_c, y_c, X_n, y_n, mask, meta = sample_data
        with pytest.raises(ValueError, match="Unsupported format"):
            export_dataset(X_c, y_c, X_n, y_n, mask, meta, str(tmp_path), ["hdf5"])

    def test_creates_output_dir(self, sample_data, tmp_path):
        X_c, y_c, X_n, y_n, mask, meta = sample_data
        nested = tmp_path / "a" / "b" / "c"
        paths = export_dataset(X_c, y_c, X_n, y_n, mask, meta, str(nested), ["npz"], "test")
        assert len(paths) == 2  # json + npz
        assert nested.exists()


class TestAutoNaming:
    def test_default_name(self, sample_data, tmp_path):
        X_c, y_c, X_n, y_n, mask, meta = sample_data
        paths = export_dataset(X_c, y_c, X_n, y_n, mask, meta, str(tmp_path))
        npz_paths = [p for p in paths if p.endswith(".npz")]
        assert len(npz_paths) == 1
        assert "dataset_" in npz_paths[0]
