"""Tests for the FastAPI generation service."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import app, _runs

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_runs():
    """Isolate run state between tests."""
    _runs.clear()
    yield
    _runs.clear()


# ---------------------------------------------------------------------------
# /topologies
# ---------------------------------------------------------------------------


class TestTopologies:
    def test_returns_all_four(self):
        r = client.get("/topologies")
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) == {
            "concentric_annuli",
            "archimedean_spirals",
            "nd_checkerboard",
            "disjoint_clusters",
        }

    def test_schema_has_params(self):
        r = client.get("/topologies")
        body = r.json()
        for topo, info in body.items():
            assert "description" in info
            assert "params" in info


# ---------------------------------------------------------------------------
# /generate
# ---------------------------------------------------------------------------


class TestGenerate:
    def _minimal(self, topo: str, extra_params: dict = {}) -> dict:
        return {
            "topology": {"name": topo, "n": 100, "d": 2, "seed": 0, "params": extra_params},
            "feature_noise": {"distribution": "gaussian", "scale": 0.1, "sparsity": 1.0},
            "label_noise": {"flip_prob": 0.05, "targeting": "uniform"},
            "export": {"formats": ["npz"]},
        }

    def test_basic_generation(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = client.post("/generate", json=self._minimal("concentric_annuli"))
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "run_id" in body
        assert len(body["files"]) == 2  # npz + json sidecar

    def test_all_topologies(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        topologies = [
            ("concentric_annuli", {}),
            ("archimedean_spirals", {}),
            ("nd_checkerboard", {}),
            ("disjoint_clusters", {}),
        ]
        for topo, extra in topologies:
            r = client.post("/generate", json=self._minimal(topo, extra))
            assert r.status_code == 200, f"{topo} failed: {r.text}"

    def test_npz_is_loadable(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = client.post("/generate", json=self._minimal("nd_checkerboard"))
        body = r.json()
        npz_path = next(f for f in body["files"] if f.endswith(".npz"))
        data = np.load(npz_path)
        assert data["X_clean"].shape == (100, 2)
        assert data["y_clean"].shape == (100,)

    def test_all_formats(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        req = self._minimal("disjoint_clusters")
        req["export"]["formats"] = ["npz", "csv", "pt"]
        r = client.post("/generate", json=req)
        assert r.status_code == 200
        exts = {f.rsplit(".", 1)[-1] for f in r.json()["files"]}
        assert {"npz", "csv", "pt", "json"} == exts

    def test_metadata_contains_topology(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = client.post("/generate", json=self._minimal("archimedean_spirals"))
        meta = r.json()["metadata"]
        assert meta["topology"]["name"] == "archimedean_spirals"
        assert meta["shapes"]["X_clean"] == [100, 2]

    def test_no_noise(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        req = self._minimal("concentric_annuli")
        req["feature_noise"]["scale"] = 0.0
        req["label_noise"]["flip_prob"] = 0.0
        r = client.post("/generate", json=req)
        assert r.status_code == 200

    def test_boundary_targeting(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        req = self._minimal("concentric_annuli")
        req["label_noise"] = {"flip_prob": 0.1, "targeting": "boundary"}
        r = client.post("/generate", json=req)
        assert r.status_code == 200

    def test_topology_specific_params(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        req = self._minimal("concentric_annuli", {"n_rings": 4, "thickness": 0.2, "margin": 0.8})
        r = client.post("/generate", json=req)
        assert r.status_code == 200

    def test_invalid_n(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        req = self._minimal("nd_checkerboard")
        req["topology"]["n"] = 0
        r = client.post("/generate", json=req)
        assert r.status_code == 422

    def test_invalid_d(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        req = self._minimal("nd_checkerboard")
        req["topology"]["d"] = 1
        r = client.post("/generate", json=req)
        assert r.status_code == 422

    def test_invalid_distribution(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        req = self._minimal("nd_checkerboard")
        req["feature_noise"]["distribution"] = "not_a_dist"
        r = client.post("/generate", json=req)
        assert r.status_code == 422

    def test_invalid_topology_params_propagate(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        req = self._minimal("concentric_annuli", {"n_rings": 1})  # invalid
        r = client.post("/generate", json=req)
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# /runs/{run_id}
# ---------------------------------------------------------------------------


class TestGetRun:
    def test_returns_run_metadata(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        gen = client.post("/generate", json={
            "topology": {"name": "nd_checkerboard", "n": 50, "d": 2, "seed": 1, "params": {}},
        })
        run_id = gen.json()["run_id"]
        r = client.get(f"/runs/{run_id}")
        assert r.status_code == 200
        assert r.json()["run_id"] == run_id

    def test_unknown_run_id(self):
        r = client.get("/runs/doesnotexist")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# /download/{run_id}/{filename}
# ---------------------------------------------------------------------------


class TestDownload:
    def test_downloads_npz(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        gen = client.post("/generate", json={
            "topology": {"name": "nd_checkerboard", "n": 50, "d": 2, "seed": 0, "params": {}},
            "export": {"formats": ["npz"]},
        })
        body = gen.json()
        run_id = body["run_id"]
        npz_filename = next(
            p.split("/")[-1] for p in body["files"] if p.endswith(".npz")
        )
        r = client.get(f"/download/{run_id}/{npz_filename}")
        assert r.status_code == 200
        assert "application/octet-stream" in r.headers["content-type"]

    def test_unknown_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        gen = client.post("/generate", json={
            "topology": {"name": "nd_checkerboard", "n": 50, "d": 2, "seed": 0, "params": {}},
        })
        run_id = gen.json()["run_id"]
        r = client.get(f"/download/{run_id}/nonexistent.npz")
        assert r.status_code == 404

    def test_path_traversal_blocked(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        gen = client.post("/generate", json={
            "topology": {"name": "nd_checkerboard", "n": 50, "d": 2, "seed": 0, "params": {}},
        })
        run_id = gen.json()["run_id"]
        r = client.get(f"/download/{run_id}/../../etc/passwd")
        assert r.status_code == 404
