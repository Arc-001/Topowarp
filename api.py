"""FastAPI service for Topowarp dataset generation.

Endpoints:
  GET  /topologies            -- list topologies and their parameter schemas
  POST /generate              -- run the full pipeline, return run metadata
  GET  /runs/{run_id}         -- fetch metadata for a completed run
  GET  /download/{run_id}/{filename} -- download a generated file
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from topowarp.export import export_dataset
from topowarp.generators import (
    archimedean_spirals,
    concentric_annuli,
    disjoint_clusters,
    nd_checkerboard,
)
from topowarp.noise import SUPPORTED_DISTRIBUTIONS, apply_feature_noise, apply_label_noise

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_ROOT = Path("output")
OUTPUT_ROOT.mkdir(exist_ok=True)

_GENERATORS = {
    "concentric_annuli": concentric_annuli,
    "archimedean_spirals": archimedean_spirals,
    "nd_checkerboard": nd_checkerboard,
    "disjoint_clusters": disjoint_clusters,
}

# Topology-specific parameter metadata (for /topologies discovery endpoint)
_TOPOLOGY_SCHEMAS: dict[str, dict] = {
    "concentric_annuli": {
        "description": "Ring-shaped clusters with configurable thickness and margin",
        "params": {
            "n_rings": {"type": "int", "default": 3, "range": [2, 10]},
            "thickness": {"type": "float", "default": 0.3, "range": [0.01, 5.0]},
            "margin": {"type": "float", "default": 0.5, "range": [0.0, 5.0]},
        },
    },
    "archimedean_spirals": {
        "description": "Interleaved spiral arms winding outward from the origin",
        "params": {
            "n_arms": {"type": "int", "default": 2, "range": [2, 8]},
            "turns": {"type": "float", "default": 2.0, "range": [0.5, 10.0]},
            "margin": {"type": "float", "default": 0.3, "range": [0.01, 2.0]},
        },
    },
    "nd_checkerboard": {
        "description": "Binary checkerboard pattern in arbitrary dimensions",
        "params": {
            "freq": {"type": "int", "default": 2, "range": [1, 8]},
        },
    },
    "disjoint_clusters": {
        "description": "Isotropic Gaussian blobs with controlled separation",
        "params": {
            "k": {"type": "int", "default": 4, "range": [2, 20]},
            "separation": {"type": "float", "default": 5.0, "range": [0.5, 50.0]},
        },
    },
}

# In-memory run registry. Keyed by run_id.
_runs: dict[str, dict] = {}

_executor = ThreadPoolExecutor(max_workers=4)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TopologyConfig(BaseModel):
    name: Literal[
        "concentric_annuli",
        "archimedean_spirals",
        "nd_checkerboard",
        "disjoint_clusters",
    ]
    n: int = Field(1000, ge=1, le=50_000, description="Total sample count")
    d: int = Field(2, ge=2, le=30, description="Number of dimensions")
    seed: int = Field(0, description="Random seed")
    # Topology-specific params passed as a free dict; validated by the generator
    params: dict[str, Any] = Field(default_factory=dict)


class FeatureNoiseConfig(BaseModel):
    distribution: str = Field("gaussian", description=f"One of {SUPPORTED_DISTRIBUTIONS}")
    scale: float = Field(0.1, ge=0.0, le=5.0)
    sparsity: float = Field(1.0, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def check_distribution(self) -> FeatureNoiseConfig:
        if self.distribution not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"distribution must be one of {SUPPORTED_DISTRIBUTIONS}"
            )
        return self


class LabelNoiseConfig(BaseModel):
    flip_prob: float = Field(0.0, ge=0.0, le=0.5)
    targeting: Literal["uniform", "boundary"] = "uniform"


class ExportConfig(BaseModel):
    formats: list[Literal["npz", "csv", "pt"]] = Field(["npz"])
    name: str | None = None


class GenerateRequest(BaseModel):
    topology: TopologyConfig
    feature_noise: FeatureNoiseConfig = Field(default_factory=FeatureNoiseConfig)
    label_noise: LabelNoiseConfig = Field(default_factory=LabelNoiseConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)


class RunResponse(BaseModel):
    run_id: str
    status: Literal["ok", "error"]
    output_dir: str
    files: list[str]
    metadata: dict[str, Any]
    error: str | None = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Topowarp API",
    description="Generate synthetic topological datasets with configurable noise.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/topologies", summary="List available topologies and their parameter schemas")
def list_topologies() -> dict:
    return _TOPOLOGY_SCHEMAS


@app.post("/generate", response_model=RunResponse, summary="Generate a dataset")
def generate(req: GenerateRequest) -> RunResponse:
    """Run the full pipeline synchronously and return file paths and metadata.

    Generation is CPU-bound and fast at typical dataset sizes (n <= 50,000),
    so it runs inline rather than returning an async job handle.
    """
    run_id = uuid.uuid4().hex
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True)

    topo = req.topology
    fn = req.feature_noise
    ln = req.label_noise
    ex = req.export

    gen_func = _GENERATORS[topo.name]

    try:
        X_clean, y_clean = gen_func(n=topo.n, d=topo.d, seed=topo.seed, **topo.params)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if fn.scale > 0:
        try:
            X_noisy = apply_feature_noise(
                X_clean,
                distribution=fn.distribution,
                scale=fn.scale,
                sparsity=fn.sparsity,
                seed=topo.seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    else:
        X_noisy = X_clean.copy()

    if ln.flip_prob > 0:
        try:
            y_noisy, flip_mask = apply_label_noise(
                y_clean,
                flip_prob=ln.flip_prob,
                targeting=ln.targeting,
                X=X_clean if ln.targeting == "boundary" else None,
                seed=topo.seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    else:
        y_noisy = y_clean.copy()
        flip_mask = np.zeros(len(y_clean), dtype=bool)

    metadata = {
        "run_id": run_id,
        "topology": topo.model_dump(),
        "feature_noise": fn.model_dump(),
        "label_noise": ln.model_dump(),
        "shapes": {
            "X_clean": list(X_clean.shape),
            "y_clean": list(y_clean.shape),
        },
    }

    written = export_dataset(
        X_clean, y_clean, X_noisy, y_noisy, flip_mask,
        metadata, str(out_dir), ex.formats, ex.name,
    )

    # Store run record for later retrieval
    _runs[run_id] = {
        "run_id": run_id,
        "output_dir": str(out_dir),
        "files": written,
        "metadata": metadata,
    }

    return RunResponse(
        run_id=run_id,
        status="ok",
        output_dir=str(out_dir),
        files=written,
        metadata=metadata,
    )


@app.get("/runs/{run_id}", response_model=RunResponse, summary="Fetch metadata for a run")
def get_run(run_id: str) -> RunResponse:
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    r = _runs[run_id]
    return RunResponse(status="ok", **r)


@app.get(
    "/download/{run_id}/{filename}",
    summary="Download a generated file",
    response_class=FileResponse,
)
def download_file(run_id: str, filename: str) -> FileResponse:
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    # Prevent path traversal
    safe_name = Path(filename).name
    file_path = OUTPUT_ROOT / run_id / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{safe_name}' not found in run")

    media_types = {".npz": "application/octet-stream", ".pt": "application/octet-stream"}
    media_type = media_types.get(file_path.suffix, "application/octet-stream")
    return FileResponse(path=str(file_path), filename=safe_name, media_type=media_type)
