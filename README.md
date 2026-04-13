# Topowarp

Topowarp is a CPU-only Python toolkit for generating synthetic topological datasets with configurable noise. It is designed for benchmarking classifiers — particularly MLP vs. RBF networks — across topologies that stress-test different aspects of decision-boundary learning.

---

## TL;DR

```bash
pip install -r requirements.txt

# TUI wizard
python tui.py

# REST API
uvicorn api:app --reload

# CLI visualizer
python -m topowarp.visualizer --input output/dataset.npz --output plots/
```

---

## Contents

- [Topologies](#topologies)
- [Noise](#noise)
- [Export formats](#export-formats)
- [Interfaces](#interfaces)
  - [Python API](#python-api)
  - [TUI wizard](#tui-wizard)
  - [REST API](#rest-api)
  - [Visualizer CLI](#visualizer-cli)
- [Project structure](#project-structure)
- [Running tests](#running-tests)

---

## Topologies

Each generator returns `(X, y)` as NumPy arrays of shape `(n, d)` and `(n,)`. All accept an explicit `seed` for reproducibility.

| Topology | Classes | Description |
|---|---|---|
| `concentric_annuli` | `n_rings` | Concentric ring-shaped clusters |
| `archimedean_spirals` | `n_arms` | Interleaved spiral arms |
| `nd_checkerboard` | 2 | Binary checkerboard in d dimensions |
| `disjoint_clusters` | `k` | Isotropic Gaussian blobs |

**Constraints:** `n` capped at 50,000, `d` capped at 30. For `d > 2`, the base topology is generated in 2D and rotated into the target dimensionality via a random orthogonal matrix (`scipy.stats.ortho_group`), preserving inter-point distances.

---

## Noise

### Feature noise

`apply_feature_noise(X, distribution, scale, sparsity, seed)` adds noise to a configurable fraction of feature columns.

| Distribution | Notes |
|---|---|
| `gaussian` | Zero-mean, standard deviation = `scale` |
| `uniform` | Symmetric, half-width = `scale` |
| `laplacian` | Zero-mean, scale parameter = `scale` |
| `exponential` | Mean-centered before adding; retains right skew |
| `erlang` | Erlang k=2, mean-centered; heavier tail than exponential |

`sparsity` controls the fraction of columns affected (1.0 = all columns, 0.1 = 10%).

### Label noise

`apply_label_noise(y, flip_prob, targeting, X, seed)` flips a fraction of labels and returns `(y_noisy, flip_mask)`.

| Targeting | Behavior |
|---|---|
| `uniform` | Flips selected uniformly at random |
| `boundary` | Prioritizes points closest to a different class, using KDTree cross-class distance as a decision-boundary proxy |

`flip_prob` is capped at 0.5.

---

## Export formats

`export_dataset(...)` writes the following:

- `.npz` — compressed NumPy archive with keys `X_clean`, `y_clean`, `X_noisy`, `y_noisy`, `flip_mask`
- `.csv` — one file per array, no pandas dependency
- `.pt` — PyTorch tensors via `torch.save`
- `.json` — metadata sidecar written alongside every export, recording topology type, all parameters, noise config, seed, and timestamp

---

## Interfaces

### Python API

```python
from topowarp.generators import concentric_annuli
from topowarp.noise import apply_feature_noise, apply_label_noise
from topowarp.export import export_dataset
from topowarp.visualizer import render_all

X, y = concentric_annuli(n=2000, d=4, n_rings=3, seed=42)

X_noisy = apply_feature_noise(X, distribution="laplacian", scale=0.3, sparsity=0.8, seed=42)
y_noisy, flip_mask = apply_label_noise(y, flip_prob=0.1, targeting="boundary", X=X, seed=42)

export_dataset(
    X, y, X_noisy, y_noisy, flip_mask,
    metadata={"topology": "concentric_annuli"},
    output_dir="output/",
    formats=["npz", "csv"],
)

render_all(X, y, X_noisy, y_noisy, flip_mask, output_dir="plots/", dataset_name="annuli")
```

### TUI wizard

```bash
python tui.py
```

A six-screen wizard: topology selection, parameter configuration, feature noise, label noise, export options, and a review screen before generation. Supports OFAT sweep mode for sweeping noise scale, flip probability, or sparsity across a predefined range.

Keybindings: `?` for contextual help, `Escape` to go back, `Ctrl+Q` to quit.

### REST API

```bash
uvicorn api:app --reload
```

Interactive docs at `http://localhost:8000/docs`.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/topologies` | List topologies and their parameter schemas |
| `POST` | `/generate` | Run the full pipeline, return run ID and file paths |
| `GET` | `/runs/{run_id}` | Fetch metadata for a completed run |
| `GET` | `/download/{run_id}/{filename}` | Download a generated file |

**Example request:**

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "topology": {"name": "concentric_annuli", "n": 2000, "d": 4, "seed": 42, "params": {"n_rings": 3}},
    "feature_noise": {"distribution": "gaussian", "scale": 0.2, "sparsity": 1.0},
    "label_noise": {"flip_prob": 0.05, "targeting": "boundary"},
    "export": {"formats": ["npz"]}
  }'
```

**Response:**

```json
{
  "run_id": "3f1a...",
  "status": "ok",
  "output_dir": "output/3f1a.../",
  "files": ["output/3f1a.../dataset.npz", "output/3f1a.../dataset.json"],
  "metadata": {
    "topology": {"name": "concentric_annuli", "n": 2000, "d": 4, "seed": 42},
    "shapes": {"X_clean": [2000, 4], "y_clean": [2000]}
  }
}
```

Run state is held in memory and cleared on restart; generated files persist on disk.

### Visualizer CLI

```bash
python -m topowarp.visualizer --input output/dataset.npz --output plots/
```

Loads any exported `.npz` and renders the full plot suite: scatter (2D/3D, with PCA projection for `d > 3`), per-feature distribution histograms, noise delta heatmap, label flip map, class balance bars, and a robustness curve scaffold. Outputs `.png` (300 dpi) and `.svg`.

---

## Project structure

```
topowarp/
├── topowarp/
│   ├── generators.py      # Manifold factory functions
│   ├── noise.py           # Feature and label noise injection
│   ├── export.py          # Multi-format export and metadata sidecar
│   └── visualizer.py      # Plots and standalone CLI
├── tui.py                 # Textual TUI wizard
├── api.py                 # FastAPI service
└── tests/
    ├── test_generators.py
    ├── test_noise.py
    ├── test_export.py
    ├── test_visualizer.py
    ├── test_tui.py
    └── test_api.py
```

---

## Running tests

```bash
pytest tests/ -v
```

116 tests covering shapes, determinism, class balance, noise statistics, export roundtrips, TUI screen flow, and API endpoints.
