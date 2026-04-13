"""Dataset export to .npz, .csv, and .pt formats with JSON metadata sidecar."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SUPPORTED_FORMATS = ("npz", "csv", "pt")


def export_dataset(
    X_clean: np.ndarray,
    y_clean: np.ndarray,
    X_noisy: np.ndarray,
    y_noisy: np.ndarray,
    flip_mask: np.ndarray | None,
    metadata: dict,
    output_dir: str,
    formats: list[str] | None = None,
    name: str | None = None,
) -> list[str]:
    """Export dataset arrays and metadata sidecar.

    Returns a list of file paths that were written.
    """
    if formats is None:
        formats = ["npz"]
    for fmt in formats:
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{fmt}'. Must be one of {SUPPORTED_FORMATS}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = name if name else f"dataset_{timestamp}"

    written: list[str] = []

    # Metadata sidecar (always written)
    meta = {
        **metadata,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arrays": {
            "X_clean": list(X_clean.shape),
            "y_clean": list(y_clean.shape),
            "X_noisy": list(X_noisy.shape),
            "y_noisy": list(y_noisy.shape),
            "flip_mask": list(flip_mask.shape) if flip_mask is not None else None,
        },
    }
    meta_path = out / f"{base}.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    written.append(str(meta_path))

    arrays = {
        "X_clean": X_clean,
        "y_clean": y_clean,
        "X_noisy": X_noisy,
        "y_noisy": y_noisy,
    }
    if flip_mask is not None:
        arrays["flip_mask"] = flip_mask

    if "npz" in formats:
        path = out / f"{base}.npz"
        np.savez_compressed(str(path), **arrays)
        written.append(str(path))

    if "csv" in formats:
        csv_dir = out / base
        csv_dir.mkdir(exist_ok=True)
        for arr_name, arr in arrays.items():
            path = csv_dir / f"{arr_name}.csv"
            np.savetxt(str(path), arr if arr.ndim == 2 else arr.reshape(-1, 1), delimiter=",")
            written.append(str(path))

    if "pt" in formats:
        import torch

        tensors = {k: torch.from_numpy(v) for k, v in arrays.items()}
        path = out / f"{base}.pt"
        torch.save(tensors, str(path))
        written.append(str(path))

    return written
