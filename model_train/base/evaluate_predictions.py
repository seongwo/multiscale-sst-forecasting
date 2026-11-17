"""Evaluate saved SST predictions over all ocean cells in physical units."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_5d(array):
    if array.ndim == 4:
        return array[:, :, None]
    if array.ndim == 5 and array.shape[2] == 1:
        return array
    raise ValueError("Expected (N,T,H,W) or (N,T,1,H,W) arrays")


def evaluate(predictions, targets, mask, mean, std, *, units="normalized", batch_size=64):
    predictions, targets = as_5d(predictions), as_5d(targets)
    mask = np.asarray(mask, dtype=bool)
    if predictions.shape != targets.shape or tuple(predictions.shape[-2:]) != mask.shape:
        raise ValueError("Prediction, target and mask shapes are inconsistent")
    if not mask.any() or not len(predictions) or batch_size < 1:
        raise ValueError("Non-empty arrays, an ocean mask and a positive batch size are required")
    if units not in {"normalized", "celsius"}:
        raise ValueError("units must be 'normalized' or 'celsius'")
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Training mean/std must be finite, with std > 0")
    mx, my = mask[:, 1:] & mask[:, :-1], mask[1:] & mask[:-1]
    if not mx.any() or not my.any():
        raise ValueError("The mask must contain valid edges in both spatial directions")
    horizon = predictions.shape[1]
    sse, sae, sape, gx, gy = (np.zeros(horizon, dtype=np.float64) for _ in range(5))
    scale = std if units == "normalized" else 1.0
    offset = mean if units == "normalized" else 0.0
    for first in range(0, len(predictions), batch_size):
        p = np.asarray(predictions[first:first + batch_size], dtype=np.float64)
        t = np.asarray(targets[first:first + batch_size], dtype=np.float64)
        if not np.isfinite(p[..., mask]).all() or not np.isfinite(t[..., mask]).all():
            raise ValueError("Non-finite SST at an ocean cell")
        error = (p[..., mask] - t[..., mask]) * scale
        sse += np.square(error).sum((0, 2, 3))
        sae += np.abs(error).sum((0, 2, 3))
        denominator = np.maximum(np.abs(t[..., mask] * scale + offset), 1e-3)
        sape += (100 * np.abs(error) / denominator).sum((0, 2, 3))
        gdx = np.abs(np.abs(np.diff(p, axis=-1)) - np.abs(np.diff(t, axis=-1)))
        gdy = np.abs(np.abs(np.diff(p, axis=-2)) - np.abs(np.diff(t, axis=-2)))
        gx += gdx[..., mx].sum((0, 2, 3)) * scale
        gy += gdy[..., my].sum((0, 2, 3)) * scale
    count = len(predictions) * int(mask.sum())
    mse_t = sse / count
    gradient_t = gx / (len(predictions) * mx.sum()) + gy / (len(predictions) * my.sum())
    return {
        "samples": len(predictions),
        "horizon": horizon,
        "ocean_cells": int(mask.sum()),
        "rmse_c": float(np.sqrt(mse_t.mean())),
        "mae_c": float(sae.sum() / (count * horizon)),
        "mape_percent": float(sape.sum() / (count * horizon)),
        "gdl_c_per_edge": float(gradient_t.mean()),
        "rmse_t_c": np.sqrt(mse_t).tolist(),
        "mae_t_c": (sae / count).tolist(),
        "gdl_t_c_per_edge": gradient_t.tolist(),
        "aggregation": "Global ocean-cell sums across all supplied samples and leads",
        "gradient_definition": "Mean horizontal magnitude error plus mean vertical magnitude error",
        "mape_denominator_floor_c": 1e-3,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True, help="JSON containing training mean and std")
    parser.add_argument("--units", choices=["normalized", "celsius"], default="normalized")
    parser.add_argument("--expected-samples", type=int, default=1806)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output already exists; select a new file")
    metadata = json.loads(args.metadata.read_text())
    mean = metadata.get("mean", metadata.get("train_mean_c"))
    std = metadata.get("std", metadata.get("train_std_c"))
    if mean is None or std is None:
        parser.error("Metadata must include mean/std or train_mean_c/train_std_c")
    pred = np.load(args.predictions, mmap_mode="r", allow_pickle=False)
    target = np.load(args.targets, mmap_mode="r", allow_pickle=False)
    if len(pred) != args.expected_samples:
        parser.error(f"Expected {args.expected_samples} samples, found {len(pred)}")
    mask = np.load(args.mask, allow_pickle=False)
    result = evaluate(pred, target, mask, float(mean), float(std), units=args.units, batch_size=args.batch_size)
    result["input_units"] = args.units
    result["normalization"] = {"train_mean_c": mean, "train_std_c": std}
    result["input_sha256"] = {name: sha256(getattr(args, name)) for name in ("predictions", "targets", "mask", "metadata")}
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
