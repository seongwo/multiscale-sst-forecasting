from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import xarray as xr

from dataset import OISSTDataset, MISSING_VALUE

def open_oisst(path: str | Path):
    path = str(path)
    if path.endswith(".zarr"):
        return xr.open_zarr(path)
    return xr.open_dataset(path)

def compute_mean_std_train(ds_train: xr.Dataset, mask: np.ndarray, var_name: str = "sst") -> Tuple[float, float]:
    arr = ds_train[var_name].values.astype(np.float32)  # (T, H, W)

    valid = np.isfinite(arr) & (arr != MISSING_VALUE)
    valid &= mask[None, :, :]

    x = arr[valid].astype(np.float64)
    if x.size == 0:
        raise ValueError("No valid ocean points found for mean/std computation. Check mask and missing values.")

    mean = float(x.mean())
    std = float(x.std() + 1e-12)
    return mean, std

def build_datasets(
    data_path: str | Path,
    mask: np.ndarray,
    tin: int = 14,
    tout: int = 7,
    var_name: str = "sst",
    train_start: str = "1983-01-01",
    train_end: str = "2015-12-31",
    val_start: str = "2016-01-01",
    val_end: str = "2020-12-31",
    test_start: str = "2021-01-01",
    test_end: str = "2025-12-31",
    stats_cache_json: Optional[str | Path] = None,
):
    mask = mask.astype(bool)

    ds = open_oisst(data_path)

    ds_train = ds.sel(time=slice(train_start, train_end))
    ds_val = ds.sel(time=slice(val_start, val_end))
    ds_test = ds.sel(time=slice(test_start, test_end))

    mean, std = compute_mean_std_train(ds_train, mask, var_name=var_name)

    if stats_cache_json is not None:
        p = Path(stats_cache_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump({"mean": mean, "std": std}, f, indent=2)

    train_set = OISSTDataset(ds_train, mask, mean, std, tin=tin, tout=tout, var_name=var_name)
    val_set = OISSTDataset(ds_val, mask, mean, std, tin=tin, tout=tout, var_name=var_name)
    test_set = OISSTDataset(ds_test, mask, mean, std, tin=tin, tout=tout, var_name=var_name)

    torch_mask = torch.from_numpy(mask).bool()  
    
    return train_set, val_set, test_set, mean, std, torch_mask
