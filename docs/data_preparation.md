# Data preparation and contract

The experiments use regional OISST sea surface temperature sequences for the East China Sea. Processed SST data and the ocean mask are not included in the current release tree. A versioned download/crop/preprocessing recipe is still needed for full reproduction.

## Expected files

The default configuration uses:

```text
./data/oisst_ecs_2533_122130.zarr
./data/oisst_spatial_mask_ecs.npy
```

`build_datasets.open_oisst` uses `xarray.open_zarr` for a `.zarr` path and `xarray.open_dataset` otherwise. The default Zarr loading uses Dask; `dask[array]` is included in the requirements. Other formats may require a compatible xarray backend.

| Field | Contract |
|---|---|
| `sst` | Daily SST in physical units before normalization, ordered `(time, lat, lon)` |
| Spatial grid | Default experiments use 32 × 32 cells; data and mask must have identical coordinate order |
| Ocean mask | NumPy array `(lat, lon)`, boolean-compatible; `True` means valid ocean |
| Time | Sorted daily observations; the loader does not resample or detect missing calendar days |
| Valid values | SST must be finite at every mask-valid cell in every input/target window |

The dataset zeros masked land cells after normalization. Although the mean/std calculation excludes non-finite values and the missing-value sentinel, the window loader does not dynamically repair invalid ocean values. Handle those consistently in preprocessing and preserve the rule in the experiment manifest.

## Chronological splits

The current `configs.py` and `build_datasets.py` defaults agree:

| Split | Inclusive interval |
|---|---|
| Training | 1983-01-01 through 2015-12-31 |
| Validation | 2016-01-01 through 2020-12-31 |
| Test | 2021-01-01 through 2025-12-31 |

Windows are created separately inside each split, with 14 inputs followed by 7 targets. They do not cross split boundaries. A split containing `T` daily records produces `T - 14 - 7 + 1` candidate windows before the trainer drops incomplete batches.

Mean and standard deviation are calculated from valid training ocean observations only, then reused for all splits. `stats_cache` is written each time datasets are built; it is not read as a validated immutable cache. Preserve it alongside the configuration and preprocessing metadata for each run.

## Information needed for a reproducible data release

Record the OISST product/version, source, date coverage, coordinate bounds and order, units, handling of missing values, mask construction, preprocessing script version, and checksums. The current path name alone does not establish these details. Provide external artifact links only after the exact artifacts and redistribution terms have been checked.

Keep raw data, Zarr stores, NetCDF files, checkpoints, and prediction arrays outside Git history. See [evaluation notes](reproducibility.md) before comparing physical-unit metrics.
