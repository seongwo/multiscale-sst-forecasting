# Data Preparation

The original experiments use OISST sea surface temperature sequences on a regional grid. Data files are not included in this GitHub release.

Expected local paths in the current config:

```text
./data/oisst_ecs_2533_122130.zarr
./data/oisst_spatial_mask_ecs.npy
```

Expected dataset shape:

```text
sst: (time, lat, lon)
mask: (lat, lon), boolean-compatible ocean mask
```

The default split in `configs.py` is:

```text
train: 1983-01-01 through 2015-12-31
val:   2016-01-01 through 2020-12-31
test:  2021-01-01 through 2025-12-31
```

For a public release, provide one of the following:

1. A script that downloads or preprocesses public OISST data into the expected zarr layout.
2. A small synthetic fixture for CI and examples.
3. External artifact links for trained checkpoints and processed prediction arrays, if redistribution is allowed.

Do not commit raw OISST zarr stores, NetCDF files, large TIFFs, checkpoints, or prediction arrays to this repository.
