# multiscale-sst-forecasting

This repository contains public release code for sea surface temperature (SST) sequence forecasting experiments associated with an academic paper. The cleanup preserves the original project layout and imports so reviewers can inspect and run the proposed model and baseline training scripts without a package migration.

Large artifacts are intentionally excluded from GitHub: raw or processed SST data, checkpoints, logs, prediction arrays, cache files, and generated sample image folders.

## Repository Layout

```text
.
├── configs.py                  # stable public default config
├── dataset.py                  # OISST sequence Dataset
├── build_datasets.py           # xarray/zarr loading and train/val/test splits
├── losses.py                   # masked metrics and GDL variants
├── trainer.py                  # unified trainer for normal, aux, and MoE-monitor paths
├── model/                      # model implementations and modules
├── model_train/                # proposed and baseline training scripts
├── notebooks/00_quickstart.ipynb
├── docs/data_preparation.md
└── figures/                    # final paper figures selected for release
```

## Installation

```bash
pip install -r requirements.txt
```

Use any Python environment with the dependencies in `requirements.txt`.

## Data Preparation

Training expects local OISST-style files, which are not included:

```text
./data/oisst_ecs_2533_122130.zarr
./data/oisst_spatial_mask_ecs.npy
```

See `docs/data_preparation.md` for the expected layout. Keep large data outside Git history.

## Quickstart Notebook

Open:

```text
notebooks/00_quickstart.ipynb
```

The notebook runs only a dummy forward pass and masked loss checks. It does not require real SST data and does not run training.

## Training Commands

After preparing data locally and reviewing `configs.py`, train the proposed model with:

```bash
python model_train/base/msf3_train.py
```

Baseline examples:

```bash
python model_train/base/convlstm_train.py
python model_train/base/simvp_train.py
python model_train/base/MIM_train.py
python model_train/base/predrnn_train.py
python model_train/base/predrnnv2_train.py
```

Do not run training scripts until `data_path`, `mask_path`, checkpoint output, and result output locations in `configs.py` are appropriate for your machine.

## Checkpoint And Artifact Policy

Do not commit:

- `data/`
- `checkpoints/`
- `logs/`
- `model_comparison/`
- `results/` unless reduced to a curated final table
- `*.pt`, `*.pth`, `*.ckpt`, `*.npy`, `*.npz`, `*.zarr/`
- Python caches or notebook checkpoints

If checkpoints or prediction arrays are needed for review, publish them as external artifacts and link them from the paper or release notes.

## Citation

`CITATION.cff` contains placeholder author and repository fields. Update it with the final paper metadata before release.

## Manual Notes

- `trainer.py` is the unified trainer for normal training, optional STGDL/TGDL, optional auxiliary loss, and optional MoE routing summaries.
