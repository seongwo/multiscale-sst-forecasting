# Reproducibility and execution notes

## What has been checked

On 2026-09-23, the existing research environment passed Python syntax checks, core module imports, and CPU forward passes for `MSFv3`, `MSFGv3`, and `MSFG2v3`. The forward check used `d_model=32`, one block, and input `(1, 14, 1, 32, 32)`; every output was finite with shape `(1, 7, 1, 32, 32)`. Known-value masked MSE/RMSE/MAE and GDL checks also passed.

| Component | Version present during checks |
|---|---|
| Python | 3.10.19 |
| PyTorch / torchvision | 2.9.1 / 0.24.1 |
| NumPy | 2.2.6 |
| xarray / Zarr | 2025.6.1 / 2.18.3 |
| timm / einops | 0.9.8 / 0.8.1 |
| Dask | 2026.3.0 |

This is an observed environment, not a verified fresh-install lockfile. Full training, GPU behavior, all baseline forwards, and version combinations other than this environment were not tested. The unpinned requirements alone do not establish exact paper reproducibility.

## Entry points and artifacts

Run from the repository root with module syntax:

```bash
python -m model_train.base.msf3_train
```

This makes root-level imports such as `configs` and `trainer` available. Direct file execution from `model_train/base` can fail to find them.

The entry point reads `configs.py`; it does not expose a command-line configuration parser. Set the device explicitly, provide the data and mask, and select a unique experiment/output name before running. The script moves the model to `cfg["device"]` before the trainer's CPU fallback, so a CUDA default requires CUDA.

Checkpoint output includes model state, configuration, best validation RMSE, and epoch. Normalization mean/std are saved separately to `stats_cache`. Preserve both with the data preprocessing version and commit ID. CSV filenames and checkpoint names are shared across scripts unless changed in the configuration; reusing them can mix rows or overwrite checkpoints.

## Metrics and evaluation support

`OISSTDataset` z-normalizes inputs and targets using training-set ocean statistics. `trainer.py` evaluates those normalized tensors directly:

- Its RMSE and MAE have normalized units; its MAPE uses normalized targets as denominators. These are not the physical-unit paper metrics.
- The trainer averages per-batch metrics, including RMSE and R². Mean batch RMSE is generally different from a global RMSE calculated from all squared errors.
- Train, validation, and test loaders currently use `drop_last=True`. Incomplete batches are omitted. The archived 14 → 7 evaluation contains 1,792 windows; sample selection must be preserved when comparing models.
- Validation RMSE selects the checkpoint. The test set is evaluated after that checkpoint is restored.

For physical-unit evaluation, denormalize **both** predictions and targets with the exact run's training mean/std, retain identical sample IDs and ocean mask for all models, and aggregate globally. MAPE must be recomputed after denormalization; multiplying normalized MAPE by the standard deviation is invalid.

These behaviors are documented without changing the existing trainer or published results. A future evaluator should record sample IDs, units, mask, normalization, aggregation, checkpoint hash, and commit together. See the [archived evidence](experiment_record.md).

## Baseline entry points

The files retain research implementations and are not complete experiment presets. In particular:

| Entry point | Configuration gaps visible in the script |
|---|---|
| `convlstm_train` | Needs `num_layers` and `num_hidden`, plus compatible model settings |
| `simvp_train` | Needs `hid_S`, `hid_T`, `N_S`, `N_T`, and `model_type` |
| `vit_gru_train` | Needs `embed_dim`, `patch_size`, `vit_dropout`, `vit_attn_dropout`, and `vit_freeze` |
| `MIM_train`, `MAU_train`, `predrnn_train`, `predrnnv2_train`, `swinlstm_train` | Supply several defaults internally; verify their shape, objective, and output settings before a comparison |

This list is an entry-point inspection, not an exhaustive validation of constructor requirements. Do not infer a baseline's paper settings from the proposed-model default. For example, inheriting `lambda_gdl=0.3` would change a baseline intended to use MSE alone. Matched baseline configuration files and a standalone physical-unit evaluator remain release work.

## Public artifacts

Keep raw/processed data, checkpoints, prediction arrays, logs, caches, credentials, and machine-specific configuration outside Git. The current release tree contains curated code and figures; older experimental branches retain large arrays and notebook outputs. Cleaning those branches requires a separate history/publication decision and has not been performed here.
