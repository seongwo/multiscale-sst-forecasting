# Reproducibility and execution notes

## Entry points and saved settings

Run commands from the repository root with module syntax:

```bash
python -m model_train.base.msf3_train --print-config
python -m model_train.base.msf3_train --seed 42 --device cuda --output-root ./outputs/seed42
```

`--data-path` and `--mask-path` override the local data locations. `--device cpu` selects CPU operation. Seeds 42–46 identify the repeated main-model fits. The main entry point uses deterministic algorithms, float32, AMP off and TF32 off. The model definition and masked training losses retain their original implementation.

`configs.py` contains the primary14 → 7, four-scale, sum-fusion, GDL 0.5 defaults. Saved [run configuration records](run_configs/MSF_ST/seed42.json) preserve the historical run IDs and architecture/optimization settings. These JSON records are reference data; the main training CLI reads `configs.py` and its supported overrides.

The primary training path uses AdamW, batch 64, at most 200 epochs, patience 20, LR 1e-4 and zero weight decay. Random sampling uses a seed-specific generator. Training drops its final incomplete batch; Validation and Test do not. `evaluation_aggregation="global"` selects the checkpoint using RMSE from all ocean-cell squared errors. Keeping `evaluation_aggregation="batch_mean"` and `val_test_drop_last=True` explicitly restores the earlier batch-mean evaluation convention for legacy comparisons.

## Outputs and units

Each seed has distinct checkpoint, normalization and CSV names under the selected output root. Preserve model state, configuration, training mean/std, input dates, ocean mask and preprocessing version together. A fresh run can depend on software/hardware and data preparation; setting the same architecture is not a claim to recover identical weights.

The shared trainer's CSV diagnostics use normalized targets. Global RMSE/MAE can be converted to°C with the recorded standard deviation. Its normalized-target MAPE is not the physical-unit MAPE in the reference result tables; use the evaluator below for physical metrics. Training loss logs remain batch diagnostics, while validation checkpoint selection and final Test RMSE use the configured full-population aggregation.

The final checkpoint is selected from Validation, then restored before Test evaluation. No Test RMSE is used as the checkpoint-selection criterion.

## Physical-unit saved-array evaluation

```bash
python -m model_train.base.evaluate_predictions \
  --predictions /path/to/test_predictions_normalized.npy \
  --targets /path/to/test_ground_truth_normalized.npy \
  --mask /path/to/ocean_mask.npy \
  --metadata docs/results/data_manifest.json \
  --expected-samples 1806 \
  --output ./outputs/evaluation.json
```

Input arrays have shape `(N,7,1,32,32)` or `(N,7,32,32)`. Predictions and targets must use identical sample ordering, target dates, mask and normalization. The evaluator reads all samples in bounded batches, calculates RMSE, MAE, MAPE and gradient magnitude error in physical units, and records input SHA-256 hashes. `--units celsius` supports already-denormalized fields. An existing output JSON is not overwritten.

This command evaluates saved predictions; it does not run a model or select a checkpoint. Array hashes in [data_manifest.json](results/data_manifest.json) identify the saved ground truth and prepared data. Checkpoints/predictions are optional external artifacts and are not bundled in Git.

## Baseline entry points

PiFormer, Earthformer, ViT-GRU and parameter-matched Single-scale use the preset-driven runner:

```bash
python -m model_train.base.baseline_train --model earthformer --print-config
python -m model_train.base.baseline_train --model earthformer --seed 42 --device cuda
```

Select `piformer`, `vit_gru` or `single_scale` with the same command. PiFormer requires its pinned official checkout; [additional model setup](additional_models.md) provides commands, adapters and objectives. The runner reads the family/seed JSON and exports physical-unit metrics after restoring the selected checkpoint.

Existing files in `model_train/base/` retain their earlier layout. Model-specific settings must be supplied independently of MSF-ST defaults:

| Entry point | Additional settings |
| --- | --- |
| `convlstm_train` | `num_layers`, `num_hidden` and compatible model settings |
| `simvp_train` | `hid_S`, `hid_T`, `N_S`, `N_T`, `model_type` |
| `vit_gru_train` | `embed_dim`, `patch_size`, `vit_dropout`, `vit_attn_dropout`, `vit_freeze` |
| `MIM_train`, `MAU_train`, `predrnn_train`, `predrnnv2_train`, `swinlstm_train` | Check script defaults, objective, dimensions and output settings |

Do not inherit MSF-ST's GDL coefficient for a baseline whose recorded objective is MSE or L1 alone. [Per-family settings](experiment_record.md#additional-comparison-models) and the new runner implement the added models. Use `baseline_train --model vit_gru` for the paper depth3 preset; the older `vit_gru_train` retains its original settings. Full ablation scheduling remains separate from model entry points.

## Environment and verification scope

The earlier release recorded Python 3.10.19, PyTorch 2.9.1, NumPy 2.2.6, xarray 2025.6.1, Zarr 2.18.3, timm 0.9.8 and einops 0.8.1. `requirements.txt` lists dependencies; it is not a fresh-install environment lock. CPU checks cover core imports, model shape, masked metrics, unequal final batches, a synthetic checkpoint cycle and saved-array evaluation for Seeds 42 and 43. These checks do not establish identical full GPU retraining on other software/hardware.

Record software versions and hardware with a new run. Keep raw data, checkpoints, prediction arrays, generated logs and machine-specific settings outside the published source tree.
