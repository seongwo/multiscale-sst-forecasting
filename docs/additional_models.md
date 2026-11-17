# Additional comparison models

The added model entry point reads per-family JSON settings from `docs/run_configs/`. It uses the same environment, SST preparation, chronological splits, ocean mask and normalization as MSF-ST.

| CLI model | Architecture | Parameters | Objective | LR |
| --- | --- | ---: | --- | ---: |
| `piformer` | Official PiFormer, D64, FF10412, patch4 | 1,611,159 | Masked L1 | 0.001 |
| `earthformer` | Cuboid Transformer, base64, two stages | 1,393,525 | Masked MSE | 0.001 |
| `vit_gru` | ViT depth3, D128, GRU forecasting | 1,920,785 | Masked MSE | 0.0001 |
| `single_scale` | MSF-ST with patch4 only, D152 | 1,660,656 | Masked MSE + 0.5 GDL | 0.0001 |

Parameter totals are implementation counts; manuscript tables display millions at their reported precision. All presets use 14 inputs, 7 outputs, a 32×32 grid, batch 64, AdamW with zero weight decay, at most 200 epochs, and patience 20. Validation and Test retain the final incomplete batch. Checkpoint selection uses global Validation RMSE.

## PiFormer

The SST adapter loads the official implementation from a pinned checkout:

```bash
git clone https://github.com/ADMIS-TONGJI/PiFormer.git external/PiFormer
git -C external/PiFormer checkout fc922dca8199eafe062e7b24cb09913aeaffc77c
python -m model_train.base.baseline_train --model piformer --check-model --device cpu
python -m model_train.base.baseline_train --model piformer --seed 42 --device cuda
```

The checkout is excluded from Git. Only upstream `models/PiFormer.py` and `layers/Embed.py` are loaded; both SHA-256 values are verified. No additional Python packages are needed beyond this repository's environment. Use `--piformer-root /path/to/PiFormer` for an existing checkout.

The adapter flattens SST into 1,024 nodes, adds the official core's terminal padding node, and removes it from the output. Original patch/query indexing is retained. Output shape is `(B,7,1,32,32)`.

The capacity-matched model uses FF10412, activation checkpointing and a 32-node feedforward chunk, with no new parameters. Chunking partitions dropout random-number draws and is part of the common execution policy across seeds. Training uses L1 with GDL disabled.

## Earthformer

The bundled core is from [Amazon Science's earth-forecasting-transformer](https://github.com/amazon-science/earth-forecasting-transformer), commit `7732b03bdb366110563516c3502315deab4c2026`. Required core files are unmodified; the license and source hashes are in `model/vendor/earthformer/`. The adapter converts `(B,T,C,H,W)` into the core's `(B,T,H,W,C)` order and reverses the output permutation.

```bash
python -m model_train.base.baseline_train --model earthformer --check-model --device cpu
python -m model_train.base.baseline_train --model earthformer --seed 42 --device cuda
```

The preset uses absolute SST with the shared normalization, axial self-attention, cross1×1 attention, encoder/decoder depths `[1,1]`, base64, four heads and no global vectors. Attention/projection/feedforward dropout is 0.1. Training uses MSE with GDL disabled.

## ViT-GRU and Single-scale

```bash
python -m model_train.base.baseline_train --model vit_gru --check-model --device cpu
python -m model_train.base.baseline_train --model single_scale --check-model --device cpu
python -m model_train.base.baseline_train --model vit_gru --seed 42 --device cuda
python -m model_train.base.baseline_train --model single_scale --seed 42 --device cuda
```

ViT-GRU uses the existing torchvision-based implementation with the depth3 preset. Single-scale uses MSF-ST with patch4 and D152 for the parameter-matched comparison; this differs from the single-scale variants in manuscript Table5.

## Configuration and outputs

`--print-config` resolves a preset without reading SST or constructing a model. `--check-model` checks one forward pass and reports parameter count/output shape; it does not train or measure forecast accuracy. Seeds 42–46 select the corresponding JSON files. `--config` accepts another JSON containing a `settings` object.

Training accepts `--data-path`, `--mask-path`, `--device` and `--output-root`. Each run requires a fresh output directory. It writes the resolved configuration, normalization, Validation-selected checkpoint, history, complete Test predictions/targets and physical-unit Test metrics. Published result tables are preserved.

The existing recurrent and SimVP entry points keep their paths and require model-specific settings. See [reproducibility notes](reproducibility.md#baseline-entry-points).
