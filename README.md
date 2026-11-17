# MSF-ST · Multi-Scale SST Forecasting

Sea surface temperature prediction in the East China Sea. First-author research by **Seong Woo Lim**.

Sea surface temperature fields contain both broad regional changes and local temperature fronts. I studied whether representing these structures at several spatial scales improves forecasting. MSF-ST combines a multi-scale Transformer with a gradient difference loss, and evaluates the design through baseline comparisons and ablations.

The implementation retains the research name `MSFv3` in Python classes and filenames. **MSF-ST** is the manuscript/project name.

## Method

- **Task:** forecast 7 daily SST fields from 14 past observations on a 32 × 32 regional grid.
- **Representation:** patch tokenization at spatial scales 2, 4, 8, and 16, with time and scale embeddings.
- **Architecture:** concatenate tokens across scales, apply factorized spatial/temporal attention, decode forecasts per scale, then fuse and refine them.
- **Objective:** ocean-masked MSE plus spatial Gradient Difference Loss (GDL), with a coefficient of 0.5 in the current revision experiments.
- **Evaluation:** comparisons with statistical and neural baselines, scale/component ablations, and spatial error analysis.
- **Contribution:** MSF-ST model and experiments. Baseline implementations are credited in [THIRD_PARTY.md](THIRD_PARTY.md).

[Architecture figure](figures/architecture.pdf) · [Implementation guide](docs/method.md) · [Experiment evidence](docs/experiment_record.md)

## Results

| MSF-ST, manuscript Table 2 (Seed 42) | Value |
| --- | ---: |
| RMSE | 0.5265 °C |
| MAE | 0.3663 °C |
| MAPE | 1.7132% |
| GDL | 0.1797 °C/grid edge |

These values are reported in manuscript Table 2. The five-seed mean RMSE is **0.5334 °C** (Table 3). [Manuscript tables](docs/results/README.md) include the overall, statistical, seasonal and ablation results at their published precision; [model configurations](docs/run_configs/MSF_ST/seed42.json) describe the architecture and training settings.

The default model uses four scales, sum fusion, and learned time/scale embeddings, with explicit spatial positional embeddings disabled. [Experiment details](docs/experiment_record.md) explain the saved evaluation scope, baseline settings, and additional embedding comparison.

## Quickstart without SST data

Run commands from the repository root. Python 3.10 was used for the CPU checks recorded in [reproducibility notes](docs/reproducibility.md).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python - <<'PY'
import torch
from model.models.msf3_model import MSFv3

model = MSFv3(
    image_size=32, tin=14, tout=7, d_model=32, depth=1,
    num_heads=4, spatial_scales=(2, 4, 8, 16),
).eval()
with torch.no_grad():
    prediction = model(torch.randn(1, 14, 1, 32, 32))
assert prediction.shape == (1, 7, 1, 32, 32)
assert torch.isfinite(prediction).all()
print(prediction.shape)
PY
```

This is a randomly initialized, reduced-size shape check. It does not evaluate forecast accuracy. The output-cleared [quickstart notebook](notebooks/00_quickstart.ipynb) also exercises masked losses. Launch it from the repository root with `PYTHONPATH=. jupyter notebook notebooks/00_quickstart.ipynb` so its imports can find the source files.

## Training with prepared data

1. Prepare the SST store and ocean mask using the [data contract](docs/data_preparation.md).
2. Review `configs.py`: 14 input days, 7 forecast days, sum fusion and GDL 0.5. Choose data paths and an output location.
3. Start the proposed-model entry point from the repository root:

```bash
python -m model_train.base.msf3_train --seed 42 --device cuda --output-root ./outputs/seed42
```

Use `--print-config` to inspect settings without loading data, or `--device cpu` for a CPU run. Each seed gets distinct checkpoint, normalization and CSV names under the selected output root. Validation and Test retain the final smaller batch; validation selection uses global ocean-cell RMSE. Training CSV errors are in normalized units; the result table above uses physical units.

For saved normalized predictions, evaluate all samples in °C:

```bash
python -m model_train.base.evaluate_predictions \
  --predictions /path/to/test_predictions_normalized.npy \
  --targets /path/to/test_ground_truth_normalized.npy \
  --mask /path/to/ocean_mask.npy \
  --metadata docs/results/data_manifest.json
```

The evaluator checks the expected 1,806 samples, includes every final batch, and reports RMSE, MAE, MAPE and gradient magnitude error. Use matching arrays and normalization metadata from the same experiment. Data/checkpoints/predictions must be supplied separately.

Additional comparison models have a configuration-driven entry point:

```bash
python -m model_train.base.baseline_train --model earthformer --seed 42 --check-model --device cpu
python -m model_train.base.baseline_train --model piformer --seed 42 --piformer-root ./external/PiFormer
python -m model_train.base.baseline_train --model vit_gru --seed 42
python -m model_train.base.baseline_train --model single_scale --seed 42
```

PiFormer loads the pinned official source from a separate checkout; see [additional model setup](docs/additional_models.md). Each family uses its own objective and learning rate. Existing recurrent and SimVP entry points retain their paths; see [baseline configuration notes](docs/reproducibility.md#baseline-entry-points).

## Repository map

| Path | Purpose |
| --- | --- |
| `model/models/msf3_model.py` | Multi-scale tokenization, factorized attention, per-scale decoding, fusion |
| `losses.py` | Ocean-masked metrics and spatial/temporal GDL variants |
| `dataset.py`, `build_datasets.py` | Chronological splits, training-set normalization, sliding windows |
| `trainer.py` | Training, full-population validation selection, checkpointing and normalized diagnostics |
| `model_train/base/` | Proposed-model and neural baseline entry points |
| `model_train/base/baseline_train.py` | Paper presets for PiFormer, Earthformer, ViT-GRU and Single-scale |
| `model_train/base/evaluate_predictions.py` | Physical-unit evaluation of saved prediction arrays |
| `docs/` | Method, data contract, evidence, and reproducibility limits |
| `docs/results/`, `docs/run_configs/` | Manuscript result tables and per-seed model configurations |
| `figures/` | Selected research figures; [index](figures/README.md) |

Data, checkpoints, prediction arrays, and generated outputs are excluded from the current release tree. Older experimental branches may still contain large artifacts; the clean release tree does not remove those branches or their history.

## Citation and credits

[CITATION.cff](CITATION.cff) records the manuscript title and authors. Full publication metadata will be added when verified; no DOI or acceptance status is asserted here.

Baseline code includes CAIRI AI Lab notices and upstream model sources. See [third-party credits and license status](THIRD_PARTY.md). A repository-wide license has not yet been established.
