# MSF-ST · Multi-Scale SST Forecasting

Sea surface temperature prediction in the East China Sea. First-author research by **Seongwoo Lim**.

Sea surface temperature fields contain both broad regional changes and local temperature fronts. I studied whether representing these structures at several spatial scales improves forecasting. MSF-ST combines a multi-scale Transformer with a gradient difference loss, and evaluates the design through baseline comparisons and ablations.

The implementation retains the research name `MSFv3` in Python classes and filenames. **MSF-ST** is the manuscript/project name.

## Method

- **Task:** forecast 7 daily SST fields from 14 past observations on a 32 × 32 regional grid.
- **Representation:** patch tokenization at spatial scales 2, 4, 8, and 16, with time and scale embeddings.
- **Architecture:** concatenate tokens across scales, apply factorized spatial/temporal attention, decode forecasts per scale, then fuse and refine them.
- **Objective:** ocean-masked MSE plus spatial Gradient Difference Loss (GDL); the reported setting uses a GDL coefficient of 0.3.
- **Evaluation:** comparisons with statistical and neural baselines, scale/component ablations, and spatial error analysis.
- **Contribution:** MSF-ST model and experiments. Baseline implementations are credited in [THIRD_PARTY.md](THIRD_PARTY.md).

[Architecture figure](figures/architecture.pdf) · [Implementation guide](docs/method.md) · [Experiment evidence](docs/experiment_record.md)

## Results

| Reported MSF-ST result | Value |
|---|---:|
| RMSE | 0.5295 °C |
| MAE | approximately 0.3693–0.3694 °C |
| MAPE | 1.7132% |

A check of the available archived predictions reproduces RMSE 0.5295 °C and MAE 0.3694 °C after rounding. It gives MAPE 1.7130%, slightly different from the reported value. The [experiment record](docs/experiment_record.md) preserves both, identifies the artifacts and normalization used, and records what remains unresolved.

The code and a data-free forward example are available. Reproducing the result table still requires the matching checkpoint, configuration, preprocessing record, and evaluation manifest. The current `gated2` default has not been matched to the reported run. Training CSV metrics are computed on normalized data; the table above uses °C. [Reproduction notes](docs/reproducibility.md) describe these differences.

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
2. Review `configs.py`, including fusion, device, data paths, split dates, and output names. Use `device="cpu"` if CUDA is unavailable.
3. Start the proposed-model entry point from the repository root:

```bash
python -m model_train.base.msf3_train
```

Use a distinct `exp_name`, `results_csv`, and normalization statistics path for each experiment so existing artifacts are not overwritten or mixed. Model and optimizer behavior are preserved from the research code.

Baseline entry points are in `model_train/base/`. Some require model-specific fields absent from the proposed-model default, so they are not all ready to run with an unchanged `configs.py`. See [baseline configuration notes](docs/reproducibility.md#baseline-entry-points) before running them. Statistical baseline evaluation and the full ablation runners are not included in this release.

## Repository map

| Path | Purpose |
|---|---|
| `model/models/msf3_model.py` | Multi-scale tokenization, factorized attention, per-scale decoding, fusion |
| `losses.py` | Ocean-masked metrics and spatial/temporal GDL variants |
| `dataset.py`, `build_datasets.py` | Chronological splits, training-set normalization, sliding windows |
| `trainer.py` | Training, validation selection, checkpointing, normalized metrics |
| `model_train/base/` | Proposed-model and neural baseline entry points |
| `docs/` | Method, data contract, evidence, and reproducibility limits |
| `figures/` | Selected research figures; [index](figures/README.md) |

Data, checkpoints, prediction arrays, and generated outputs are excluded from the current release tree. Older experimental branches may still contain large artifacts; the clean release tree does not remove those branches or their history.

## Citation and credits

[CITATION.cff](CITATION.cff) records the manuscript title and authors. Full publication metadata will be added when verified; no DOI or acceptance status is asserted here.

Baseline code includes CAIRI AI Lab notices and upstream model sources. See [third-party credits and license status](THIRD_PARTY.md). A repository-wide license has not yet been established.
