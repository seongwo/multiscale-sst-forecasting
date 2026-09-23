# Experiment evidence and open provenance questions

This record separates the author's reported MSF-ST result, an existing local report, and a new arithmetic check of archived arrays. The check does not retrain the model or revise the reported results.

## Reported result

The author reports RMSE **0.5295 °C**, MAE approximately **0.3693–0.3694 °C**, and MAPE **1.7132%** for MSF-ST. The local `reports/traditional_baseline_report_ko.md` records 0.5295, 0.3693, and 1.7132 respectively. That report also lists Persistence, Moving Average, Exponential Smoothing, and Pixelwise ARIMA comparisons. Neural baseline code is included in this release, but a complete public table with matched evaluation records has not yet been assembled.

The reported experiment uses 14 input days and 7 forecast days. Archived comparison arrays have shape `(1792, 7, 1, 32, 32)`. The current split is train 1983–2015, validation 2016–2020, and test 2021–2025. This split agrees with the traditional-baseline report.

## Archived-array check, 2026-09-23

Available local artifacts were read without modification:

| Artifact | SHA-256 |
|---|---|
| `model_comparison/msfv3_gdl03_preds.npy` | `139eb848dd51667b184d3028a0a59fd191d4ac9fa7affe0368664543ce697ba9` |
| `model_comparison/ground_truth.npy` | `2913231a8e297b3b61a413d5b654579784e51addda6d1468090b3e1b5f4bbe41` |
| `data/oisst_spatial_mask_ecs.npy` | `dafd4540f1691d4a50b85d5774b150e1eaa2916bead3c559ff3683e7c4fb5af2` |

These large artifacts are not bundled in the current release tree. Relative names identify the local evidence; they are not download links.

The archived evaluation notebook `result2.1.ipynb` defines global ocean-masked metrics and denormalization constants `mean=22.5081844329834`, `std=4.8933515548706055`. Applying those constants to the arrays above, using float64 accumulation, all 1,792 windows, all seven horizons, and the static ocean mask gives:

| Metric | Archived-array check | Rounded to four decimals |
|---|---:|---:|
| RMSE (°C) | 0.5294666950852455 | 0.5295 |
| MAE (°C) | 0.3693618462822876 | 0.3694 |
| MAPE (%) | 1.7130078607543084 | 1.7130 |

There are 12,807,424 evaluated scalar targets: 1,792 × 7 × 1,021 ocean cells. MAPE uses `max(abs(target_C), 1e-6)` in its denominator. RMSE is the square root of globally averaged squared error; MAE and MAPE are globally averaged as well.

This supports the reported RMSE, but does not exactly recover the report's MAE rounding or MAPE. A separate traditional-baseline metadata file records different normalization statistics (`mean=22.508429888017154`, `std=4.889253383831301`). Their relationship to the neural prediction export needs to be reconciled before claiming one completely reproduced comparison table.

## Configuration identity is still unresolved

The public `configs.py` selects `fusion="gated2"`; the archived experiments also contain sum-fusion variants. The checkpoint corresponding to `msfv3_gdl03_preds.npy` was not present at the expected local checkpoint path during this review. A filename alone cannot establish the model configuration.

The current full-size model has 1,611,132 parameters with sum fusion and 1,677,312 with `gated2`. The old report lists 1,611,130 for MSF-ST. This is another reason to recover the precise checkpoint and configuration rather than infer them or silently change public defaults.

No new training was performed, no experiment value was overwritten, and the current model/trainer/configuration was preserved. The next reproducibility step is to associate the final manuscript table with an exact checkpoint, prediction export, mask, normalization record, sample IDs, and evaluation script.
