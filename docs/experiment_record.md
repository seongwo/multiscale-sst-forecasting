# Experiment records

## Manuscript results

[Result tables](results/README.md) reproduce the revised manuscript Tables 2–7 and the spatial-PE reviewer-response table. Table 2 reports Seed 42; Tables3–4 summarize independent Seeds 42–46. Values retain the manuscript's reporting precision. The five-seed mean is a mean of independently evaluated model RMSEs, not an ensemble-prediction RMSE.

MSF-ST uses scales 2,4,8,16, D128, four blocks/four heads, time/scale embeddings, no explicit spatial positional embeddings, sum fusion, CNN refinement and masked MSE + 0.5×GDL. Table 2 reports RMSE 0.5265°C, MAE 0.3663°C, MAPE 1.7132% and GDL 0.1797°C/grid edge; Table 3 reports mean RMSE 0.5334°C.

## Additional comparison models

| Model | Configuration | Training objective | Learning rate |
| --- | --- | --- | --- |
| PiFormer, parameter matched | [D64, FF10412](run_configs/PiFormer/seed42.json) | Masked L1 | 0.001 |
| Earthformer | [Two encoder/decoder stages](run_configs/Earthformer/seed42.json) | Masked MSE | 0.001 |
| ViT-GRU | [Three ViT layers, D128](run_configs/ViT_GRU/seed42.json) | Masked MSE | 0.0001 |
| Single-scale | [Patch4, D152](run_configs/Single_D152/seed42.json) | Masked MSE + 0.5×GDL | 0.0001 |

[Additional model setup and commands](additional_models.md) describe the official-core SST adapters, exact configuration loading and output shapes. All models share the chronological data split,14→7 task,32×32 ocean mask, training normalization, batch 64, AdamW, maximum 200 epochs and patience 20. The environment and data pipeline are shared with MSF-ST.

## Evaluation

- Training normalization uses ocean-only training mean/std. Dataset split metadata are in [data_manifest.json](results/data_manifest.json).
- Evaluation includes all 1,806 Test windows and all seven forecast leads; Validation contains 1,807 windows.
- RMSE is computed from global ocean-cell squared-error sums; MAE and MAPE use physical°C values, with a 0.001°C denominator floor for MAPE.
- GDL is the sum of the separately normalized horizontal and vertical gradient-magnitude errors on valid ocean-cell pairs.
- Validation global RMSE selects the checkpoint; Test does not select hyperparameters/checkpoints.

The numerical tables contain manuscript-reported results. A new execution writes its own checkpoint, resolved configuration, history and measured metrics under the requested output directory; it does not overwrite these tables.
