# Manuscript result tables

The CSV tables reproduce the numerical values reported in the revised MSF-ST manuscript and reviewer response. They are published result tables, not raw per-seed logs or output produced by the current invocation of a training script. Table mapping and precision are recorded in [table_sources.json](table_sources.json).

| File | Source |
| --- | --- |
| [overall.csv](overall.csv) | Table2: Seed42 overall comparison |
| [five_seed_summary.csv](five_seed_summary.csv) | Table3: independent Seeds42–46 |
| [seasonal.csv](seasonal.csv) | Table4: five-seed seasonal RMSE |
| [scale_configurations.csv](scale_configurations.csv) | Table5: single-scale and LOSO |
| [embedding.csv](embedding.csv) | Table6: Time/Scale embeddings |
| [refinement_gdl.csv](refinement_gdl.csv) | Table7: refinement and GDL weights |
| [spatial_pe_seed42.csv](spatial_pe_seed42.csv) | Spatial PE reviewer response |
| [data_manifest.json](data_manifest.json) | Data splits and normalization metadata |

The displayed SD is truncated to three decimal places as in the manuscript. The CSV preserves reported confidence intervals and adjusted p values without recalculating them from the truncated SD. Raw per-seed measurements are not inferred from the published summary.
