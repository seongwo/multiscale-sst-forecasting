# Research figures

The current figures follow the revision manuscript numbering. Existing public filenames are retained where they describe the same topic.

| Figure | File | Scope |
| --- | --- | --- |
| 1 | [overview_map_with_currents_refined.png](overview_map_with_currents_refined.png) | Study area and oceanographic background |
| 2 | [architecture.pdf](architecture.pdf) | Architecture and factorized Transformer |
| 3 | [results_fig1_daily_absolute_abc_bottom.png](results_fig1_daily_absolute_abc_bottom.png) | Monthly, lead-wise and daily-bias curves; five-fit metric averages |
| 4 | [result_winter_summer_rmse_difference.png](result_winter_summer_rmse_difference.png) | Summer/winter pixel RMSE differences, baseline minus MSF-ST |
| 5 | [result_combined_scale_analysis_final.png](result_combined_scale_analysis_final.png) | Seed 42 scale outputs and Test-wide origin distributions |
| 6 | [result_attention_analysis_combined.png](result_attention_analysis_combined.png) | Spatial attention and token-count enrichment |
| S1 | [result_comparison_71.png](result_comparison_71.png) | Selected qualitative prediction/error comparison |
| S2 | [temporal_attention.png](temporal_attention.png) | Scale-wise cumulative temporal attention |
| S3 | [Analysis_gdl_D1357_560.png](Analysis_gdl_D1357_560.png) | Refinement and GDL comparison |

Figures 3–4 use per-fit metrics averaged over Seeds 42–46, rather than metrics of an averaged prediction ensemble. Figure 5 uses one fitted model; its distribution bands describe Test origins and are not confidence intervals over training seeds. The manuscript reports Figure 6/S2 attention averaged across five independent training runs and all 1,807 Validation inputs. Token-count enrichment is a reference ratio, not a contribution to accuracy.

These assets reproduce the selected manuscript illustrations. Numerical result records and evaluation definitions are in the [experiment documentation](../docs/experiment_record.md). Intermediate sample images and full prediction arrays remain external artifacts.
