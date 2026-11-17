"""Public default configuration for SST forecasting experiments.

Data, checkpoints, and generated results are intentionally not included in the
GitHub repository. Prepare the expected local data files under ``./data/`` or
override these paths in your own experiment config.
"""


def get_default_cfg():
    return {
        # Local data placeholders. See docs/data_preparation.md.
        "data_path": "./data/oisst_ecs_2533_122130.zarr",
        "mask_path": "./data/oisst_spatial_mask_ecs.npy",
        "stats_cache": "./checkpoints/msf_st_full4_gdl05_seed42_stats.json",
        "ckpt_dir": "./checkpoints",
        "results_dir": "./results",
        "results_csv": "./results/msf_st_full4_gdl05_seed42_results.csv",
        "var_name": "sst",

        # Sequence and split settings used by the paper experiments.
        "input_len": 14,
        "pred_len": 7,
        "train_start": "1983-01-01",
        "train_end": "2015-12-31",
        "val_start": "2016-01-01",
        "val_end": "2020-12-31",
        "test_start": "2021-01-01",
        "test_end": "2025-12-31",

        # Runtime and optimization defaults.
        "seed": 42,
        "device": "cuda",
        "batch_size": 64,
        "num_workers": 4,
        "max_epoch": 200,
        "patience": 20,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "eps_mape": 1e-3,
        "deterministic_algorithms": True,
        "train_drop_last": True,
        "val_test_drop_last": False,
        "evaluation_aggregation": "global",

        # Main revision setting: masked MSE + 0.5 * spatial GDL.
        # Set lambda_gdl=0.0 for the MSFv3-base ablation.
        "tmse": False,
        "logistic_k": 1.0,
        "logistic_midpoint": None,
        "lambda_gdl": 0.5,
        "alpha_tgdl": 0.0,
        "aux_coef": 0.0,
        "moe_monitor": False,

        # Proposed MSFv3 model defaults.
        "exp_name": "msf_st_full4_gdl05_seed42",
        "image_size": 32,
        "d_model": 128,
        "depth": 4,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "attn_dropout": 0.0,
        "spatial_scales": (2, 4, 8, 16),
        "fusion": "sum",
        "time_pool": "last",
        "use_pad": True,
        "use_pos_emb": False,
    }
