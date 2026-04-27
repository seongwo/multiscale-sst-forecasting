# configs.py
def get_default_cfg():

    return {

        # data paths
        "data_path": "./data/oisst_ecs_2533_122130.zarr",
        "mask_path": "./data/oisst_spatial_mask_ecs.npy",
        "stats_cache": "./checkpoints/train_stats.json",
        "ckpt_dir": "./checkpoints",

        "var_name": "sst",
        "input_len": 14,
        "pred_len": 7,

        "train_start": "1983-01-01",
        "train_end": "2015-12-31",
        "val_start": "2016-01-01",
        "val_end": "2020-12-31",
        "test_start": "2021-01-01",
        "test_end": "2025-12-31",

        "lr": 1e-4,
        "weight_decay": 0.0,
        "batch_size": 64,
        "num_workers": 4,
        "max_epoch": 200,
        "patience": 30,
        "seed": 42,
        "device": "cuda",

        "tmse": False,
        "lambda_gdl": 0.0,
        "alpha_tgdl": 0.0,


        # # simvp model parameters
        # "results_csv": "./results/simvp_incepu_results.csv",
        # "exp_name": "simvp_gsta",
        # "hid_S": 16,
        # "hid_T": 64,
        # "N_S": 4,
        # "N_T": 4,
        # "model_type": "incepu",  # "gsta" or "incepu"

        # convlstm parameters
        # "results_csv": "./results/convlstm_results.csv",
        # "exp_name": "convlstm",
        # "patch_size": 1,
        # "in_shape": (14, 1, 32, 32),
        # "filter_size": 3,
        # "stride": 1,
        # "layer_norm": False,
        # "num_layers": 3,
        # "num_hidden" : [64, 64, 64]


        # #swinLstm_D parameters
        # "results_csv": "./results/swinLstm_D_results.csv",
        # "exp_name": "swinLstm_D",
        # "depths_downsample": [2, 2, 2],
        # "depths_upsample": [2, 2, 2],
        # "num_heads": [2, 4, 8],
        # "in_shape": (14, 1, 32, 32),
        # "patch_size": 2, # 1 
        # "embed_dim": 32,
        # "window_size": 4,
    

        # swinLstm_B parameters
        # "results_csv": "./results/swinLstm_B_results.csv",
        # "exp_name": "swinLstm_B",
        # "in_shape": (14, 1, 32, 32),
        # "num_heads": 4,
        # "patch_size": 2,
        # "embed_dim": 128,
        # "window_size": 4,
        # "depths": 3

        # predrnn parameters
        "results_csv": "./results/predrnn_results.csv",
        "exp_name": "predrnn",
        "in_shape": (14, 1, 32, 32),

        "patch_size": 1,
        "num_layers": 3,
        "num_hidden": [64, 64, 64],

        "filter_size": 3,
        "stride": 1,
        "layer_norm": 0,

        # "reverse_scheduled_sampling": 0,



        # predrnnpp parameters
        # "results_csv": "./results/predrnnpp_results.csv",
        # "exp_name": "predrnnpp",
        # "in_shape": (14, 1, 32, 32),

        # "patch_size": 1,
        # "num_layers": 2,
        # "num_hidden": [32, 32],

        # "filter_size": 3,
        # "stride": 1,
        # "layer_norm": 0,


        # predrnnv2
        # "results_csv": "./results/predrnnv2_results.csv",
        # "exp_name": "predrnnv2",
        # "in_shape": (14, 1, 32, 32),

        # "patch_size": 1,
        # "num_layers": 3,
        # "num_hidden": [32, 32, 32],

        # "filter_size": 3,
        # "stride": 1,
        # "layer_norm": True,

        # "decouple_beta": 0.1,



        # MIM parameters (matched to convlstm scale)
        # "results_csv": "./results/MIM_results.csv",
        # "exp_name": "MIM",
        # "in_shape": (14, 1, 32, 32), 
        # "patch_size": 1,

        # "num_layers": 2,
        # "num_hidden": [32, 32],

        # "filter_size": 3,
        # "stride": 1,
        # "layer_norm": 0,



        # # MAU parameters
        # "in_shape": (14, 1, 32, 32), 
        # "results_csv": "./results/MAU_results.csv",
        # "exp_name": "MAU",
        # "tau": 5,                  # temporal buffer 길이 (최근 몇 step을 attention으로 볼지)
        # "cell_mode": "normal",     # MAUCell 내부 모드 (프로젝트 코드/오픈STL 설정과 맞춰야 함)
        # "model_mode": "normal",    # 'normal' or 'recall' (코드에서 states 체크함)

        # "sr_size": 1,              # 1이면 다운샘플 없음. 2/4로 하면 encoder/decoder가 down/up 수행
        # "patch_size": 1,           
        # "filter_size": 3,          
        # "stride": 1,               
        # "layer_norm": 0,     

        # "num_hidden": [32, 32],   
        # "num_layers": 2,



        # vit_gru
        # "results_csv": "./results/vit_gru_results.csv",
        # "exp_name": "vit_gru",
        # "image_size": 32,
        # "patch_size": 4,
        # "embed_dim": 128,
        # "depth": 3,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "vit_dropout": 0.0,
        # "vit_attn_dropout": 0.0,
        # "vit_freeze": False,



        # vit (used by vit_train.py)
        # "results_csv": "./results/vit_results.csv",
        # "exp_name": "vit",
        # "image_size": 32,
        # "patch_size": 4,
        # "embed_dim": 128,
        # "depth": 3,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "vit_dropout": 0.0,
        # "vit_attn_dropout": 0.0,
        # "vit_freeze": False,
        # "rollout_mode": "ar",

        # "in_shape": (14, 1, 32, 32),


        # Multi-ScaleB

        # "results_csv": "./results/multiscaleB_results.csv",
        # "exp_name": "multiscaleB_patch",

        # "image_size": 32,
        # "patch_size": 4,
        # "embed_dim": 128,

        # "temporal_scales": (1, 2, 7, 14),   # Tin=14 기준 스케일

        # "encoder_use_gru": True,            # True: GRU encoder / False: MLP encoder

        # "fusion": "gated",                  # "gated" or "add"
        # "in_shape": (14, 1, 32, 32),



        ## TimeSformer parameters
        # "results_csv": "./results/TimeSformer_results.csv",
        # "exp_name": "TimeSformer",
        # "image_size": 32,
        # "spatial_patch": 2,
        # "d_model": 128,
        # "depth": 3,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "dropout": 0.0,


        

        ## TimeSformer_2 parameters
        # "results_csv": "./results/TimeSformer2_results.csv",
        # "image_size": 32,
        # "in_shape": (14, 1, 32, 32),
        # "exp_name": "TimeSformer2",
        # "spatial_patch": 2,
        # "d_model": 128,
        # "depth": 3,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "dropout": 0.0,

        ## MSF - concat
        # "results_csv": "./results/msf_results.csv",
        # "exp_name": "msf",
        # "d_model": 128,
        # "depth": 3,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "dropout": 0.0,
        # "attn_dropout": 0.0,
        # "spatial_scales": (4, 8, 16, 32),
        # "base_scale": 4,
        # "use_pos_emb": False,
        # "fusion": "concat",     # "concat" or "gated"

        ## MSFG - gated
        # "results_csv": "./results/msfg_results.csv",
        # "exp_name": "msfg",
        # "fusion": "gated",


        # ## MSFv2 - concat
        # "results_csv": "./results/msfv2_base2_results.csv",
        # "exp_name": "msfv2_base2",
        # "d_model": 128,
        # "depth": 4,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "dropout": 0.0,
        # "attn_dropout": 0.0,
        # "spatial_scales": (4, 8, 16, 32),
        # "fusion": "sum", 
        # "image_size": 32,


        ## MSFv3 - concat
        # "results_csv": "./results/msfg3_gdl03_results.csv",
        # "exp_name": "msfv3_mean03",
        # "d_model": 128,
        # "depth": 4,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "dropout": 0.0,
        # "attn_dropout": 0.0,  
        # "spatial_scales": (2, 4, 8, 16),
        # "fusion": "sum", 
        # "image_size": 32,
        # "time_pool": "last",

        # "lambda_gdl": 0.3,


        #MSFv3
        # "results_csv": "./results/msfv3_ec_f.csv",
        # "exp_name": "msfv3_ec_f",
        # "d_model": 128,
        # "depth": 4,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "dropout": 0.0,
        # "attn_dropout": 0.0,
        # "spatial_scales": (2, 4, 8, 16),
        # "fusion": "sum", 
        # "image_size": 32,
        # "use_time_emb": True,
        # "use_scale_emb": True,
        # "use_pos_emb": False,

        # "freeze_time_emb": True,
        # "freeze_scale_emb": True,
        # "freeze_pos_emb": False,

        # "use_pad": True,
        # "time_pool": "last",
    

        # MSFv3 lk - concat
        # "results_csv": "./results/msfv3_pgdl03_results.csv",
        # "exp_name": "msfv3_pgdl03",
        # "d_model": 128,
        # "depth": 4,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "dropout": 0.0,
        # "attn_dropout": 0.0,
        # "spatial_scales": (2, 4, 8, 16),
        # "fusion": "sum", 
        # "image_size": 32,
        # "use_pos_emb": True,

        # "lambda_gdl": 0.3,
        # "alpha_tgdl": 0.0,
 
        # [0.0948, 0.2384, 0.5378, 1.0000, 1.4622, 1.7616, 1.9052]

        # msfv3_conv
        # "results_csv": "./results/msfv3_conv_base03_results.csv",
        # "exp_name": "msfv3_conv_base03",

        # "d_model": 128,
        # "depth": 4,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "dropout": 0.0,
        # "attn_dropout": 0.0,

        # "spatial_scales": (2, 4, 8, 16),
        # "fusion": "sum",            # "sum", "gated", "gated2"
        # "image_size": 32,
        # "use_pos_emb": False,
        # "time_pool": "last",

        # # per-scale conv refiner
        # "decoder_refine_mode": "residual",   # "residual" or "replace"
        # "decoder_refine_hidden": 8,          # 추천: 8 또는 16
        # "decoder_refine_depth": 5,



        ## MSFv3 - concat
        # "results_csv": "./results/msfv3_base2_results.csv",
        # "exp_name": "msfv3_base2",
        # "d_model": 128,
        # "depth": 4,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "dropout": 0.0,
        # "attn_dropout": 0.0,
        # "spatial_scales": (4, 8, 16),
        # "fusion": "sum", 
        # "image_size": 32,



        ## MSFv3 - concat
        # "results_csv": "./results/msfv3_base01_results.csv",
        # "exp_name": "msfv3_base01",
        # "d_model": 128,
        # "depth": 3,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "dropout": 0.0,
        # "attn_dropout": 0.0,
        # "spatial_scales": (2, 4, 8, 16),
        # "fusion": "sum", 
        # "image_size": 32,

        ## MSFv3 - concat
        # "results_csv": "./results/msfv3_base02_results.csv",
        # "exp_name": "msfv3_base02",
        # "d_model": 64,
        # "depth": 4,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "dropout": 0.0,
        # "attn_dropout": 0.0,
        # "spatial_scales": (2, 4, 8, 16),
        # "fusion": "sum", 
        # "image_size": 32,


        ## MSFv3 - concat
        # "results_csv": "./results/msfv3_depth02_results.csv",
        # "exp_name": "msfv3_depth02",
        # "d_model": 128,
        # "depth": 2,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "dropout": 0.0,
        # "attn_dropout": 0.0,
        # "spatial_scales": (2, 4, 8, 16),
        # "fusion": "sum", 
        # "image_size": 32,

        # MSFv3 - concat
        # "results_csv": "./results/msfv4_base_results.csv",
        # "exp_name": "msfv4_base",
        # "d_model": 128,
        # "depth": 4,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "dropout": 0.0,
        # "attn_dropout": 0.0,
        # "spatial_scales": (2, 4, 8, 16),
        # "fusion": "sum", 
        # "image_size": 32,


        # msfv3_small_conv
        # "results_csv": "./results/msfv3_smallconv_residual_base03_results.csv",
        # "exp_name": "msfv3_smallconv_residual_base03",
        # "d_model": 128,
        # "depth": 4,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "dropout": 0.0,
        # "attn_dropout": 0.0,
        # "spatial_scales": (2, 4, 8, 16),
        # "fusion": "sum",
        # "image_size": 32,
        # "use_pos_emb": False,
        # "decoder_refine_mode": "residual",  # replace, residual

        #resdecode
        # "results_csv": "./results/msfv3_resdecode_base03_results.csv",
        # "exp_name": "msfv3_resdecode_base03",
        # "d_model": 128,
        # "depth": 4,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "dropout": 0.0,
        # "attn_dropout": 0.0,
        # "spatial_scales": (2, 4, 8, 16),
        # "fusion": "sum",
        # "image_size": 32,
        # "use_pos_emb": False,
        # "time_pool": "last",

        # MSFv3 encoding
        # "results_csv": "./results/msfv3_encoding_results.csv",
        # "exp_name": "msfv3_encoding",

        # "d_model": 128,
        # "depth": 4,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "dropout": 0.0,
        # "attn_dropout": 0.0,

        # "spatial_scales": (2, 4, 8, 16),
        # "fusion": "sum",          # "sum", "gated", "gated2"
        # "image_size": 32,

        # "use_pad": True,
        # "use_pos_emb": False,
        # "time_pool": "last",

        # "factorize_time_emb": True,
        # "factorize_scale_emb": True,
        # "factorize_pos_emb": False,

        # "time_rank": 1,
        # "scale_rank": 1,
        # "pos_rank": 1,

        # "lambda_gdl": 0.3,
        # "alpha_tgdl": 0.0,

        # 실험 2) MSFv3 - Scale-head MoE (scale별 future_head를 MoE로)

        # "exp_name": "msfv3_moe8_base",
        # "results_csv": "./results/msfv3_moe8_base_results.csv",
        
        # "fusion": "sum",             # "sum"
        
        # "head_num_experts": 8,
        # "head_top_k": None,          # None=soft mixture, 1=top1, 2=top2
        # "head_temperature": 1.0,
        
        # "aux_balance_coef": 1e-2,
        # "aux_z_coef": 1e-3,
        
        # "lambda_gdl": 0.0,
        # "alpha_tgdl": 0.0,

        # "aux_coef": 0.0,
        # "d_model": 128,
        # "depth": 4,
        # "num_heads": 4,
        # "mlp_ratio": 4.0,
        # "dropout": 0.0,
        # "attn_dropout": 0.0,
        # "spatial_scales": (4, 8, 16),
        # "time_pool": "last",
        # "image_size": 32,

        # 해보기 :
        # - soft: head_top_k=None
        # - top2: head_top_k=2
        # - top1: head_top_k=1
        # - fusion까지 같이: fusion="gated"



    }

    
