import os
import csv
import numpy as np
import torch

from configs import get_default_cfg
from build_datasets import build_datasets
from trainer import train, set_seed

from model.models.msf3_encoding import MSFv3, MSFGv3, MSFG2v3


CSV_COLUMNS = [
    "exp_name",
    "tag",
    "epoch",
    "is_best_val",

    "model",
    "fusion",
    "tin",
    "tout",
    "ckpt_path",
    "image_size",

    "d_model",
    "depth",
    "num_heads",
    "mlp_ratio",
    "dropout",
    "attn_dropout",
    "spatial_scales",

    "use_pad",
    "use_pos_emb",
    "time_pool",

    "factorize_time_emb",
    "factorize_scale_emb",
    "factorize_pos_emb",
    "time_rank",
    "scale_rank",
    "pos_rank",

    "train_mse", "train_rmse", "train_mae", "train_mape", "train_r2",
    "train_rmse_t", "train_mae_t",
    "val_mse",   "val_rmse",   "val_mae",   "val_mape",   "val_r2",
    "val_rmse_t", "val_mae_t",
    "test_mse",  "test_rmse",  "test_mae",  "test_mape",  "test_r2",
    "test_rmse_t", "test_mae_t",
]


def fmt_list(x, nd=6):
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return "(" + ", ".join([f"{float(v):.{nd}f}" for v in x]) + ")"
    return str(x)


def append_result_csv(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def build_model(cfg: dict):
    fusion = str(cfg["fusion"]).lower()

    model_map = {
        "sum": MSFv3,
        "gated": MSFGv3,
        "gated2": MSFG2v3,
    }
    if fusion not in model_map:
        raise ValueError(f"Unsupported fusion: {fusion}. Choose from ['sum', 'gated', 'gated2'].")

    ModelCls = model_map[fusion]

    model = ModelCls(
        image_size=cfg["image_size"],
        in_chans=1,
        tin=cfg["input_len"],
        tout=cfg["pred_len"],
        d_model=cfg["d_model"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        dropout=cfg["dropout"],
        attn_dropout=cfg["attn_dropout"],
        spatial_scales=cfg["spatial_scales"],
        fusion=fusion,
        use_pad=cfg["use_pad"],
        use_pos_emb=cfg["use_pos_emb"],
        time_pool=cfg["time_pool"],

        # factorized embedding options
        factorize_time_emb=cfg["factorize_time_emb"],
        factorize_scale_emb=cfg["factorize_scale_emb"],
        factorize_pos_emb=cfg["factorize_pos_emb"],
        time_rank=cfg["time_rank"],
        scale_rank=cfg["scale_rank"],
        pos_rank=cfg["pos_rank"],
    ).to(cfg["device"])

    return ModelCls, model


def main():
    cfg = get_default_cfg()
    set_seed(cfg["seed"])

    # ------------------------------------------------------------------
    # defaults
    # ------------------------------------------------------------------
    cfg.setdefault("exp_name", "msf3_factorized_all3")
    cfg.setdefault("results_csv", "./results/msf3_factorized_all3_results.csv")

    cfg.setdefault("image_size", 32)
    cfg.setdefault("d_model", 128)
    cfg.setdefault("depth", 4)
    cfg.setdefault("num_heads", 4)
    cfg.setdefault("mlp_ratio", 4.0)
    cfg.setdefault("dropout", 0.0)
    cfg.setdefault("attn_dropout", 0.0)

    cfg.setdefault("spatial_scales", (2, 4, 8, 16, 32))
    cfg.setdefault("fusion", "sum")          # "sum", "gated", "gated2"
    cfg.setdefault("use_pad", True)
    cfg.setdefault("use_pos_emb", True)
    cfg.setdefault("time_pool", "last")      # "mean" or "last"

    # factorized embedding defaults
    cfg.setdefault("factorize_time_emb", True)
    cfg.setdefault("factorize_scale_emb", True)
    cfg.setdefault("factorize_pos_emb", True)

    cfg.setdefault("time_rank", 4)
    cfg.setdefault("scale_rank", 4)
    cfg.setdefault("pos_rank", 4)

    exp_name = cfg["exp_name"]
    ckpt_path = os.path.join(cfg["ckpt_dir"], f"{exp_name}.pt")
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # load mask & datasets
    # ------------------------------------------------------------------
    mask = np.load(cfg["mask_path"]).astype(bool)
    H, W = mask.shape
    cfg["image_size"] = int(H)

    train_set, val_set, test_set, mean, std, torch_mask = build_datasets(
        data_path=cfg["data_path"],
        mask=mask,
        tin=cfg["input_len"],
        tout=cfg["pred_len"],
        var_name=cfg["var_name"],
        train_start=cfg["train_start"],
        train_end=cfg["train_end"],
        val_start=cfg["val_start"],
        val_end=cfg["val_end"],
        test_start=cfg["test_start"],
        test_end=cfg["test_end"],
        stats_cache_json=cfg["stats_cache"],
    )

    ModelCls, model = build_model(cfg)

    test_metrics, best_epoch, best_val_rmse, history = train(
        model=model,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        mask_hw=torch_mask,
        cfg=cfg,
        ckpt_path=ckpt_path,
    )

    # ------------------------------------------------------------------
    # save results
    # ------------------------------------------------------------------
    base = {
        "exp_name": exp_name,
        "model": ModelCls.__name__,
        "fusion": cfg["fusion"],
        "tin": cfg["input_len"],
        "tout": cfg["pred_len"],
        "ckpt_path": ckpt_path,
        "image_size": cfg["image_size"],

        "d_model": cfg["d_model"],
        "depth": cfg["depth"],
        "num_heads": cfg["num_heads"],
        "mlp_ratio": cfg["mlp_ratio"],
        "dropout": cfg["dropout"],
        "attn_dropout": cfg["attn_dropout"],
        "spatial_scales": str(tuple(cfg["spatial_scales"])),

        "use_pad": int(bool(cfg["use_pad"])),
        "use_pos_emb": int(bool(cfg["use_pos_emb"])),
        "time_pool": cfg["time_pool"],

        "factorize_time_emb": int(bool(cfg["factorize_time_emb"])),
        "factorize_scale_emb": int(bool(cfg["factorize_scale_emb"])),
        "factorize_pos_emb": int(bool(cfg["factorize_pos_emb"])),

        "time_rank": int(cfg["time_rank"]),
        "scale_rank": int(cfg["scale_rank"]),
        "pos_rank": int(cfg["pos_rank"]),
    }

    for h in history:
        row = dict(base)
        row.update({
            "tag": "epoch",
            "epoch": h["epoch"],
            "is_best_val": h["is_best_val"],

            "train_mse": h["train_mse"],
            "train_rmse": h["train_rmse"],
            "train_mae": h["train_mae"],
            "train_mape": h["train_mape"],
            "train_r2": h["train_r2"],
            "train_rmse_t": fmt_list(h.get("train_rmse_t")),
            "train_mae_t": fmt_list(h.get("train_mae_t")),

            "val_mse": h["val_mse"],
            "val_rmse": h["val_rmse"],
            "val_mae": h["val_mae"],
            "val_mape": h["val_mape"],
            "val_r2": h["val_r2"],
            "val_rmse_t": fmt_list(h.get("val_rmse_t")),
            "val_mae_t": fmt_list(h.get("val_mae_t")),
        })
        append_result_csv(cfg["results_csv"], row)

    best_row = dict(base)
    best_row.update({
        "tag": "best_test",
        "epoch": int(best_epoch),
        "is_best_val": 1,

        "val_rmse": float(best_val_rmse),
        "test_mse": float(test_metrics["mse"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_mae": float(test_metrics["mae"]),
        "test_mape": float(test_metrics["mape"]),
        "test_r2": float(test_metrics["r2"]),
        "test_rmse_t": fmt_list(test_metrics.get("rmse_t")),
        "test_mae_t": fmt_list(test_metrics.get("mae_t")),
    })
    append_result_csv(cfg["results_csv"], best_row)

    print(f"Saved results -> {cfg['results_csv']}")
    print(
        f"[MODEL CONFIG] fusion={cfg['fusion']} | "
        f"time={cfg['factorize_time_emb']} (rank={cfg['time_rank']}) | "
        f"scale={cfg['factorize_scale_emb']} (rank={cfg['scale_rank']}) | "
        f"pos={cfg['factorize_pos_emb']} (rank={cfg['pos_rank']}) | "
        f"use_pos_emb={cfg['use_pos_emb']}"
    )


if __name__ == "__main__":
    main()