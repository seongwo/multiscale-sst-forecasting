import os
import csv
from datetime import datetime

import numpy as np
import torch

from configs import get_default_cfg
from build_datasets import build_datasets
from trainer import set_seed, train
from model.models.swinlstm_model import SwinLSTM_B_Model, SwinLSTM_D_Model

def fmt_list(x, nd=6):
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return "(" + ", ".join([f"{float(v):.{nd}f}" for v in x]) + ")"
    return str(x)

CSV_COLUMNS = [
    "exp_name",
    "tag",
    "epoch",
    "is_best_val",

    "model",
    "tin",
    "tout",
    "ckpt_path",

    "train_mse", "train_rmse", "train_mae", "train_mape", "train_r2",
    "train_rmse_t", "train_mae_t",
    "val_mse",   "val_rmse",   "val_mae",   "val_mape",   "val_r2",
    "val_rmse_t", "val_mae_t",
    "test_mse",  "test_rmse",  "test_mae",  "test_mape",  "test_r2",
    "test_rmse_t", "test_mae_t",
]

def append_result_csv(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def main():
    cfg = get_default_cfg()

    set_seed(cfg["seed"])
    
    exp_name = cfg.get("exp_name")
    ckpt_path = os.path.join(cfg["ckpt_dir"], f"{exp_name}.pt")

    mask = np.load(cfg["mask_path"]).astype(bool)

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

    model = SwinLSTM_B_Model(
        configs= cfg
    ).to(cfg["device"])

    # model = SwinLSTM_B_Model(
    #     depths_downsample=cfg["depths_downsample"],
    #     depths_upsample=cfg["depths_upsample"],
    #     num_heads=cfg["num_heads"],
    #     configs= cfg
    # ).to(cfg["device"])

    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

    test_metrics, best_epoch, best_val_rmse, history = train(
        model=model,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        mask_hw=torch_mask,
        cfg=cfg,
        ckpt_path=ckpt_path,
    )
    base = {
        "exp_name": exp_name,
        "model": "SwinLSTM",
        "tin": cfg["input_len"],
        "tout": cfg["pred_len"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "batch_size": cfg["batch_size"],
        "ckpt_path": ckpt_path,
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

if __name__ == "__main__":
    main()
