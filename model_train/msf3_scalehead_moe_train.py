# msf3_moe_train.py
import os
import csv
import numpy as np
import json
import torch

from configs import get_default_cfg
from build_datasets import build_datasets
from trainer2 import train, set_seed

from model.models.msf3_model_scalehead_moe import MSFv3_ScaleHeadMoE, MSFGv3_ScaleHeadMoE


def _to_jsonable(obj):
    """Recursively convert numpy/torch types into JSON-serializable python types."""

    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()

    # numpy scalar/array
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()

    # dict
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]

    # python scalar
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj

    # fallback: string
    return str(obj)

def fmt_json(x):
    if x is None:
        return ""
    return json.dumps(_to_jsonable(x), ensure_ascii=False)
# =========================
# CSV columns (기존 + MoE 하이퍼 + aux 기록)
# =========================
CSV_COLUMNS = [
    "exp_name",
    "tag",
    "epoch",
    "is_best_val",

    "model",
    "tin",
    "tout",
    "ckpt_path",

    # ---- HP (기존 msf3처럼 저장) ----
    "d_model",
    "depth",
    "num_heads",
    "spatial_scales",
    "fusion",
    "time_pool",

    # ---- Optim ----
    "lr",
    "weight_decay",
    "batch_size",

    # ---- MoE head params ----
    "head_num_experts",
    "head_top_k",
    "head_temperature",
    "aux_balance_coef",
    "aux_z_coef",
    "aux_coef",

    # ---- Train metrics ----
    "train_loss",
    "train_mse", "train_rmse", "train_mae", "train_mape", "train_r2",
    "train_gradient_loss",
    "train_aux_loss",
    "train_rmse_t", "train_mae_t",

    # ---- Val metrics ----
    "val_loss",
    "val_mse", "val_rmse", "val_mae", "val_mape", "val_r2",
    "val_gradient_loss",
    "val_aux_loss",
    "val_rmse_t", "val_mae_t",

    # ---- Test metrics ----
    "test_loss",
    "test_mse", "test_rmse", "test_mae", "test_mape", "test_r2",
    "test_gradient_loss",
    "test_aux_loss",
    "test_rmse_t", "test_mae_t",

    "val_moe_mean_probs",
    "val_moe_top1_share",
    "val_moe_entropy",
    "val_moe_scale_mean_probs",

    "test_moe_mean_probs",
    "test_moe_top1_share",
    "test_moe_entropy",
    "test_moe_scale_mean_probs",
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


def main():
    cfg = get_default_cfg()
    set_seed(cfg["seed"])

    # =============
    # Defaults
    # =============
    cfg.setdefault("exp_name")
    cfg.setdefault("results_csv")

    # model hp (MSFv3 base)
    cfg.setdefault("image_size", 32)
    cfg.setdefault("d_model", 128)
    cfg.setdefault("depth", 4)
    cfg.setdefault("num_heads", 4)
    cfg.setdefault("mlp_ratio", 4.0)
    cfg.setdefault("dropout", 0.0)
    cfg.setdefault("attn_dropout", 0.0)
    cfg.setdefault("spatial_scales", (2, 4, 8, 16, 32))
    cfg.setdefault("fusion", "sum")
    cfg.setdefault("use_pad", True)
    cfg.setdefault("use_pos_emb", False)
    cfg.setdefault("time_pool", "last")

    # head MoE params
    cfg.setdefault("head_num_experts", 4)
    cfg.setdefault("head_top_k", None)          # None=soft, 1=top1, 2=top2 ...
    cfg.setdefault("head_temperature", 1.0)
    cfg.setdefault("aux_balance_coef", 1e-2)
    cfg.setdefault("aux_z_coef", 1e-3)

    # trainer에서 최종 loss에 더할 계수 (mse + gdl + aux_coef * aux_loss)
    cfg.setdefault("aux_coef", 1.0)

    exp_name = cfg["exp_name"]
    ckpt_path = os.path.join(cfg["ckpt_dir"], f"{exp_name}.pt")
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

    # =============
    # load mask & datasets
    # =============
    mask = np.load(cfg["mask_path"]).astype(bool)
    H, W = mask.shape
    cfg["image_size"] = int(H)  # dataset grid와 일치

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

    device = cfg["device"]

    ModelCls = MSFv3_ScaleHeadMoE if cfg["fusion"] == "sum" else MSFGv3_ScaleHeadMoE

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
        fusion=cfg["fusion"],
        use_pad=cfg["use_pad"],
        use_pos_emb=cfg["use_pos_emb"],
        time_pool=cfg["time_pool"],

        head_num_experts=cfg["head_num_experts"],
        head_top_k=cfg["head_top_k"],
        head_temperature=cfg["head_temperature"],
        aux_balance_coef=cfg["aux_balance_coef"],
        aux_z_coef=cfg["aux_z_coef"],
    ).to(device)

    test_metrics, best_epoch, best_val_rmse, history = train(
        model=model,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        mask_hw=torch_mask,
        cfg=cfg,
        ckpt_path=ckpt_path,
    )

    # =============
    # base row (고정 메타)
    # =============
    base = {
        "exp_name": exp_name,
        "model": ModelCls.__name__,
        "tin": cfg["input_len"],
        "tout": cfg["pred_len"],
        "ckpt_path": ckpt_path,

        "d_model": cfg["d_model"],
        "depth": cfg["depth"],
        "num_heads": cfg["num_heads"],
        "spatial_scales": str(tuple(cfg["spatial_scales"])),
        "fusion": cfg["fusion"],
        "time_pool": cfg["time_pool"],

        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "batch_size": cfg["batch_size"],

        "head_num_experts": cfg["head_num_experts"],
        "head_top_k": cfg["head_top_k"],
        "head_temperature": cfg["head_temperature"],
        "aux_balance_coef": cfg["aux_balance_coef"],
        "aux_z_coef": cfg["aux_z_coef"],
        "aux_coef": cfg["aux_coef"],
    }

    # =============
    # epoch rows
    # =============
    for h in history:

        ms = h.get("val_moe_stats", None)
        row = dict(base)
        
        row.update({
            "tag": "epoch",
            "epoch": h["epoch"],
            "is_best_val": h.get("is_best_val", 0),

            # train
            "train_loss": h.get("train_loss", ""),
            "train_mse": h.get("train_mse", ""),
            "train_rmse": h.get("train_rmse", ""),
            "train_mae": h.get("train_mae", ""),
            "train_mape": h.get("train_mape", ""),
            "train_r2": h.get("train_r2", ""),
            "train_gradient_loss": h.get("train_gradient_loss", ""),
            "train_aux_loss": h.get("train_aux_loss", ""),  

            "train_rmse_t": fmt_list(h.get("train_rmse_t")),
            "train_mae_t": fmt_list(h.get("train_mae_t")),

            # val
            "val_loss": h.get("val_loss", ""),
            "val_mse": h.get("val_mse", ""),
            "val_rmse": h.get("val_rmse", ""),
            "val_mae": h.get("val_mae", ""),
            "val_mape": h.get("val_mape", ""),
            "val_r2": h.get("val_r2", ""),
            "val_gradient_loss": h.get("val_gradient_loss", ""),
            "val_aux_loss": h.get("val_aux_loss", ""),      

            "val_rmse_t": fmt_list(h.get("val_rmse_t")),
            "val_mae_t": fmt_list(h.get("val_mae_t")),

            "val_moe_mean_probs": fmt_json(None if ms is None else ms.get("moe_mean_probs")),
            "val_moe_top1_share": fmt_json(None if ms is None else ms.get("moe_top1_share")),
            "val_moe_entropy": ("" if ms is None else ms.get("moe_entropy", "")),
            "val_moe_scale_mean_probs": fmt_json(None if ms is None else ms.get("moe_scale_mean_probs")),

        })
        append_result_csv(cfg["results_csv"], row)

    # best epoch row 찾기
    best_h = next((hh for hh in history if int(hh.get("epoch", -1)) == int(best_epoch)), None)

    best_row = dict(base)
    tms = test_metrics.get("moe_stats", None)

    best_row.update({
        "tag": "best_test",
        "epoch": int(best_epoch),
        "is_best_val": 1,

        # best val info
        "val_rmse": float(best_val_rmse),
        "val_loss": (best_h.get("val_loss") if best_h is not None else ""),
        "val_gradient_loss": (best_h.get("val_gradient_loss") if best_h is not None else ""),
        "val_aux_loss": (best_h.get("val_aux_loss") if best_h is not None else ""),

        # test metrics (trainer2가 반환)
        "test_loss": float(test_metrics.get("loss", 0.0)),
        "test_mse": float(test_metrics.get("mse", 0.0)),
        "test_rmse": float(test_metrics.get("rmse", 0.0)),
        "test_mae": float(test_metrics.get("mae", 0.0)),
        "test_mape": float(test_metrics.get("mape", 0.0)),
        "test_r2": float(test_metrics.get("r2", 0.0)),
        "test_gradient_loss": float(test_metrics.get("gradient_loss", 0.0)),
        "test_aux_loss": float(test_metrics.get("aux_loss", 0.0)),  
        "test_rmse_t": fmt_list(test_metrics.get("rmse_t")),
        "test_mae_t": fmt_list(test_metrics.get("mae_t")),

        "test_moe_mean_probs": fmt_json(None if tms is None else tms.get("moe_mean_probs")),
        "test_moe_top1_share": fmt_json(None if tms is None else tms.get("moe_top1_share")),
        "test_moe_entropy": ("" if tms is None else tms.get("moe_entropy", "")),
        "test_moe_scale_mean_probs": fmt_json(None if tms is None else tms.get("moe_scale_mean_probs")),

    })
    append_result_csv(cfg["results_csv"], best_row)

    print(f"Saved results -> {cfg['results_csv']}")
    print(f"Saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()