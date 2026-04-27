import os
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from losses import masked_gdl
from losses import (
    masked_mse, masked_rmse, masked_mae, masked_mape, masked_r2,
    masked_rmse_per_t, masked_mae_per_t
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_5d_batched(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5:
        return x
    if x.dim() == 4:
        return x.unsqueeze(2)
    raise ValueError(f"Expected 4D/5D batched tensor, got {tuple(x.shape)}")


def ensure_pred_5d(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 5:
        out = pred
    elif pred.dim() == 4:
        out = pred.unsqueeze(2)
    else:
        raise ValueError(f"Expected pred 4D/5D, got {tuple(pred.shape)}")

    if out.shape[:2] != y.shape[:2] or out.shape[-2:] != y.shape[-2:]:
        raise ValueError(f"pred shape mismatch: pred={tuple(out.shape)} vs y={tuple(y.shape)}")
    if out.shape[2] != y.shape[2]:
        raise ValueError(f"channel mismatch: pred C={out.shape[2]} vs y C={y.shape[2]}")
    return out


@torch.no_grad()
def eval_epoch(model, loader, mask_hw, device, eps_mape: float, lambda_gdl: float):
    model.eval()
    sums = {
        "loss": 0.0,
        "mse": 0.0, "rmse": 0.0, "mae": 0.0, "mape": 0.0, "r2": 0.0,
        "gradient_loss": 0.0,
        "aux_loss": 0.0,
    }
    rmse_t_sum = None
    mae_t_sum = None
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x = ensure_5d_batched(x)
        y = ensure_5d_batched(y)

        pred, aux_loss = model(x, return_aux_loss=True)
        pred = ensure_pred_5d(pred, y)

        mse_loss = masked_mse(pred, y, mask_hw)
        total_loss = mse_loss + aux_loss

        gdl_scaled = torch.zeros((), device=device, dtype=mse_loss.dtype)
        if lambda_gdl != 0.0:
            gdl_scaled = lambda_gdl * masked_gdl(pred, y, mask_hw)
            total_loss = total_loss + gdl_scaled

        sums["loss"] += float(total_loss.item())
        sums["mse"] += float(mse_loss.item())
        sums["rmse"] += float(masked_rmse(pred, y, mask_hw).item())
        sums["mae"] += float(masked_mae(pred, y, mask_hw).item())
        sums["mape"] += float(masked_mape(pred, y, mask_hw, eps=eps_mape, as_percent=True).item())
        sums["r2"] += float(masked_r2(pred, y, mask_hw).item())
        sums["gradient_loss"] += float(gdl_scaled.item())
        sums["aux_loss"] += float(aux_loss.item())

        rmse_t = masked_rmse_per_t(pred, y, mask_hw).detach().cpu()
        mae_t = masked_mae_per_t(pred, y, mask_hw).detach().cpu()

        if rmse_t_sum is None:
            rmse_t_sum = rmse_t.clone()
            mae_t_sum = mae_t.clone()
        else:
            rmse_t_sum += rmse_t
            mae_t_sum += mae_t

        n += 1

    for k in sums:
        sums[k] /= max(n, 1)

    if rmse_t_sum is None:
        sums["rmse_t"] = None
        sums["mae_t"] = None
    else:
        sums["rmse_t"] = (rmse_t_sum / max(n, 1)).tolist()
        sums["mae_t"] = (mae_t_sum / max(n, 1)).tolist()

    return sums


def train(model, train_set, val_set, test_set, mask_hw: torch.Tensor, cfg: dict, ckpt_path: str):
    set_seed(int(cfg.get("seed", 42)))

    use_cuda = torch.cuda.is_available() and str(cfg.get("device", "cuda")).startswith("cuda")
    device = torch.device("cuda" if use_cuda else "cpu")

    bs = int(cfg.get("batch_size", 64))
    nw = int(cfg.get("num_workers", 4))
    lr = float(cfg.get("lr", 1e-4))
    wd = float(cfg.get("weight_decay", 0.0))
    max_epoch = int(cfg.get("max_epoch", 200))
    patience = int(cfg.get("patience", 20))
    eps_mape = float(cfg.get("eps_mape", 1e-3))
    lambda_gdl = float(cfg.get("lambda_gdl", 0.0))

    pin = True if use_cuda else False

    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True, num_workers=nw,
        drop_last=True, pin_memory=pin, persistent_workers=(nw > 0)
    )
    val_loader = DataLoader(
        val_set, batch_size=bs, shuffle=False, num_workers=nw,
        pin_memory=pin, persistent_workers=(nw > 0), drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=bs, shuffle=False, num_workers=nw,
        pin_memory=pin, persistent_workers=(nw > 0), drop_last=True
    )

    model = model.to(device)
    mask_hw = mask_hw.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

    best_val_rmse = float("inf")
    best_epoch = -1
    wait = 0
    history = []

    for epoch in range(max_epoch):
        model.train()
        sums = {
            "loss": 0.0,
            "mse": 0.0, "rmse": 0.0, "mae": 0.0, "mape": 0.0, "r2": 0.0,
            "gradient_loss": 0.0,
            "aux_loss": 0.0,
        }
        rmse_t_sum = None
        mae_t_sum = None
        n = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x = ensure_5d_batched(x)
            y = ensure_5d_batched(y)

            pred, aux_loss = model(x, return_aux_loss=True)
            pred = ensure_pred_5d(pred, y)

            mse_loss = masked_mse(pred, y, mask_hw)
            train_loss = mse_loss + aux_loss

            gradient_loss = torch.zeros((), device=device, dtype=mse_loss.dtype)
            if lambda_gdl != 0.0:
                gradient_loss = lambda_gdl * masked_gdl(pred, y, mask_hw)
                train_loss = train_loss + gradient_loss

            opt.zero_grad(set_to_none=True)
            train_loss.backward()
            opt.step()

            with torch.no_grad():
                sums["loss"] += float(train_loss.item())
                sums["mse"] += float(mse_loss.item())
                sums["rmse"] += math.sqrt(max(float(mse_loss.item()), 0.0))
                sums["mae"] += float(masked_mae(pred, y, mask_hw).item())
                sums["mape"] += float(masked_mape(pred, y, mask_hw, eps=eps_mape, as_percent=True).item())
                sums["r2"] += float(masked_r2(pred, y, mask_hw).item())
                sums["gradient_loss"] += float(gradient_loss.item())
                sums["aux_loss"] += float(aux_loss.item())

                rmse_t = masked_rmse_per_t(pred, y, mask_hw).detach().cpu()
                mae_t = masked_mae_per_t(pred, y, mask_hw).detach().cpu()

                if rmse_t_sum is None:
                    rmse_t_sum = rmse_t.clone()
                    mae_t_sum = mae_t.clone()
                else:
                    rmse_t_sum += rmse_t
                    mae_t_sum += mae_t

                n += 1

        for k in sums:
            sums[k] /= max(n, 1)

        train_rmse_t = (rmse_t_sum / max(n, 1)).tolist() if rmse_t_sum is not None else None
        train_mae_t = (mae_t_sum / max(n, 1)).tolist() if mae_t_sum is not None else None

        val_sums = eval_epoch(model, val_loader, mask_hw, device, eps_mape, lambda_gdl=lambda_gdl)
        is_best = float(val_sums["rmse"]) < best_val_rmse

        history.append({
            "epoch": epoch,
            "train_loss": sums["loss"],
            "train_mse": sums["mse"],
            "train_rmse": sums["rmse"],
            "train_mae": sums["mae"],
            "train_mape": sums["mape"],
            "train_r2": sums["r2"],
            "train_rmse_t": train_rmse_t,
            "train_mae_t": train_mae_t,
            "train_gradient_loss": sums["gradient_loss"],
            "train_aux_loss": sums["aux_loss"],

            "val_loss": val_sums["loss"],
            "val_mse": val_sums["mse"],
            "val_rmse": val_sums["rmse"],
            "val_mae": val_sums["mae"],
            "val_mape": val_sums["mape"],
            "val_r2": val_sums["r2"],
            "val_rmse_t": val_sums.get("rmse_t"),
            "val_mae_t": val_sums.get("mae_t"),
            "val_gradient_loss": val_sums["gradient_loss"],
            "val_aux_loss": val_sums["aux_loss"],

            "is_best_val": int(is_best),
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train Loss {sums['loss']:.6f} MSE {sums['mse']:.6f} RMSE {sums['rmse']:.6f} "
            f"MAE {sums['mae']:.6f} Aux {sums['aux_loss']:.6f} GDL {sums['gradient_loss']:.6f} "
            f"MAPE {sums['mape']:.3f} R2 {sums['r2']:.4f} || "
            f"val Loss {val_sums['loss']:.6f} MSE {val_sums['mse']:.6f} RMSE {val_sums['rmse']:.6f} "
            f"MAE {val_sums['mae']:.6f} Aux {val_sums['aux_loss']:.6f} GDL {val_sums['gradient_loss']:.6f} "
            f"MAPE {val_sums['mape']:.3f} R2 {val_sums['r2']:.4f}"
        )

        if is_best:
            best_val_rmse = float(val_sums["rmse"])
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "best_val_rmse": best_val_rmse,
                    "best_epoch": best_epoch,
                },
                ckpt_path,
            )
            print(f"Saved best: val RMSE={best_val_rmse:.6f} (epoch {best_epoch}) -> {ckpt_path}")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_sums = eval_epoch(model, test_loader, mask_hw, device, eps_mape, lambda_gdl=lambda_gdl)
    print(
        f"TEST | Loss {test_sums['loss']:.6f} MSE {test_sums['mse']:.6f} RMSE {test_sums['rmse']:.6f} "
        f"MAE {test_sums['mae']:.6f} Aux {test_sums['aux_loss']:.6f} GDL {test_sums['gradient_loss']:.6f} "
        f"MAPE {test_sums['mape']:.3f} R2 {test_sums['r2']:.4f}"
    )

    return test_sums, best_epoch, float(best_val_rmse), history