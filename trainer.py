import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from losses import (
    masked_gdl,
    masked_mae,
    masked_mae_per_t,
    masked_logistic_temporal_mse,
    masked_mape,
    masked_mse,
    masked_r2,
    masked_rmse,
    masked_rmse_per_t,
)

try:
    from losses import masked_stgdl
except ImportError:  # pragma: no cover - compatibility with older losses.py
    masked_stgdl = None


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


def _entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def _collect_moe_probs(model) -> dict:
    out = {"all": None, "by_scale": {}}
    all_probs = []

    core = getattr(model, "net", model)
    future_heads = getattr(core, "future_heads", None)
    if future_heads is None or not hasattr(future_heads, "items"):
        return out

    for scale_key, head in future_heads.items():
        probs = getattr(head, "last_probs", None)
        if probs is None or not torch.is_tensor(probs):
            continue
        probs = probs.detach()
        out["by_scale"][str(scale_key)] = probs
        all_probs.append(probs)

    if all_probs:
        out["all"] = torch.cat(all_probs, dim=0)
    return out


def _as_aux_tensor(aux, device, dtype) -> torch.Tensor:
    if aux is None:
        return torch.zeros((), device=device, dtype=dtype)
    if not torch.is_tensor(aux):
        return torch.tensor(aux, device=device, dtype=dtype)
    return aux.to(device=device, dtype=dtype)


def _get_aux_loss(model, device, dtype) -> torch.Tensor:
    aux = getattr(model, "aux_loss", None)
    return _as_aux_tensor(aux, device=device, dtype=dtype)


def _forward_model(model, x, device, dtype, return_aux_loss: bool = False):
    if not return_aux_loss:
        return model(x), _get_aux_loss(model, device=device, dtype=dtype)

    out = model(x, return_aux_loss=True)
    if isinstance(out, tuple):
        if len(out) < 2:
            return out[0], torch.zeros((), device=device, dtype=dtype)
        return out[0], _as_aux_tensor(out[1], device=device, dtype=dtype)
    return out, _get_aux_loss(model, device=device, dtype=dtype)


def _gradient_loss(pred, target, mask_hw, lambda_gdl: float, alpha_tgdl: float):
    if lambda_gdl == 0.0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    if alpha_tgdl != 0.0 and masked_stgdl is not None:
        return lambda_gdl * masked_stgdl(pred, target, mask_hw, alpha_t=alpha_tgdl)
    return lambda_gdl * masked_gdl(pred, target, mask_hw)


def _base_losses(pred, target, mask_hw, use_tmse, logistic_k, logistic_midpoint):
    mse_metric = masked_mse(pred, target, mask_hw)
    tmse_metric = masked_logistic_temporal_mse(
        pred,
        target,
        mask_hw,
        eps=1e-8,
        k=logistic_k,
        midpoint=logistic_midpoint,
    )
    base_loss = tmse_metric if use_tmse else mse_metric
    return base_loss, mse_metric, tmse_metric


def _empty_metric_sums():
    return {
        "loss": 0.0,
        "mse": 0.0,
        "tmse": 0.0,
        "rmse": 0.0,
        "mae": 0.0,
        "mape": 0.0,
        "r2": 0.0,
        "gradient_loss": 0.0,
        "aux_loss": 0.0,
    }


def _update_moe_accumulators(model, device, acc):
    moe = _collect_moe_probs(model)
    if moe["all"] is None:
        return

    probs = moe["all"].to(device=device).float()
    num_experts = probs.shape[-1]

    if acc["all_sum"] is None:
        acc["all_sum"] = torch.zeros((num_experts,), device=device, dtype=torch.float32)
        acc["top1_count"] = torch.zeros((num_experts,), device=device, dtype=torch.float32)

    acc["all_sum"] += probs.sum(dim=0)
    top1 = torch.argmax(probs, dim=-1)
    acc["top1_count"] += torch.bincount(top1, minlength=num_experts).float()
    acc["entropy_sum"] += float(_entropy_from_probs(probs).sum().item())
    acc["total"] += int(probs.shape[0])

    for scale_key, scale_probs in moe["by_scale"].items():
        scale_probs = scale_probs.to(device=device).float()
        if scale_key not in acc["scale_sum"]:
            acc["scale_sum"][scale_key] = torch.zeros((num_experts,), device=device, dtype=torch.float32)
            acc["scale_count"][scale_key] = 0
        acc["scale_sum"][scale_key] += scale_probs.sum(dim=0)
        acc["scale_count"][scale_key] += int(scale_probs.shape[0])


def _finalize_moe_stats(acc):
    if acc["total"] <= 0 or acc["all_sum"] is None:
        return None

    total = float(acc["total"])
    scale_mean_probs = {}
    for scale_key, scale_sum in acc["scale_sum"].items():
        count = max(int(acc["scale_count"].get(scale_key, 0)), 1)
        scale_mean_probs[scale_key] = (scale_sum / float(count)).detach().cpu().numpy()

    return {
        "moe_mean_probs": (acc["all_sum"] / total).detach().cpu().numpy(),
        "moe_top1_share": (acc["top1_count"] / total).detach().cpu().numpy(),
        "moe_entropy": float(acc["entropy_sum"] / total),
        "moe_scale_mean_probs": scale_mean_probs,
    }


@torch.no_grad()
def eval_epoch(
    model,
    loader,
    mask_hw,
    device,
    eps_mape: float,
    lambda_gdl: float = 0.0,
    alpha_tgdl: float = 0.0,
    aux_coef: float = 0.0,
    moe_monitor: bool = False,
    model_return_aux_loss: bool = False,
    use_tmse: bool = False,
    logistic_k: float = 1.0,
    logistic_midpoint=None,
):
    model.eval()
    sums = _empty_metric_sums()
    rmse_t_sum = None
    mae_t_sum = None
    n = 0

    moe_acc = {
        "all_sum": None,
        "top1_count": None,
        "entropy_sum": 0.0,
        "total": 0,
        "scale_sum": {},
        "scale_count": {},
    }

    for x, y in loader:
        x = ensure_5d_batched(x.to(device, non_blocking=True))
        y = ensure_5d_batched(y.to(device, non_blocking=True))

        pred, aux_raw = _forward_model(
            model,
            x,
            device=device,
            dtype=x.dtype,
            return_aux_loss=model_return_aux_loss,
        )
        pred = ensure_pred_5d(pred, y)

        base_loss, mse_metric, tmse_metric = _base_losses(
            pred, y, mask_hw, use_tmse, logistic_k, logistic_midpoint
        )
        gradient_loss = _gradient_loss(pred, y, mask_hw, lambda_gdl, alpha_tgdl)
        total_loss = base_loss + gradient_loss + aux_coef * aux_raw

        sums["loss"] += float(total_loss.item())
        sums["mse"] += float(mse_metric.item())
        sums["tmse"] += float(tmse_metric.item())
        sums["rmse"] += float(masked_rmse(pred, y, mask_hw).item())
        sums["mae"] += float(masked_mae(pred, y, mask_hw).item())
        sums["mape"] += float(masked_mape(pred, y, mask_hw, eps=eps_mape, as_percent=True).item())
        sums["r2"] += float(masked_r2(pred, y, mask_hw).item())
        sums["gradient_loss"] += float(gradient_loss.item())
        sums["aux_loss"] += float(aux_raw.item())

        rmse_t = masked_rmse_per_t(pred, y, mask_hw).detach().cpu()
        mae_t = masked_mae_per_t(pred, y, mask_hw).detach().cpu()
        rmse_t_sum = rmse_t.clone() if rmse_t_sum is None else rmse_t_sum + rmse_t
        mae_t_sum = mae_t.clone() if mae_t_sum is None else mae_t_sum + mae_t

        if moe_monitor:
            _update_moe_accumulators(model, device, moe_acc)

        n += 1

    for key in sums:
        sums[key] /= max(n, 1)

    sums["rmse_t"] = (rmse_t_sum / max(n, 1)).tolist() if rmse_t_sum is not None else None
    sums["mae_t"] = (mae_t_sum / max(n, 1)).tolist() if mae_t_sum is not None else None
    sums["moe_stats"] = _finalize_moe_stats(moe_acc) if moe_monitor else None
    return sums


def train(model, train_set, val_set, test_set, mask_hw: torch.Tensor, cfg: dict, ckpt_path: str):
    set_seed(int(cfg.get("seed", 42)))

    use_cuda = torch.cuda.is_available() and str(cfg.get("device", "cuda")).startswith("cuda")
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 4))
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    max_epoch = int(cfg.get("max_epoch", 200))
    patience = int(cfg.get("patience", 20))
    eps_mape = float(cfg.get("eps_mape", 1e-3))

    lambda_gdl = float(cfg.get("lambda_gdl", 0.0))
    alpha_tgdl = float(cfg.get("alpha_tgdl", 0.0))
    model_return_aux_loss = bool(cfg.get("model_return_aux_loss", False))
    aux_coef = float(cfg.get("aux_coef", 0.0))
    moe_monitor = bool(cfg.get("moe_monitor", False))
    moe_print = bool(cfg.get("moe_print", moe_monitor))
    moe_print_scales = bool(cfg.get("moe_print_scales", True))

    use_tmse = bool(cfg.get("tmse", False))
    logistic_k = float(cfg.get("logistic_k", 1.0))
    logistic_midpoint = cfg.get("logistic_midpoint", None)

    pin_memory = bool(use_cuda)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    model = model.to(device)
    mask_hw = mask_hw.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

    best_val_rmse = float("inf")
    best_epoch = -1
    wait = 0
    history = []

    for epoch in range(max_epoch):
        model.train()
        sums = _empty_metric_sums()
        rmse_t_sum = None
        mae_t_sum = None
        n = 0

        for x, y in train_loader:
            x = ensure_5d_batched(x.to(device, non_blocking=True))
            y = ensure_5d_batched(y.to(device, non_blocking=True))

            pred, aux_raw = _forward_model(
                model,
                x,
                device=device,
                dtype=x.dtype,
                return_aux_loss=model_return_aux_loss,
            )
            pred = ensure_pred_5d(pred, y)

            base_loss, mse_metric, tmse_metric = _base_losses(
                pred, y, mask_hw, use_tmse, logistic_k, logistic_midpoint
            )
            gradient_loss = _gradient_loss(pred, y, mask_hw, lambda_gdl, alpha_tgdl)
            train_loss = base_loss + gradient_loss + aux_coef * aux_raw

            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                sums["loss"] += float(train_loss.item())
                sums["mse"] += float(mse_metric.item())
                sums["tmse"] += float(tmse_metric.item())
                sums["rmse"] += float(masked_rmse(pred, y, mask_hw).item())
                sums["mae"] += float(masked_mae(pred, y, mask_hw).item())
                sums["mape"] += float(masked_mape(pred, y, mask_hw, eps=eps_mape, as_percent=True).item())
                sums["r2"] += float(masked_r2(pred, y, mask_hw).item())
                sums["gradient_loss"] += float(gradient_loss.item())
                sums["aux_loss"] += float(aux_raw.item())

                rmse_t = masked_rmse_per_t(pred, y, mask_hw).detach().cpu()
                mae_t = masked_mae_per_t(pred, y, mask_hw).detach().cpu()
                rmse_t_sum = rmse_t.clone() if rmse_t_sum is None else rmse_t_sum + rmse_t
                mae_t_sum = mae_t.clone() if mae_t_sum is None else mae_t_sum + mae_t
                n += 1

        for key in sums:
            sums[key] /= max(n, 1)

        train_rmse_t = (rmse_t_sum / max(n, 1)).tolist() if rmse_t_sum is not None else None
        train_mae_t = (mae_t_sum / max(n, 1)).tolist() if mae_t_sum is not None else None

        val_sums = eval_epoch(
            model,
            val_loader,
            mask_hw,
            device,
            eps_mape,
            lambda_gdl=lambda_gdl,
            alpha_tgdl=alpha_tgdl,
            aux_coef=aux_coef,
            moe_monitor=moe_monitor,
            model_return_aux_loss=model_return_aux_loss,
            use_tmse=use_tmse,
            logistic_k=logistic_k,
            logistic_midpoint=logistic_midpoint,
        )

        is_best = float(val_sums["rmse"]) < best_val_rmse
        history.append({
            "epoch": epoch,
            "train_loss": sums["loss"],
            "train_mse": sums["mse"],
            "train_tmse": sums["tmse"],
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
            "val_tmse": val_sums["tmse"],
            "val_rmse": val_sums["rmse"],
            "val_mae": val_sums["mae"],
            "val_mape": val_sums["mape"],
            "val_r2": val_sums["r2"],
            "val_rmse_t": val_sums.get("rmse_t"),
            "val_mae_t": val_sums.get("mae_t"),
            "val_gradient_loss": val_sums["gradient_loss"],
            "val_aux_loss": val_sums["aux_loss"],
            "val_moe_stats": val_sums.get("moe_stats"),
            "is_best_val": int(is_best),
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train Loss {sums['loss']:.6f} MSE {sums['mse']:.6f} TMSE {sums['tmse']:.6f} "
            f"RMSE {sums['rmse']:.6f} MAE {sums['mae']:.6f} "
            f"GDL {sums['gradient_loss']:.6f} AUX {sums['aux_loss']:.6f} "
            f"MAPE {sums['mape']:.3f} R2 {sums['r2']:.4f} || "
            f"val Loss {val_sums['loss']:.6f} MSE {val_sums['mse']:.6f} TMSE {val_sums['tmse']:.6f} "
            f"RMSE {val_sums['rmse']:.6f} MAE {val_sums['mae']:.6f} "
            f"GDL {val_sums['gradient_loss']:.6f} AUX {val_sums['aux_loss']:.6f} "
            f"MAPE {val_sums['mape']:.3f} R2 {val_sums['r2']:.4f}"
        )

        if moe_print and val_sums.get("moe_stats") is not None:
            moe_stats = val_sums["moe_stats"]
            print("[MoE][VAL] moe_mean_probs:", moe_stats["moe_mean_probs"])
            print("[MoE][VAL] moe_top1_share:", moe_stats["moe_top1_share"])
            print("[MoE][VAL] moe_entropy:", moe_stats["moe_entropy"])
            if moe_print_scales:
                print("[MoE][VAL] moe_scale_mean_probs:", moe_stats["moe_scale_mean_probs"])

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

    test_sums = eval_epoch(
        model,
        test_loader,
        mask_hw,
        device,
        eps_mape,
        lambda_gdl=lambda_gdl,
        alpha_tgdl=alpha_tgdl,
        aux_coef=aux_coef,
        moe_monitor=moe_monitor,
        model_return_aux_loss=model_return_aux_loss,
        use_tmse=use_tmse,
        logistic_k=logistic_k,
        logistic_midpoint=logistic_midpoint,
    )

    print(
        f"TEST | Loss {test_sums['loss']:.6f} MSE {test_sums['mse']:.6f} TMSE {test_sums['tmse']:.6f} "
        f"RMSE {test_sums['rmse']:.6f} MAE {test_sums['mae']:.6f} "
        f"GDL {test_sums['gradient_loss']:.6f} AUX {test_sums['aux_loss']:.6f} "
        f"MAPE {test_sums['mape']:.3f} R2 {test_sums['r2']:.4f}"
    )

    if moe_print and test_sums.get("moe_stats") is not None:
        moe_stats = test_sums["moe_stats"]
        print("[MoE][TEST] moe_mean_probs:", moe_stats["moe_mean_probs"])
        print("[MoE][TEST] moe_top1_share:", moe_stats["moe_top1_share"])
        print("[MoE][TEST] moe_entropy:", moe_stats["moe_entropy"])
        if moe_print_scales:
            print("[MoE][TEST] moe_scale_mean_probs:", moe_stats["moe_scale_mean_probs"])

    return test_sums, best_epoch, float(best_val_rmse), history
