# trainer2.py
import os
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from losses import (
    masked_stgdl,
    masked_mse, masked_mae, masked_mape, masked_r2,
    masked_rmse_per_t, masked_mae_per_t
)

# ============================================================
# (NEW) MoE routing monitor utils
# ============================================================

def _entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)

def _collect_moe_probs(model) -> dict:
    """
    MSFv3_ScaleHeadMoE 계열에서 scale별 future_heads[*].last_probs 수집.
    반환:
      {
        "all": Tensor (M, K) or None,
        "by_scale": { "2": Tensor(B,K), ... }
      }
    """
    out = {"all": None, "by_scale": {}}
    all_probs = []

    # wrapper면 .net 안으로
    core = getattr(model, "net", model)

    fh = getattr(core, "future_heads", None)
    if fh is None:
        return out

    # ModuleDict / dict 모두 대응
    items = fh.items() if hasattr(fh, "items") else []
    for sk, head in items:
        p = getattr(head, "last_probs", None)
        if p is None or (not torch.is_tensor(p)):
            continue
        p = p.detach()
        out["by_scale"][str(sk)] = p
        all_probs.append(p)

    if len(all_probs) > 0:
        out["all"] = torch.cat(all_probs, dim=0)
    return out

def _summarize_probs(p: torch.Tensor) -> dict:
    """
    p: (M, K)
    return:
      mean_probs (K,), top1_share (K,), entropy (scalar)
    """
    mean_probs = p.mean(dim=0)
    top1 = torch.argmax(p, dim=-1)
    K = p.shape[-1]
    top1_share = torch.bincount(top1, minlength=K).float() / float(p.shape[0])
    ent = _entropy_from_probs(p).mean()
    return {"mean_probs": mean_probs, "top1_share": top1_share, "entropy": ent}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_5d_batched(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5:
        return x
    if x.dim() == 4:
        return x.unsqueeze(2)  # (B,T,1,H,W)
    raise ValueError(f"Expected 4D/5D batched tensor, got {tuple(x.shape)}")


def ensure_pred_5d(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 5:
        out = pred
    elif pred.dim() == 4:
        out = pred.unsqueeze(2)  # (B,T,1,H,W)
    else:
        raise ValueError(f"Expected pred 4D/5D, got {tuple(pred.shape)}")

    if out.shape[:2] != y.shape[:2] or out.shape[-2:] != y.shape[-2:]:
        raise ValueError(f"pred shape mismatch: pred={tuple(out.shape)} vs y={tuple(y.shape)}")
    if out.shape[2] != y.shape[2]:
        raise ValueError(f"channel mismatch: pred C={out.shape[2]} vs y C={y.shape[2]}")
    return out


def _get_aux_loss(model, device, dtype) -> torch.Tensor:
    """
    model.aux_loss가 있으면 tensor로 안전 변환해서 반환.
    없으면 0 tensor 반환.
    """
    aux = getattr(model, "aux_loss", None)
    if aux is None:
        return torch.zeros((), device=device, dtype=dtype)
    if not torch.is_tensor(aux):
        return torch.tensor(aux, device=device, dtype=dtype)
    return aux.to(device=device, dtype=dtype)


@torch.no_grad()
def eval_epoch(
    model,
    loader,
    mask_hw,
    device,
    eps_mape: float,
    lambda_gdl: float,
    alpha_tgdl: float,
    aux_coef: float,
):
    model.eval()
    sums = {
        "loss": 0.0,
        "mse": 0.0,
        "rmse": 0.0,
        "mae": 0.0,
        "mape": 0.0,
        "r2": 0.0,
        "gradient_loss": 0.0,
        "aux_loss": 0.0,            # ✅ 추가
        "moe_stats": None,          # ✅ NEW
    }
    moe_monitor = True
    rmse_t_sum = None
    mae_t_sum = None
    n = 0

    # ✅ NEW: moe 누적용
    moe_all_sum = None          # (K,)
    moe_top1_cnt = None         # (K,)
    moe_ent_sum = 0.0
    moe_total = 0

    moe_scale_sum = {}          # scale -> (K,)
    moe_scale_cnt = {}          # scale -> total rows

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x = ensure_5d_batched(x)
        y = ensure_5d_batched(y)

        pred = model(x)
        pred = ensure_pred_5d(pred, y)

        mse_loss = masked_mse(pred, y, mask_hw)

        gdl_scaled = torch.zeros((), device=device, dtype=mse_loss.dtype)
        if lambda_gdl != 0.0:
            gdl_scaled = lambda_gdl * masked_stgdl(pred, y, mask_hw, alpha_t=alpha_tgdl)

        # ✅ aux 기록 (loss에 더하기 전 원본 aux_loss 값)
        aux_raw = _get_aux_loss(model, device=device, dtype=mse_loss.dtype)

        total_loss = mse_loss + gdl_scaled
        if aux_coef != 0.0:
            total_loss = total_loss + aux_coef * aux_raw

        sums["loss"] += float(total_loss.item())
        sums["mse"] += float(mse_loss.item())
        sums["rmse"] += float(torch.sqrt(mse_loss.clamp_min(0.0)).item())
        sums["mae"] += float(masked_mae(pred, y, mask_hw).item())
        sums["mape"] += float(masked_mape(pred, y, mask_hw, eps=eps_mape, as_percent=True).item())
        sums["r2"] += float(masked_r2(pred, y, mask_hw).item())
        sums["gradient_loss"] += float(gdl_scaled.item())
        sums["aux_loss"] += float(aux_raw.item())

        # ✅ NEW: MoE routing monitor
        if moe_monitor:
            moe = _collect_moe_probs(model)
            if moe["all"] is not None:
                p = moe["all"].to(device=device)  # (M, K)
                K = p.shape[-1]

                if moe_all_sum is None:
                    moe_all_sum = torch.zeros((K,), device=device, dtype=torch.float32)
                    moe_top1_cnt = torch.zeros((K,), device=device, dtype=torch.float32)

                moe_all_sum += p.float().sum(dim=0)
                top1 = torch.argmax(p, dim=-1)
                moe_top1_cnt += torch.bincount(top1, minlength=K).float()
                moe_ent_sum += float(_entropy_from_probs(p.float()).sum().item())
                moe_total += int(p.shape[0])

                for sk, sp in moe["by_scale"].items():
                    sp = sp.to(device=device).float()  # (B, K)
                    if sk not in moe_scale_sum:
                        moe_scale_sum[sk] = torch.zeros((K,), device=device, dtype=torch.float32)
                        moe_scale_cnt[sk] = 0
                    moe_scale_sum[sk] += sp.sum(dim=0)
                    moe_scale_cnt[sk] += int(sp.shape[0])

        rmse_t = masked_rmse_per_t(pred, y, mask_hw).detach().cpu()
        mae_t = masked_mae_per_t(pred, y, mask_hw).detach().cpu()

        if rmse_t_sum is None:
            rmse_t_sum = rmse_t.clone()
            mae_t_sum = mae_t.clone()
        else:
            rmse_t_sum += rmse_t
            mae_t_sum += mae_t

        n += 1

    for k in ("loss", "mse", "rmse", "mae", "mape", "r2", "gradient_loss", "aux_loss"):
        sums[k] /= max(n, 1)

    sums["rmse_t"] = (rmse_t_sum / max(n, 1)).tolist() if rmse_t_sum is not None else None
    sums["mae_t"] = (mae_t_sum / max(n, 1)).tolist() if mae_t_sum is not None else None

    # ✅ NEW: epoch 전체 기준 moe stats 최종 계산
    if moe_monitor and moe_total > 0 and moe_all_sum is not None:
        mean_probs = (moe_all_sum / float(moe_total)).detach().cpu().numpy()
        top1_share = (moe_top1_cnt / float(moe_total)).detach().cpu().numpy()
        entropy = float(moe_ent_sum / float(moe_total))

        scale_mean_probs = {}
        for sk, ssum in moe_scale_sum.items():
            cnt = max(int(moe_scale_cnt.get(sk, 0)), 1)
            scale_mean_probs[sk] = (ssum / float(cnt)).detach().cpu().numpy()

        sums["moe_stats"] = {
            "moe_mean_probs": mean_probs,
            "moe_top1_share": top1_share,
            "moe_entropy": entropy,
            "moe_scale_mean_probs": scale_mean_probs,
        }

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
    alpha_tgdl = float(cfg.get("alpha_tgdl", 0.0))
    aux_coef = float(cfg.get("aux_coef", 0.0))

    moe_print = True
    moe_print_scales = True

    pin = True if use_cuda else False

    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True, num_workers=nw,
        drop_last=True, pin_memory=pin, persistent_workers=(nw > 0)
    )
    val_loader = DataLoader(
        val_set, batch_size=bs, shuffle=False, num_workers=nw,
        drop_last=True, pin_memory=pin, persistent_workers=(nw > 0)
    )
    test_loader = DataLoader(
        test_set, batch_size=bs, shuffle=False, num_workers=nw,
        drop_last=True, pin_memory=pin, persistent_workers=(nw > 0)
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
            "mse": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "mape": 0.0,
            "r2": 0.0,
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

            pred = model(x)
            pred = ensure_pred_5d(pred, y)

            mse_loss = masked_mse(pred, y, mask_hw)

            gdl_scaled = torch.zeros((), device=device, dtype=mse_loss.dtype)
            if lambda_gdl != 0.0:
                gdl_scaled = lambda_gdl * masked_stgdl(pred, y, mask_hw, alpha_t=alpha_tgdl)

            aux_raw = _get_aux_loss(model, device=device, dtype=mse_loss.dtype)

            train_loss = mse_loss + gdl_scaled
            if aux_coef != 0.0:
                train_loss = train_loss + aux_coef * aux_raw

            opt.zero_grad(set_to_none=True)
            train_loss.backward()
            opt.step()

            with torch.no_grad():
                sums["loss"] += float(train_loss.item())
                mse = float(mse_loss.item())
                sums["mse"] += mse
                sums["rmse"] += math.sqrt(max(mse, 0.0))
                sums["mae"] += float(masked_mae(pred, y, mask_hw).item())
                sums["mape"] += float(masked_mape(pred, y, mask_hw, eps=eps_mape, as_percent=True).item())
                sums["r2"] += float(masked_r2(pred, y, mask_hw).item())
                sums["gradient_loss"] += float(gdl_scaled.item())
                sums["aux_loss"] += float(aux_raw.item())

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

        val_sums = eval_epoch(
            model, val_loader, mask_hw, device, eps_mape,
            lambda_gdl=lambda_gdl, alpha_tgdl=alpha_tgdl, aux_coef=aux_coef
        )

        # (옵션) train split에서도 MoE 통계 보고 싶으면: epoch 끝나고 eval_epoch로 train_loader 한번 더
        train_moe_stats = None

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

            # ✅ NEW: 라우팅 통계도 history에 넣고 싶으면(원하면 csv에도 확장 가능)
            "val_moe_stats": val_sums.get("moe_stats"),
            "train_moe_stats": train_moe_stats,

            "is_best_val": int(is_best),
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train Loss {sums['loss']:.6f} MSE {sums['mse']:.6f} RMSE {sums['rmse']:.6f} "
            f"MAE {sums['mae']:.6f} GDL {sums['gradient_loss']:.6f} AUX {sums['aux_loss']:.6f} "
            f"MAPE {sums['mape']:.3f} R2 {sums['r2']:.4f} || "
            f"val Loss {val_sums['loss']:.6f} MSE {val_sums['mse']:.6f} RMSE {val_sums['rmse']:.6f} "
            f"MAE {val_sums['mae']:.6f} GDL {val_sums['gradient_loss']:.6f} AUX {val_sums['aux_loss']:.6f} "
            f"MAPE {val_sums['mape']:.3f} R2 {val_sums['r2']:.4f}"
        )

        # ✅ NEW: MoE stats print (val 기준)
        if moe_print and val_sums.get("moe_stats") is not None:
            ms = val_sums["moe_stats"]
            print("[MoE][VAL] moe_mean_probs:", ms["moe_mean_probs"])
            print("[MoE][VAL] moe_top1_share:", ms["moe_top1_share"])
            print("[MoE][VAL] moe_entropy:", ms["moe_entropy"])
            if moe_print_scales:
                print("[MoE][VAL] moe_scale_mean_probs:", ms["moe_scale_mean_probs"])

        if is_best:
            best_val_rmse = float(val_sums["rmse"])
            best_epoch = epoch
            wait = 0
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "best_val_rmse": best_val_rmse, "best_epoch": best_epoch},
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
        model, test_loader, mask_hw, device, eps_mape,
        lambda_gdl=lambda_gdl, alpha_tgdl=alpha_tgdl, aux_coef=aux_coef,
    )

    print(
        f"TEST | Loss {test_sums['loss']:.6f} MSE {test_sums['mse']:.6f} RMSE {test_sums['rmse']:.6f} "
        f"MAE {test_sums['mae']:.6f} GDL {test_sums['gradient_loss']:.6f} AUX {test_sums['aux_loss']:.6f} "
        f"MAPE {test_sums['mape']:.3f} R2 {test_sums['r2']:.4f}"
    )

    if moe_print and test_sums.get("moe_stats") is not None:
        ms = test_sums["moe_stats"]
        print("[MoE][TEST] moe_mean_probs:", ms["moe_mean_probs"])
        print("[MoE][TEST] moe_top1_share:", ms["moe_top1_share"])
        print("[MoE][TEST] moe_entropy:", ms["moe_entropy"])
        if moe_print_scales:
            print("[MoE][TEST] moe_scale_mean_probs:", ms["moe_scale_mean_probs"])

    return test_sums, best_epoch, float(best_val_rmse), history