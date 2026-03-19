import torch


def _broadcast_mask(mask, x):
    # mask의 차원을 x와 동일하게 맞춥니다.
    if mask.dim() < x.dim():
        # 부족한 앞 차원들을 1로 채워줍니다.
        # 예: x가 (C, H, W)이고 mask가 (H, W)면 (1, H, W)로 변환
        diff = x.dim() - mask.dim()
        for _ in range(diff):
            mask = mask.unsqueeze(0)
            
    mask = mask.to(dtype=x.dtype, device=x.device)
    
    # 이제 x의 형상에 맞춰 확장할 수 있습니다.
    return mask.expand_as(x)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = _broadcast_mask(mask, pred)
    diff2 = (pred - target) ** 2
    num = (diff2 * m).sum()
    den = m.sum().clamp_min(eps)
    return num / den

def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(masked_mse(pred, target, mask, eps=eps))

def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = _broadcast_mask(mask, pred)
    diff = (pred - target).abs()
    num = (diff * m).sum()
    den = m.sum().clamp_min(eps)
    return num / den

def masked_mape(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-3,
    as_percent: bool = True,
) -> torch.Tensor:
    m = _broadcast_mask(mask, pred)
    denom = target.abs().clamp_min(eps)
    ape = ((pred - target).abs() / denom) * m
    num = ape.sum()
    den = m.sum().clamp_min(1e-8)
    out = num / den
    return out * 100.0 if as_percent else out

def masked_r2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = _broadcast_mask(mask, pred)
    wsum = m.sum().clamp_min(eps)

    y_mean = (target * m).sum() / wsum
    sse = (((pred - target) ** 2) * m).sum()
    sst = (((target - y_mean) ** 2) * m).sum().clamp_min(eps)
    return 1.0 - (sse / sst)


def masked_mse_per_t(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

    m = _broadcast_mask(mask, pred)  # (B,T,C,H,W)
    diff2 = (pred - target) ** 2

    # (B,T,C,H,W) -> sum over (B,C,H,W) => (T,)
    num_t = (diff2 * m).sum(dim=(0, 2, 3, 4))
    den_t = m.sum(dim=(0, 2, 3, 4)).clamp_min(eps)
    return num_t / den_t

def masked_rmse_per_t(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

    return torch.sqrt(masked_mse_per_t(pred, target, mask, eps=eps).clamp_min(0.0))

def masked_mae_per_t(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

    m = _broadcast_mask(mask, pred)
    diff = (pred - target).abs()
    num_t = (diff * m).sum(dim=(0, 2, 3, 4))
    den_t = m.sum(dim=(0, 2, 3, 4)).clamp_min(eps)
    return num_t / den_t



def masked_logistic_temporal_mse(pred, target, mask, eps=1e-8, k=1.0, midpoint=None):
    m = _broadcast_mask(mask, pred) 
    diff2 = (pred - target) ** 2

    # timestep별 합
    num_t = (diff2 * m).sum(dim=(0, 2, 3, 4))   
    den_t = m.sum(dim=(0, 2, 3, 4)).clamp_min(eps)
    mse_t = num_t / den_t                        

    T = pred.shape[1]
    t = torch.arange(T, device=pred.device, dtype=pred.dtype)

    if midpoint is None:
        midpoint = (T - 1) / 2.0

    w = torch.sigmoid(k * (t - midpoint))
    w = w / w.mean().clamp_min(eps)            

    return (w * mse_t).mean()



def masked_gdl(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:

    """
    Gradient Difference Loss (GDL)
    pred/target: (B,T,C,H,W)
    mask: (H,W) or broadcastable

    dx, dy에서 "양쪽 픽셀이 모두 유효(mask=1)"인 곳만 반영해서
    해안선(마스크 경계) 때문에 gradient가 터지는 걸 줄임.
    """
    m = _broadcast_mask(mask, pred)  # (B,T,C,H,W)

    # dx (W 방향)
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    gdx = (dx_p.abs() - dx_t.abs()).abs()
    mx = m[..., :, 1:] * m[..., :, :-1]  # (B,T,C,H,W-1)

    # dy (H 방향)
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    gdy = (dy_p.abs() - dy_t.abs()).abs()
    my = m[..., 1:, :] * m[..., :-1, :]  # (B,T,C,H-1,W)

    loss_dx = (gdx * mx).sum() / mx.sum().clamp_min(eps)
    loss_dy = (gdy * my).sum() / my.sum().clamp_min(eps)
    return loss_dx + loss_dy

def masked_tgdl(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Temporal Gradient Difference Loss (tGDL)
    pred/target: (B,T,C,H,W)
    mask: (H,W) or broadcastable

    시간 방향 dt에서 | |dt_pred| - |dt_target| | 를 최소화.
    마스크는 "연속 두 프레임 모두 유효"한 위치만 반영.
    """
    m = _broadcast_mask(mask, pred)  # (B,T,C,H,W)

    # dt (time 방향)
    dt_p = pred[:, 1:] - pred[:, :-1]      # (B,T-1,C,H,W)
    dt_t = target[:, 1:] - target[:, :-1]  # (B,T-1,C,H,W)
    gdt = (dt_p.abs() - dt_t.abs()).abs()  # | |dt_p| - |dt_t| |

    mt = m[:, 1:] * m[:, :-1]              # (B,T-1,C,H,W)

    loss_dt = (gdt * mt).sum() / mt.sum().clamp_min(eps)
    return loss_dt

def masked_stgdl(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    alpha_t: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Spatio-Temporal GDL = spatial GDL + alpha_t * temporal GDL
    (스케일링(lambda)은 trainer에서 하는 걸 추천)
    """
    return masked_gdl(pred, target, mask, eps=eps) + alpha_t * masked_tgdl(pred, target, mask, eps=eps)