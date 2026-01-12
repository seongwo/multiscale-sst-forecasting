import torch


def masked_mse(pred, target, mask, eps=1e-8):
    """
    pred, target: [B, T, 1, H, W]
    mask: [H, W]
    """
    diff = (pred - target) ** 2
    mask = mask[None, None, None]
    diff = diff * mask
    return diff.sum() / (mask.sum() * pred.shape[0] * pred.shape[1] + eps)

def masked_rmse(pred, target, mask, eps=1e-8):
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask
    mse = diff2.sum() / (mask.sum() + eps)
    return torch.sqrt(mse)
