import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedSTBlock(nn.Module):
    """Factorized Space-Time Transformer block.

    Input x: (B, T, N, D)
      1) Spatial attention per time step: (B*T, N, D)
      2) Temporal attention per patch:    (B*N, T, D)
      3) MLP

    This is identical to the baseline block in msf2_model.py.
    """

    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.ln_s = nn.LayerNorm(d_model)
        self.attn_s = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout, batch_first=True)

        self.ln_t = nn.LayerNorm(d_model)
        self.attn_t = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout, batch_first=True)

        self.ln_m = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, T, N, D = x.shape
        xs = self.ln_s(x).reshape(B * T, N, D)
        ys, _ = self.attn_s(xs, xs, xs, need_weights=False)
        x = x + ys.reshape(B, T, N, D)

        xt = self.ln_t(x).permute(0, 2, 1, 3).reshape(B * N, T, D)
        yt, _ = self.attn_t(xt, xt, xt, need_weights=False)
        x = x + yt.reshape(B, N, T, D).permute(0, 2, 1, 3)

        x = x + self.mlp(self.ln_m(x))
        return x


def _topk_masked_softmax(logits: torch.Tensor, top_k: int, temperature: float = 1.0) -> torch.Tensor:
    """Softmax with optional top-k sparsification.

    logits: (B, S)
    returns probs: (B, S)
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    z = logits / temperature
    if top_k is None or top_k <= 0 or top_k >= z.shape[-1]:
        return torch.softmax(z, dim=-1)

    topk_vals, topk_idx = torch.topk(z, k=top_k, dim=-1)
    masked = torch.full_like(z, float("-inf"))
    masked.scatter_(dim=-1, index=topk_idx, src=topk_vals)
    return torch.softmax(masked, dim=-1)


def _load_balance_loss(probs: torch.Tensor) -> torch.Tensor:
    """Simple load-balance loss for soft routing.

    probs: (B, S) soft routing distribution.

    We encourage the mean routing probability to be close to uniform.
    """
    S = probs.shape[-1]
    p = probs.mean(dim=0)  # (S,)
    target = torch.full_like(p, 1.0 / S)
    return ((p - target) ** 2).mean() * float(S)


def _router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """Router z-loss (ST-MoE style): mean((logsumexp(logits))^2)."""
    z = torch.logsumexp(logits, dim=-1)
    return (z ** 2).mean()


class MSFormerV2_MoMFusion(nn.Module):
    """MSFv2 with upgraded MoM fusion.

    Baseline MSFv2 already supports fusion="gated" (soft weights over scales).
    This variant upgrades it into a more explicit MoM router:

    - Router input is computed from *post-transformer* scale slices (h), not from pre-transformer tokens.
    - Supports top-k scale routing (sparse) via top-k masked softmax.
    - Provides aux losses to prevent collapse:
        * load-balance loss (scale usage)
        * router z-loss (logit magnitude regularizer)
    - Exposes statistics for analysis:
        self.last_scale_probs: (B, S)
        self.last_scale_logits: (B, S)
        self.aux_loss: scalar tensor

    I/O is identical to baseline: input (B,Tin,C,H,W) -> output (B,Tout,C,H,W)
    """

    def __init__(
        self,
        image_size=32,
        in_chans=1,
        tin=14,
        tout=7,
        d_model=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
        attn_dropout=0.0,
        spatial_scales=(2, 4, 8, 16, 32),
        use_pad=True,
        use_pos_emb=False,
        time_pool="mean",  # "mean" or "last"
        # --- MoM fusion params
        mom_top_k: Optional[int] = None,  # None => dense soft routing
        mom_temperature: float = 1.0,
        aux_balance_coef: float = 1e-2,
        aux_z_coef: float = 1e-3,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.in_chans = int(in_chans)
        self.tin = int(tin)
        self.tout = int(tout)
        self.d_model = int(d_model)
        self.depth = int(depth)

        self.scales = tuple(int(s) for s in spatial_scales)
        assert len(self.scales) >= 1

        self.use_pad = bool(use_pad)
        self.use_pos_emb = bool(use_pos_emb)
        self.time_pool = str(time_pool)
        assert self.time_pool in ["mean", "last"]

        # MoM params
        self.mom_top_k = mom_top_k
        self.mom_temperature = float(mom_temperature)
        self.aux_balance_coef = float(aux_balance_coef)
        self.aux_z_coef = float(aux_z_coef)

        # tokenizers per scale
        self.tokenizers = nn.ModuleDict()
        for s in self.scales:
            self.tokenizers[str(s)] = nn.Conv2d(self.in_chans, self.d_model, kernel_size=s, stride=s, bias=True)

        # embeddings
        self.time_emb = nn.Embedding(self.tin, self.d_model)
        self.scale_emb = nn.Embedding(len(self.scales), self.d_model)

        self.pos_emb = nn.ModuleDict()
        if self.use_pos_emb:
            for s in self.scales:
                gs = self.image_size // s
                Ns = gs * gs
                self.pos_emb[str(s)] = nn.Embedding(Ns, self.d_model)

        # transformer blocks
        self.blocks = nn.ModuleList([
            FactorizedSTBlock(self.d_model, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(self.d_model)

        # scale heads + decoders
        self.future_heads = nn.ModuleDict()
        self.token_to_patch = nn.ModuleDict()
        for s in self.scales:
            self.future_heads[str(s)] = nn.Linear(self.d_model, self.tout * self.d_model)
            self.token_to_patch[str(s)] = nn.Linear(self.d_model, self.in_chans * s * s)

        # MoM router over scales (batch-wise)
        self.gate = nn.Sequential(
            nn.Linear(self.d_model * len(self.scales), self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, len(self.scales)),
        )

        # stats
        self.last_scale_probs: Optional[torch.Tensor] = None
        self.last_scale_logits: Optional[torch.Tensor] = None
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def _pad_to_multiple(self, x, multiple: int):
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (pad_h, pad_w)

    def _tokenize_one_scale(self, x_btchw, s: int):
        tok = self.tokenizers[str(s)](x_btchw)          # (B*T, D, gh, gw)
        gh, gw = tok.shape[-2], tok.shape[-1]
        tok = tok.flatten(2).transpose(1, 2)            # (B*T, N_s, D)
        return tok, gh, gw

    def _decode_scale(self, fut, s: int, Hp: int, Wp: int, gh: int, gw: int):
        """fut: (B, Tout, N_s, D) -> y_s: (B, Tout, C, Hp, Wp)"""
        B = fut.shape[0]
        patch_vec = self.token_to_patch[str(s)](fut)  # (B, Tout, N_s, C*s*s)
        patch_vec = patch_vec.reshape(B, self.tout, gh, gw, self.in_chans, s, s)
        patch_vec = patch_vec.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        y = patch_vec.reshape(B, self.tout, self.in_chans, gh * s, gw * s)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Tin, C, H, W)
        B, Tin, C, H, W = x.shape
        assert C == self.in_chans
        assert Tin == self.tin, f"Tin mismatch: got {Tin}, expected {self.tin}"

        # pad to max scale
        xt = x.reshape(B * Tin, C, H, W)
        if self.use_pad:
            xt, (pad_h, pad_w) = self._pad_to_multiple(xt, max(self.scales))
        else:
            pad_h, pad_w = 0, 0
        Hp, Wp = xt.shape[-2], xt.shape[-1]

        # tokenize per scale + emb add
        scale_meta = []  # (s, Ns, gh, gw)
        toks = []
        for sid, s in enumerate(self.scales):
            tok_s, gh, gw = self._tokenize_one_scale(xt, s)  # (B*Tin, Ns, D)
            Ns = gh * gw
            tok_s = tok_s.reshape(B, Tin, Ns, self.d_model)

            t_idx = torch.arange(Tin, device=x.device)
            tok_s = tok_s + self.time_emb(t_idx).view(1, Tin, 1, self.d_model)
            tok_s = tok_s + self.scale_emb(torch.tensor(sid, device=x.device)).view(1, 1, 1, self.d_model)

            if self.use_pos_emb:
                p_idx = torch.arange(Ns, device=x.device)
                tok_s = tok_s + self.pos_emb[str(s)](p_idx).view(1, 1, Ns, self.d_model)

            toks.append(tok_s)
            scale_meta.append((s, Ns, gh, gw))

        z = torch.cat(toks, dim=2)  # (B, Tin, N_total, D)
        h = z
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)

        # --- MoM fusion router input from post-transformer slices
        reps = []
        cursor = 0
        for (s, Ns, gh, gw) in scale_meta:
            tok = h[:, :, cursor:cursor + Ns, :]  # (B,Tin,Ns,D)
            cursor += Ns
            reps.append(tok.mean(dim=(1, 2)))  # (B,D)
        cat = torch.cat(reps, dim=-1)  # (B, D*S)
        logits = self.gate(cat)  # (B,S)
        probs = _topk_masked_softmax(logits, top_k=self.mom_top_k, temperature=self.mom_temperature)

        # stats
        self.last_scale_logits = logits.detach()
        self.last_scale_probs = probs.detach()

        # aux loss
        aux = torch.zeros((), device=x.device)
        if self.aux_balance_coef > 0:
            aux = aux + self.aux_balance_coef * _load_balance_loss(probs)
        if self.aux_z_coef > 0:
            aux = aux + self.aux_z_coef * _router_z_loss(logits)
        self.aux_loss = aux

        # scale-wise: slice -> pool -> head -> decode -> fuse
        ys = []
        cursor = 0
        for i, (s, Ns, gh, gw) in enumerate(scale_meta):
            tok = h[:, :, cursor:cursor + Ns, :]
            cursor += Ns

            if self.time_pool == "last":
                rep = tok[:, -1, :, :]
            else:
                rep = tok.mean(dim=1)

            fut = self.future_heads[str(s)](rep).reshape(B, Ns, self.tout, self.d_model)
            fut = fut.permute(0, 2, 1, 3).contiguous()  # (B,Tout,Ns,D)
            y_s = self._decode_scale(fut, s, Hp, Wp, gh, gw)

            # apply MoM scale weight
            y_s = y_s * probs[:, i].view(B, 1, 1, 1, 1)
            ys.append(y_s)

        y = torch.stack(ys, dim=0).sum(dim=0)  # (B,Tout,C,Hp,Wp)

        # crop back
        if self.use_pad and (pad_h > 0 or pad_w > 0):
            y = y[..., :H, :W]
        return y


# short names (keep similar API)
class MSFv2_MoMFusion(nn.Module):
    """MSFv2 variant: MoM fusion over scales."""

    def __init__(self, **kwargs):
        super().__init__()
        self.net = MSFormerV2_MoMFusion(**kwargs)

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# Backward-compatible aliases (optional)
# If you want to minimize changes in your training script, you can import
# MSFv2 / MSFGv2 from this file.
# NOTE: this variant *always* performs MoM fusion over scales.
# -----------------------------------------------------------------------------

class MSFv2(nn.Module):
    """Alias for MSFv2_MoMFusion (MoM fusion over scales)."""

    def __init__(self, **kwargs):
        super().__init__()
        self.net = MSFormerV2_MoMFusion(**kwargs)

    def forward(self, x):
        return self.net(x)


class MSFGv2(MSFv2):
    """Alias kept for compatibility (same as MSFv2 in this variant)."""
    pass

# ---- convenience properties for trainer/analysis

def _aux_loss_prop(self):
    # returns scalar tensor
    return getattr(self.net, 'aux_loss', torch.zeros((), device=next(self.parameters()).device))

def _scale_probs_prop(self):
    return getattr(self.net, 'last_scale_probs', None)

def _scale_logits_prop(self):
    return getattr(self.net, 'last_scale_logits', None)

for _cls in (MSFv2_MoMFusion, MSFv2, MSFGv2):
    _cls.aux_loss = property(_aux_loss_prop)
    _cls.last_scale_probs = property(_scale_probs_prop)
    _cls.last_scale_logits = property(_scale_logits_prop)
