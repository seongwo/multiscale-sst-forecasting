from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Factorized ST Block (그대로)
# =========================
class FactorizedSTBlock(nn.Module):
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


# =========================
# MoE Utilities
# =========================
def _topk_masked_softmax(logits: torch.Tensor, top_k: Optional[int], temperature: float = 1.0) -> torch.Tensor:
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
    K = probs.shape[-1]
    p = probs.mean(dim=0)
    target = torch.full_like(p, 1.0 / K)
    return ((p - target) ** 2).mean() * float(K)


def _router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    z = torch.logsumexp(logits, dim=-1)
    return (z ** 2).mean()


class MoMLinear(nn.Module):
    """
    Sample-level MoE head for token-wise linear projection.

    x: (B, N, D)
    y: (B, N, out_dim)

    Router uses mean over N: (B, D) -> logits (B, K)
    Mixture is dense or top-k.
    """

    def __init__(
        self,
        d_model: int,
        out_dim: int,
        num_experts: int = 4,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
        aux_balance_coef: float = 1e-2,
        aux_z_coef: float = 1e-3,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.out_dim = int(out_dim)
        self.num_experts = int(num_experts)
        self.top_k = top_k
        self.temperature = float(temperature)
        self.aux_balance_coef = float(aux_balance_coef)
        self.aux_z_coef = float(aux_z_coef)

        self.router = nn.Linear(self.d_model, self.num_experts)
        self.experts = nn.ModuleList([nn.Linear(self.d_model, self.out_dim) for _ in range(self.num_experts)])

        self.last_probs: Optional[torch.Tensor] = None
        self.last_logits: Optional[torch.Tensor] = None
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        router_in = x.mean(dim=1)            # (B, D)
        logits = self.router(router_in)      # (B, K)
        probs = _topk_masked_softmax(logits, top_k=self.top_k, temperature=self.temperature)

        self.last_logits = logits.detach()
        self.last_probs = probs.detach()

        aux = torch.zeros((), device=x.device)
        if self.aux_balance_coef > 0:
            aux = aux + self.aux_balance_coef * _load_balance_loss(probs)
        if self.aux_z_coef > 0:
            aux = aux + self.aux_z_coef * _router_z_loss(logits)
        self.aux_loss = aux

        out = 0.0
        for k, expert in enumerate(self.experts):
            yk = expert(x)  # (B, N, out_dim)
            out = out + yk * probs[:, k].view(B, 1, 1)
        return out


# MSFormerV3 + ScaleHeadMoE
#  - 기존 MSFv3에서 future_heads만 MoMLinear로 교체
#  - smoother, tokenizer, transformer, decode, fusion 모두 그대로

class MSFormerV3_ScaleHeadMoE(nn.Module):
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
        fusion="sum",          # "sum" or "gated"
        use_pad=True,
        use_pos_emb=False,
        time_pool="last",      # "mean" or "last"


        head_num_experts: int = 4,
        head_top_k: Optional[int] = None,     # None=soft mixture, 1=top1, 2=top2...
        head_temperature: float = 1.0,
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
        self.fusion = str(fusion)
        assert self.fusion in ["sum", "gated"]
        self.use_pad = bool(use_pad)
        self.use_pos_emb = bool(use_pos_emb)
        self.time_pool = str(time_pool)
        assert self.time_pool in ["mean", "last"]


        self.smoother = nn.Sequential(
            nn.Conv2d(self.in_chans, self.in_chans, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.in_chans, self.in_chans, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.in_chans, self.in_chans, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.in_chans, self.in_chans, kernel_size=3, padding=1),
        )

        # tokenizers per scale (기존 그대로)
        self.tokenizers = nn.ModuleDict()
        for s in self.scales:
            self.tokenizers[str(s)] = nn.Conv2d(self.in_chans, self.d_model, kernel_size=s, stride=s, bias=True)

        # embeddings (기존 그대로)
        self.time_emb = nn.Embedding(self.tin, self.d_model)
        self.scale_emb = nn.Embedding(len(self.scales), self.d_model)

        self.pos_emb = nn.ModuleDict()
        if self.use_pos_emb:
            for s in self.scales:
                gs = self.image_size // s
                Ns = gs * gs
                self.pos_emb[str(s)] = nn.Embedding(Ns, self.d_model)

        # transformer blocks (기존 그대로)
        self.blocks = nn.ModuleList([
            FactorizedSTBlock(self.d_model, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(self.d_model)


        self.future_heads = nn.ModuleDict()
        self.token_to_patch = nn.ModuleDict()
        for s in self.scales:
            self.future_heads[str(s)] = MoMLinear(
                d_model=self.d_model,
                out_dim=self.tout * self.d_model,
                num_experts=head_num_experts,
                top_k=head_top_k,
                temperature=head_temperature,
                aux_balance_coef=aux_balance_coef,
                aux_z_coef=aux_z_coef,
            )
            self.token_to_patch[str(s)] = nn.Linear(self.d_model, self.in_chans * s * s)

        # fusion gate (기존 그대로)
        if self.fusion == "gated":
            self.gate = nn.Sequential(
                nn.Linear(self.d_model * len(self.scales), self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, len(self.scales)),
            )
        else:
            self.gate = None

        # expose aux_loss
        self.aux_loss: torch.Tensor = torch.tensor(0.0)
        self.last_scale_probs: Optional[torch.Tensor] = None  # gated일 때만 의미

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

    def _decode_scale(self, fut, s: int, gh: int, gw: int):
        B = fut.shape[0]
        patch_vec = self.token_to_patch[str(s)](fut)  # (B, Tout, N_s, C*s*s)
        patch_vec = patch_vec.reshape(B, self.tout, gh, gw, self.in_chans, s, s)
        patch_vec = patch_vec.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        y = patch_vec.reshape(B, self.tout, self.in_chans, gh * s, gw * s)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Tin, C, H, W = x.shape
        assert C == self.in_chans
        assert Tin == self.tin, f"Tin mismatch: got {Tin}, expected {self.tin}"

        # pad
        xt = x.reshape(B * Tin, C, H, W)
        if self.use_pad:
            xt, (pad_h, pad_w) = self._pad_to_multiple(xt, max(self.scales))
        else:
            pad_h, pad_w = 0, 0

        # tokenize + emb
        scale_tokens = []
        scale_meta = []  # (s, Ns, gh, gw)
        t_idx = torch.arange(Tin, device=x.device)

        for sid, s in enumerate(self.scales):
            tok_s, gh, gw = self._tokenize_one_scale(xt, s)  # (B*Tin, Ns, D)
            Ns = gh * gw
            tok_s = tok_s.reshape(B, Tin, Ns, self.d_model)

            tok_s = tok_s + self.time_emb(t_idx).view(1, Tin, 1, self.d_model)
            tok_s = tok_s + self.scale_emb(torch.tensor(sid, device=x.device)).view(1, 1, 1, self.d_model)

            if self.use_pos_emb:
                p_idx = torch.arange(Ns, device=x.device)
                tok_s = tok_s + self.pos_emb[str(s)](p_idx).view(1, 1, Ns, self.d_model)

            scale_tokens.append(tok_s)
            scale_meta.append((s, Ns, gh, gw))

        # transformer
        z = torch.cat(scale_tokens, dim=2)  # (B, Tin, N_total, D)
        h = z
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)

        # gated fusion weights (기존 그대로)
        if self.fusion == "gated":
            reps = [tok_s.mean(dim=(1, 2)) for tok_s in scale_tokens]  # (B,D) x S (pre-transformer reps)
            cat = torch.cat(reps, dim=-1)                               # (B, D*S)
            w = torch.softmax(self.gate(cat), dim=-1)                   # (B,S)
            self.last_scale_probs = w.detach()
        else:
            w = None
            self.last_scale_probs = None

        # decode scales + collect aux
        ys = []
        aux_total = torch.zeros((), device=x.device)

        cursor = 0
        for i, (s, Ns, gh, gw) in enumerate(scale_meta):
            tok = h[:, :, cursor:cursor + Ns, :]  # (B,Tin,Ns,D)
            cursor += Ns

            rep = tok[:, -1, :, :] if self.time_pool == "last" else tok.mean(dim=1)  # (B,Ns,D)

            fut_flat = self.future_heads[str(s)](rep)     # (B, Ns, Tout*D)  (MoE)
            aux_total = aux_total + self.future_heads[str(s)].aux_loss

            fut = fut_flat.reshape(B, Ns, self.tout, self.d_model).permute(0, 2, 1, 3).contiguous()
            y_s = self._decode_scale(fut, s, gh, gw)  # (B,Tout,C,Hp,Wp)

            if w is not None:
                y_s = y_s * w[:, i].view(B, 1, 1, 1, 1)

            ys.append(y_s)

        self.aux_loss = aux_total

        y = torch.stack(ys, dim=0).sum(dim=0)  # (B,Tout,C,Hp,Wp)

        # crop back
        if self.use_pad and (pad_h > 0 or pad_w > 0):
            y = y[..., :H, :W]

        # smoother (기존 V3 그대로)
        B2, T2, C2, H2, W2 = y.shape
        y2 = y.reshape(B2 * T2, C2, H2, W2)
        y2 = y2 + self.smoother(y2)
        y = y2.reshape(B2, T2, C2, H2, W2)
        return y



class MSFv3_ScaleHeadMoE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs.setdefault("fusion", "sum")
        self.net = MSFormerV3_ScaleHeadMoE(**kwargs)

    def forward(self, x):
        return self.net(x)

    @property
    def aux_loss(self):
        # trainer에서 model.aux_loss로 접근 가능
        return getattr(self.net, "aux_loss", torch.zeros((), device=next(self.parameters()).device))

    @property
    def last_scale_probs(self):
        return getattr(self.net, "last_scale_probs", None)


class MSFGv3_ScaleHeadMoE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs["fusion"] = "gated"
        self.net = MSFormerV3_ScaleHeadMoE(**kwargs)

    def forward(self, x):
        return self.net(x)

    @property
    def aux_loss(self):
        return getattr(self.net, "aux_loss", torch.zeros((), device=next(self.parameters()).device))

    @property
    def last_scale_probs(self):
        return getattr(self.net, "last_scale_probs", None)