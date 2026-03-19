import torch
import torch.nn as nn
import torch.nn.functional as F


def _topk_masked_softmax(logits: torch.Tensor, top_k: int | None, temperature: float = 1.0) -> torch.Tensor:
    """Softmax with optional top-k masking.

    logits: (B, K)
    top_k: None -> dense softmax
           1/2/... -> keep only top-k entries per row
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    x = logits / float(temperature)
    if top_k is None:
        return torch.softmax(x, dim=-1)
    k = int(top_k)
    if k <= 0:
        raise ValueError("top_k must be >= 1 or None")
    if k >= x.shape[-1]:
        return torch.softmax(x, dim=-1)

    # mask all but top-k
    vals, idx = torch.topk(x, k=k, dim=-1)
    mask = torch.full_like(x, float("-inf"))
    mask.scatter_(dim=-1, index=idx, src=vals)
    return torch.softmax(mask, dim=-1)


def _load_balance_loss(probs: torch.Tensor) -> torch.Tensor:
    """Simple load-balance loss to discourage collapse.

    probs: (B, K)
    We want mean usage across batch to be close to uniform.
    """
    if probs.numel() == 0:
        return torch.zeros((), device=probs.device)
    mean = probs.mean(dim=0)  # (K,)
    k = mean.numel()
    # MSE to uniform
    target = torch.full_like(mean, 1.0 / k)
    return torch.mean((mean - target) ** 2)


def _router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """Router z-loss (stabilizes logits magnitude)."""
    # logsumexp is always >= max(logits); squaring discourages large magnitudes
    return torch.mean(torch.logsumexp(logits, dim=-1) ** 2)


class MoMLinear(nn.Module):
    """Sample-level MoM Linear.

    Applies K expert linear heads to x (B, N, D), and mixes with router probs.
    Router uses x.mean(dim=1) -> (B, D).
    """

    def __init__(
        self,
        d_model: int,
        out_dim: int,
        num_experts: int = 4,
        top_k: int | None = None,
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

        self.router = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.num_experts, bias=True),
        )

        self.experts = nn.ModuleList([
            nn.Linear(self.d_model, self.out_dim, bias=True)
            for _ in range(self.num_experts)
        ])

        # stats
        self.last_probs: torch.Tensor | None = None
        self.last_logits: torch.Tensor | None = None
        self.aux_loss: torch.Tensor = torch.zeros(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) -> (B, N, out_dim)"""
        if x.dim() != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"Expected x shape (B,N,{self.d_model}), got {tuple(x.shape)}")

        router_in = x.mean(dim=1)  # (B, D)
        logits = self.router(router_in)  # (B, K)
        probs = _topk_masked_softmax(logits, top_k=self.top_k, temperature=self.temperature)  # (B, K)

        # compute experts
        ys = []
        for e in self.experts:
            ys.append(e(x))  # (B, N, out_dim)
        y_stack = torch.stack(ys, dim=-1)  # (B, N, out_dim, K)
        y = torch.einsum("bnok,bk->bno", y_stack, probs)

        # aux losses
        lb = _load_balance_loss(probs)
        zl = _router_z_loss(logits)
        self.aux_loss = self.aux_balance_coef * lb + self.aux_z_coef * zl

        # save stats (detach so they don't keep graph)
        self.last_probs = probs.detach()
        self.last_logits = logits.detach()

        return y


class FactorizedSTBlock(nn.Module):
    """Baseline Factorized Space-Time Transformer block.

    x: (B, T, N, D)
      1) Spatial self-attention per frame: (B*T, N, D)
      2) Temporal self-attention per patch: (B*N, T, D)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.ln_s1 = nn.LayerNorm(d_model)
        self.attn_s = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout, batch_first=True)
        self.ln_s2 = nn.LayerNorm(d_model)
        self.mlp_s = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )

        self.ln_t1 = nn.LayerNorm(d_model)
        self.attn_t = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout, batch_first=True)
        self.ln_t2 = nn.LayerNorm(d_model)
        self.mlp_t = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        xs = self.ln_s1(x).reshape(B * T, N, D)
        ys, _ = self.attn_s(xs, xs, xs, need_weights=False)
        x = x + ys.reshape(B, T, N, D)
        x = x + self.mlp_s(self.ln_s2(x))

        xt = self.ln_t1(x).permute(0, 2, 1, 3).reshape(B * N, T, D)
        yt, _ = self.attn_t(xt, xt, xt, need_weights=False)
        yt = yt.reshape(B, N, T, D).permute(0, 2, 1, 3)
        x = x + yt
        x = x + self.mlp_t(self.ln_t2(x))
        return x


class TimeSformer_MoMHead(nn.Module):
    """TimeSformer with Head-level MoM (MoM future_head).

    I/O:
      input : (B, Tin, 1, H, W)
      output: (B, Tout, 1, H, W)

    This variant replaces the one-shot future_head with MoMLinear.
    """

    def __init__(
        self,
        image_size: int = 32,
        spatial_patch: int = 4,
        in_chans: int = 1,
        tin: int = 14,
        tout: int = 7,
        d_model: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_pad: bool = True,
        # MoM head params
        head_num_experts: int = 4,
        head_top_k: int | None = None,
        head_temperature: float = 1.0,
        aux_balance_coef: float = 1e-2,
        aux_z_coef: float = 1e-3,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.spatial_patch = int(spatial_patch)
        self.in_chans = int(in_chans)
        self.tin = int(tin)
        self.tout = int(tout)
        self.d_model = int(d_model)
        self.use_pad = bool(use_pad)

        self.patch_embed = nn.Conv2d(in_chans, d_model, kernel_size=self.spatial_patch, stride=self.spatial_patch, bias=True)

        self.time_pos = nn.Embedding(self.tin + 1, d_model)
        max_gh = (self.image_size + self.spatial_patch - 1) // self.spatial_patch
        max_gw = (self.image_size + self.spatial_patch - 1) // self.spatial_patch
        self.max_patches = max_gh * max_gw
        self.patch_pos = nn.Embedding(self.max_patches + 1, d_model)

        self.blocks = nn.ModuleList([
            FactorizedSTBlock(d_model, num_heads, mlp_ratio=mlp_ratio, dropout=dropout, attn_dropout=attn_dropout)
            for _ in range(int(depth))
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # ✅ MoM future head
        self.future_head = MoMLinear(
            d_model=d_model,
            out_dim=self.tout * d_model,
            num_experts=head_num_experts,
            top_k=head_top_k,
            temperature=head_temperature,
            aux_balance_coef=aux_balance_coef,
            aux_z_coef=aux_z_coef,
        )

        self.to_patch = nn.Linear(d_model, in_chans * self.spatial_patch * self.spatial_patch)

        # exposed for trainer
        self.aux_loss: torch.Tensor = torch.zeros(())

    def _pad_to_multiple(self, x: torch.Tensor, multiple: int):
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (pad_h, pad_w)

    def _patchify(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        x2 = x.reshape(B * T, C, H, W)
        if self.use_pad:
            x2, (pad_h, pad_w) = self._pad_to_multiple(x2, self.spatial_patch)
        else:
            pad_h, pad_w = 0, 0

        Hp, Wp = x2.shape[-2], x2.shape[-1]
        tok = self.patch_embed(x2)  # (B*T, D, gh, gw)
        gh, gw = tok.shape[-2:]
        tok = tok.flatten(2).transpose(1, 2)  # (B*T, N, D)
        tok = tok.reshape(B, T, gh * gw, self.d_model)
        meta = (gh, gw, Hp, Wp, pad_h, pad_w, H, W)
        return tok, meta

    def _unpatchify(self, tok: torch.Tensor, meta):
        gh, gw, Hp, Wp, pad_h, pad_w, H, W = meta
        B, T, N, D = tok.shape
        ps = self.spatial_patch
        x = self.to_patch(tok)  # (B, Tout, N, C*ps*ps)
        x = x.view(B, T, gh, gw, self.in_chans, ps, ps)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(B, T, self.in_chans, Hp, Wp)
        if pad_h > 0:
            x = x[..., :H, :]
        if pad_w > 0:
            x = x[..., :W]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Tin, C, H, W = x.shape
        assert Tin == self.tin, f"Tin mismatch: got {Tin}, expected {self.tin}"
        assert C == self.in_chans, f"in_chans mismatch: got {C}, expected {self.in_chans}"

        tok, meta = self._patchify(x)  # (B, Tin, N, D)
        gh, gw, Hp, Wp, pad_h, pad_w, H0, W0 = meta
        N = gh * gw

        t_idx = torch.arange(Tin, device=x.device).clamp_max(self.tin)
        tok = tok + self.time_pos(t_idx).view(1, Tin, 1, self.d_model)

        if N <= self.max_patches:
            p_idx = torch.arange(N, device=x.device)
            tok = tok + self.patch_pos(p_idx).view(1, 1, N, self.d_model)

        h = tok
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        rep = h[:, -1, :, :]  # (B, N, D)

        # MoM head
        fut_flat = self.future_head(rep)  # (B, N, Tout*D)
        self.aux_loss = getattr(self.future_head, "aux_loss", torch.zeros((), device=x.device))
        fut = fut_flat.view(B, N, self.tout, self.d_model).permute(0, 2, 1, 3).contiguous()  # (B, Tout, N, D)
        y = self._unpatchify(fut, meta)
        return y


# short alias
class TimeSformerMoMHead(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = TimeSformer_MoMHead(**kwargs)
    def forward(self, x):
        return self.net(x)
