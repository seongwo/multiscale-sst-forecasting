
import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedSTBlock(nn.Module):
    """
    Factorized Space-Time Transformer block
    x: (B, T, N, D)

    1) Spatial self-attention per frame: (B*T, N, D)
    2) Temporal self-attention per patch: (B*N, T, D)
    """
    def __init__(self, d_model: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        # spatial sub-layer
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

        # temporal sub-layer
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

        # ---- Spatial mixing: for each time t, attend over N patches
        xs = self.ln_s1(x).reshape(B * T, N, D)                 # (B*T, N, D)
        ys, _ = self.attn_s(xs, xs, xs, need_weights=False)     # (B*T, N, D)
        x = x + ys.reshape(B, T, N, D)
        x = x + self.mlp_s(self.ln_s2(x))

        # ---- Temporal mixing: for each patch n, attend over T steps
        xt = self.ln_t1(x).permute(0, 2, 1, 3).reshape(B * N, T, D)  # (B*N, T, D)
        yt, _ = self.attn_t(xt, xt, xt, need_weights=False)          # (B*N, T, D)
        yt = yt.reshape(B, N, T, D).permute(0, 2, 1, 3)              # (B, T, N, D)
        x = x + yt
        x = x + self.mlp_t(self.ln_t2(x))

        return x


class TimeSformer(nn.Module):
    """
    Option A1 (2D SST):
    - Spatial patch tokenization (Conv2d patchify)
    - Factorized mixing: Spatial (per frame) -> Temporal (per patch)
    - One-shot forecast head
    - Token -> patch decode (fold to frames)

    I/O:
      input : (B, Tin, 1, H, W)
      output: (B, Tout, 1, H, W)
    """

    def __init__(
        self,
        image_size=32,
        spatial_patch=4,
        in_chans=1,
        tin=14,
        tout=7,
        d_model=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
        attn_dropout=0.0,
        use_pad=True,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.spatial_patch = int(spatial_patch)
        self.in_chans = int(in_chans)
        self.tin = int(tin)
        self.tout = int(tout)
        self.d_model = int(d_model)
        self.use_pad = bool(use_pad)

        # spatial patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, d_model, kernel_size=self.spatial_patch, stride=self.spatial_patch, bias=True
        )

        # embeddings
        self.time_pos = nn.Embedding(self.tin + 1, d_model)

        # patch id embedding (assume fixed max based on image_size)
        max_gh = (self.image_size + self.spatial_patch - 1) // self.spatial_patch
        max_gw = (self.image_size + self.spatial_patch - 1) // self.spatial_patch
        self.max_patches = max_gh * max_gw
        self.patch_pos = nn.Embedding(self.max_patches + 1, d_model)

        # factorized blocks
        self.blocks = nn.ModuleList([
            FactorizedSTBlock(d_model, num_heads, mlp_ratio=mlp_ratio, dropout=dropout, attn_dropout=attn_dropout)
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        self.gru_proj = nn.Sequential( nn.LayerNorm(d_model), nn.Linear(d_model, d_model))

        # token -> pixel patches
        self.to_patch = nn.Linear(d_model, in_chans * self.spatial_patch * self.spatial_patch)

        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)



    def _pad_to_multiple(self, x: torch.Tensor, multiple: int):
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (pad_h, pad_w)

    def _patchify(self, x):
        # x: (B,T,C,H,W)
        B, T, C, H, W = x.shape
        x2 = x.reshape(B * T, C, H, W)

        if self.use_pad:
            x2, (pad_h, pad_w) = self._pad_to_multiple(x2, self.spatial_patch)
        else:
            pad_h, pad_w = 0, 0

        Hp, Wp = x2.shape[-2], x2.shape[-1]
        tok = self.patch_embed(x2)              # (B*T, D, gh, gw)
        gh, gw = tok.shape[-2:]
        tok = tok.flatten(2).transpose(1, 2)    # (B*T, N, D)
        tok = tok.reshape(B, T, gh * gw, self.d_model)  # (B, T, N, D)

        meta = (gh, gw, Hp, Wp, pad_h, pad_w, H, W)
        return tok, meta

    def _unpatchify(self, tok, meta):
        # tok: (B, Tout, N, D)
        gh, gw, Hp, Wp, pad_h, pad_w, H, W = meta
        B, T, N, D = tok.shape
        ps = self.spatial_patch

        x = self.to_patch(tok)                  # (B, Tout, N, C*ps*ps)
        x = x.view(B, T, gh, gw, self.in_chans, ps, ps)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()  # (B, Tout, C, gh, ps, gw, ps)
        x = x.view(B, T, self.in_chans, Hp, Wp)

        # remove padding
        if pad_h > 0:
            x = x[..., :H, :]
        if pad_w > 0:
            x = x[..., :W]
        return x

    def forward(self, x):
        """
        x: (B, Tin, C, H, W)
        y: (B, Tout, C, H, W)
        """
        B, Tin, C, H, W = x.shape
        assert Tin == self.tin, f"Tin mismatch: got {Tin}, expected {self.tin}"
        assert C == self.in_chans, f"in_chans mismatch: got {C}, expected {self.in_chans}"

        tok, meta = self._patchify(x)   # (B, Tin, N, D)
        gh, gw, Hp, Wp, pad_h, pad_w, H0, W0 = meta
        N = gh * gw

        # add time embedding
        t_idx = torch.arange(Tin, device=x.device).clamp_max(self.tin)
        tok = tok + self.time_pos(t_idx).view(1, Tin, 1, self.d_model)

        # add patch embedding (only if within max)
        if N <= self.max_patches:
            p_idx = torch.arange(N, device=x.device)
            tok = tok + self.patch_pos(p_idx).view(1, 1, N, self.d_model)

        # factorized blocks: spatial -> temporal
        h = tok
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)  # (B, Tin, N, D)

        seq = h.permute(0,2,1,3).reshape(B*N, Tin, self.d_model)  # (B*N, Tin, D)
        out, h0 = self.gru(seq)

        preds = []
        cur = out[:, -1:, :]
        hcur = h0

        for _ in range(self.tout):
            y1, hcur = self.gru(cur, hcur)
            y1 = self.gru_proj(y1)
            preds.append(y1)
            cur = y1

        pred_tokens = torch.cat(preds, dim=1)  # (B*N, Tout, D)
        pred = pred_tokens.reshape(B,N, self.tout, self.d_model).permute(0,2,1,3)  # (B, Tout, N, D)
        y = self._unpatchify(pred, meta) 

        return y
