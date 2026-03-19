import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedSTBlock(nn.Module):
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.ln_s = nn.LayerNorm(d_model)
        self.attn_s = nn.MultiheadAttention(
            d_model, nhead, dropout=attn_dropout, batch_first=True
        )

        self.ln_t = nn.LayerNorm(d_model)
        self.attn_t = nn.MultiheadAttention(
            d_model, nhead, dropout=attn_dropout, batch_first=True
        )

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
        # x: (B, T, N, D)
        B, T, N, D = x.shape

        xs = self.ln_s(x).reshape(B * T, N, D)
        ys, _ = self.attn_s(xs, xs, xs, need_weights=False)
        x = x + ys.reshape(B, T, N, D)

        xt = self.ln_t(x).permute(0, 2, 1, 3).reshape(B * N, T, D)
        yt, _ = self.attn_t(xt, xt, xt, need_weights=False)
        x = x + yt.reshape(B, N, T, D).permute(0, 2, 1, 3)

        x = x + self.mlp(self.ln_m(x))
        return x


class SmallScaleRefiner(nn.Module):
    """
    Per-scale spatial refiner.

    변경점:
    - 마지막 global smoother 제거
    - 각 scale branch 내부에서만 refinement 수행
    - depth 기본 5
    - residual / replace 지원
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = None,
        depth: int = 5,
        mode: str = "residual",
    ):
        super().__init__()

        hidden_channels = int(hidden_channels or channels)
        depth = int(depth)
        mode = str(mode)

        if depth < 1:
            raise ValueError("decoder_refine_depth must be >= 1")
        if mode not in {"residual", "replace"}:
            raise ValueError(f"Unsupported decoder_refine_mode: {mode}")

        self.mode = mode

        layers = []
        in_ch = channels
        for i in range(depth):
            out_ch = channels if i == depth - 1 else hidden_channels
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if i != depth - 1:
                layers.append(nn.GELU())
            in_ch = hidden_channels

        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, Tout, C, H, W)
        B, T, C, H, W = y.shape
        y2 = y.reshape(B * T, C, H, W)
        refined = self.net(y2)

        if self.mode == "residual":
            out = y2 + refined
        else:
            out = refined

        return out.reshape(B, T, C, H, W)


class MSFormerV3Conv(nn.Module):
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
        fusion="sum",          # "sum", "gated", "gated2"
        use_pad=True,
        use_pos_emb=False,
        time_pool="last",      # "mean" or "last"
        decoder_refine_hidden=None,
        decoder_refine_depth=5,
        decoder_refine_mode="residual",  # "residual" or "replace"
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
        assert self.fusion in ["sum", "gated", "gated2"]

        self.use_pad = bool(use_pad)
        self.use_pos_emb = bool(use_pos_emb)

        self.time_pool = str(time_pool)
        assert self.time_pool in ["mean", "last"]

        self.decoder_refine_mode = str(decoder_refine_mode)
        assert self.decoder_refine_mode in ["residual", "replace"]
        self.decoder_refine_depth = int(decoder_refine_depth)

        # tokenizer per scale
        self.tokenizers = nn.ModuleDict()
        for s in self.scales:
            self.tokenizers[str(s)] = nn.Conv2d(
                self.in_chans,
                self.d_model,
                kernel_size=s,
                stride=s,
                bias=True,
            )

        # embeddings
        self.time_emb = nn.Embedding(self.tin, self.d_model)
        self.scale_emb = nn.Embedding(len(self.scales), self.d_model)

        self.pos_emb = nn.ModuleDict()
        if self.use_pos_emb:
            for s in self.scales:
                gs = self.image_size // s
                Ns = gs * gs
                self.pos_emb[str(s)] = nn.Embedding(Ns, self.d_model)

        # transformer
        self.blocks = nn.ModuleList([
            FactorizedSTBlock(
                self.d_model,
                num_heads,
                mlp_ratio,
                dropout,
                attn_dropout,
            )
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(self.d_model)

        # scale-wise heads / decoder / refiner
        self.future_heads = nn.ModuleDict()
        self.token_to_patch = nn.ModuleDict()
        self.scale_refiners = nn.ModuleDict()

        for s in self.scales:
            self.future_heads[str(s)] = nn.Linear(
                self.d_model, self.tout * self.d_model
            )
            self.token_to_patch[str(s)] = nn.Linear(
                self.d_model, self.in_chans * s * s
            )
            self.scale_refiners[str(s)] = SmallScaleRefiner(
                channels=self.in_chans,
                hidden_channels=decoder_refine_hidden,
                depth=self.decoder_refine_depth,
                mode=self.decoder_refine_mode,
            )

        # gate
        if self.fusion in ["gated", "gated2"]:
            self.gate = nn.Sequential(
                nn.Linear(self.d_model * len(self.scales), self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, len(self.scales)),
            )
        else:
            self.gate = None

    def _pad_to_multiple(self, x, multiple: int):
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple

        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)

        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (pad_h, pad_w)

    def _tokenize_one_scale(self, x_btchw, s: int):
        tok = self.tokenizers[str(s)](x_btchw)   # (B*T, D, gh, gw)
        gh, gw = tok.shape[-2], tok.shape[-1]
        tok = tok.flatten(2).transpose(1, 2)     # (B*T, Ns, D)
        return tok, gh, gw

    def _decode_scale(self, fut, s: int, gh: int, gw: int):
        # fut: (B, Tout, Ns, D)
        B = fut.shape[0]
        patch_vec = self.token_to_patch[str(s)](fut)  # (B, Tout, Ns, C*s*s)

        patch_vec = patch_vec.reshape(
            B, self.tout, gh, gw, self.in_chans, s, s
        )
        patch_vec = patch_vec.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        y = patch_vec.reshape(B, self.tout, self.in_chans, gh * s, gw * s)
        return y

    def _pool_scale_rep(self, tok):
        # tok: (B, Tin, Ns, D)
        if self.time_pool == "last":
            rep = tok[:, -1].mean(dim=1)   # (B, D)
        else:
            rep = tok.mean(dim=(1, 2))     # (B, D)
        return rep

    def _refine_scale(self, y_s: torch.Tensor, s: int) -> torch.Tensor:
        return self.scale_refiners[str(s)](y_s)

    def forward(self, x):
        # x: (B, Tin, C, H, W)
        B, Tin, C, H, W = x.shape
        assert C == self.in_chans
        assert Tin == self.tin, f"Tin mismatch: got {Tin}, expected {self.tin}"

        xt = x.reshape(B * Tin, C, H, W)

        if self.use_pad:
            xt, (pad_h, pad_w) = self._pad_to_multiple(xt, max(self.scales))
        else:
            pad_h, pad_w = 0, 0

        scale_tokens = []
        scale_meta = []
        t_idx = torch.arange(Tin, device=x.device)

        # tokenize per scale
        for sid, s in enumerate(self.scales):
            tok_s, gh, gw = self._tokenize_one_scale(xt, s)
            Ns = gh * gw
            tok_s = tok_s.reshape(B, Tin, Ns, self.d_model)

            tok_s = tok_s + self.time_emb(t_idx).view(1, Tin, 1, self.d_model)

            sid_tensor = torch.tensor(sid, device=x.device)
            tok_s = tok_s + self.scale_emb(sid_tensor).view(1, 1, 1, self.d_model)

            if self.use_pos_emb:
                p_idx = torch.arange(Ns, device=x.device)
                tok_s = tok_s + self.pos_emb[str(s)](p_idx).view(1, 1, Ns, self.d_model)

            scale_tokens.append(tok_s)
            scale_meta.append((s, Ns, gh, gw))

        # pre-transformer gate
        if self.fusion == "gated":
            reps = [self._pool_scale_rep(tok_s) for tok_s in scale_tokens]
            cat = torch.cat(reps, dim=-1)
            w = torch.softmax(self.gate(cat), dim=-1)
        else:
            w = None

        # transformer
        z = torch.cat(scale_tokens, dim=2)  # (B, Tin, N_total, D)
        h = z
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)

        # post-transformer gate
        if self.fusion == "gated2":
            reps_h = []
            cursor = 0
            for _, Ns, _, _ in scale_meta:
                tok_h = h[:, :, cursor:cursor + Ns, :]
                cursor += Ns
                reps_h.append(self._pool_scale_rep(tok_h))

            cat_h = torch.cat(reps_h, dim=-1)
            w = torch.softmax(self.gate(cat_h), dim=-1)

        # decode per scale
        ys = []
        cursor = 0
        for i, (s, Ns, gh, gw) in enumerate(scale_meta):
            tok = h[:, :, cursor:cursor + Ns, :]
            cursor += Ns

            if self.time_pool == "last":
                rep = tok[:, -1, :, :]      # (B, Ns, D)
            else:
                rep = tok.mean(dim=1)       # (B, Ns, D)

            fut = self.future_heads[str(s)](rep).reshape(
                B, Ns, self.tout, self.d_model
            )
            fut = fut.permute(0, 2, 1, 3).contiguous()  # (B, Tout, Ns, D)

            y_s = self._decode_scale(fut, s, gh, gw)
            y_s = self._refine_scale(y_s, s)

            if w is not None:
                y_s = y_s * w[:, i].view(B, 1, 1, 1, 1)

            ys.append(y_s)

        y = torch.stack(ys, dim=0).sum(dim=0)

        if self.use_pad and (pad_h > 0 or pad_w > 0):
            y = y[..., :H, :W]

        return y


class MSFv3Conv(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs.setdefault("fusion", "sum")
        self.net = MSFormerV3Conv(**kwargs)

    def forward(self, x):
        return self.net(x)


class MSFGv3Conv(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs["fusion"] = "gated"
        self.net = MSFormerV3Conv(**kwargs)

    def forward(self, x):
        return self.net(x)


class MSFG2v3Conv(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs["fusion"] = "gated2"
        self.net = MSFormerV3Conv(**kwargs)

    def forward(self, x):
        return self.net(x)