import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankAxisEmbedding(nn.Module):
    """
    Factorized 1D embedding:
        E[i, d] = sum_r axis_factors[i, r] * channel_basis[r, d]

    Params:
        axis_factors : (L, R)
        channel_basis: (R, D)

    -> full table (L, D) 대신 low-rank factorization 사용
    -> 해석:
       - axis_factors[i] : i번째 축 원소(예: time step, scale id)의 rank-wise coefficient
       - channel_basis[r] : 공통 basis direction
    """
    def __init__(self, length: int, d_model: int, rank: int = 4, init_std: float = 0.02):
        super().__init__()
        self.length = int(length)
        self.d_model = int(d_model)
        self.rank = int(rank)

        self.axis_factors = nn.Parameter(torch.randn(self.length, self.rank) * init_std)
        self.channel_basis = nn.Parameter(torch.randn(self.rank, self.d_model) * init_std)

    def weight(self) -> torch.Tensor:
        return self.axis_factors @ self.channel_basis  # (L, D)

    def forward(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        w = self.weight()
        if idx is None:
            return w
        return w[idx]


class LowRankGridEmbedding2D(nn.Module):
    """
    Factorized 2D positional embedding with shared channel basis:
        row_emb[h, d] = sum_r row_factors[h, r] * channel_basis[r, d]
        col_emb[w, d] = sum_r col_factors[w, r] * channel_basis[r, d]
        pos[h, w, d]  = row_emb[h, d] + col_emb[w, d]

    Params:
        row_factors  : (H, R)
        col_factors  : (W, R)
        channel_basis: (R, D)

    -> flat pos table (H*W, D)보다 더 구조적이고 해석적임
    """
    def __init__(self, max_h: int, max_w: int, d_model: int, rank: int = 4, init_std: float = 0.02):
        super().__init__()
        self.max_h = int(max_h)
        self.max_w = int(max_w)
        self.d_model = int(d_model)
        self.rank = int(rank)

        self.row_factors = nn.Parameter(torch.randn(self.max_h, self.rank) * init_std)
        self.col_factors = nn.Parameter(torch.randn(self.max_w, self.rank) * init_std)
        self.channel_basis = nn.Parameter(torch.randn(self.rank, self.d_model) * init_std)

    def weight_2d(self, h: int | None = None, w: int | None = None) -> torch.Tensor:
        if h is None:
            h = self.max_h
        if w is None:
            w = self.max_w
        h = int(h)
        w = int(w)

        row = self.row_factors[:h] @ self.channel_basis  # (h, D)
        col = self.col_factors[:w] @ self.channel_basis  # (w, D)
        return row[:, None, :] + col[None, :, :]         # (h, w, D)

    def forward(self, h: int | None = None, w: int | None = None) -> torch.Tensor:
        pos = self.weight_2d(h=h, w=w)
        return pos.reshape(-1, self.d_model)             # (h*w, D)


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


class MSFormerV3(nn.Module):
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
        fusion="sum",              # "sum", "gated", "gated2"
        use_pad=True,
        use_pos_emb=True,          # 3개 모두 사용하려면 True 권장
        time_pool="last",          # "mean" or "last"

        # factorized embedding options
        factorize_time_emb=True,
        factorize_scale_emb=True,
        factorize_pos_emb=True,
        time_rank=4,
        scale_rank=4,
        pos_rank=4,
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
        self.fusion = fusion
        assert self.fusion in ["sum", "gated", "gated2"]
        self.use_pad = bool(use_pad)
        self.use_pos_emb = bool(use_pos_emb)
        self.time_pool = str(time_pool)
        assert self.time_pool in ["mean", "last"]

        self.factorize_time_emb = bool(factorize_time_emb)
        self.factorize_scale_emb = bool(factorize_scale_emb)
        self.factorize_pos_emb = bool(factorize_pos_emb)

        self.time_rank = int(time_rank)
        self.scale_rank = int(scale_rank)
        self.pos_rank = int(pos_rank)

        self.smoother = nn.Sequential(
            nn.Conv2d(self.in_chans, self.in_chans, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.in_chans, self.in_chans, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.in_chans, self.in_chans, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.in_chans, self.in_chans, kernel_size=3, padding=1),
        )

        self.tokenizers = nn.ModuleDict()
        for s in self.scales:
            self.tokenizers[str(s)] = nn.Conv2d(
                self.in_chans, self.d_model, kernel_size=s, stride=s, bias=True
            )

        # ------------------------------------------------------------------
        # 1) Time embedding
        # ------------------------------------------------------------------
        if self.factorize_time_emb:
            self.time_emb = LowRankAxisEmbedding(
                length=self.tin,
                d_model=self.d_model,
                rank=self.time_rank,
            )
        else:
            self.time_emb = nn.Embedding(self.tin, self.d_model)

        # ------------------------------------------------------------------
        # 2) Scale embedding
        # ------------------------------------------------------------------
        if self.factorize_scale_emb:
            self.scale_emb = LowRankAxisEmbedding(
                length=len(self.scales),
                d_model=self.d_model,
                rank=self.scale_rank,
            )
        else:
            self.scale_emb = nn.Embedding(len(self.scales), self.d_model)

        # ------------------------------------------------------------------
        # 3) Position embedding
        # ------------------------------------------------------------------
        self.pos_emb = nn.ModuleDict()
        if self.use_pos_emb:
            for s in self.scales:
                max_gh = math.ceil(self.image_size / s)
                max_gw = math.ceil(self.image_size / s)

                if self.factorize_pos_emb:
                    self.pos_emb[str(s)] = LowRankGridEmbedding2D(
                        max_h=max_gh,
                        max_w=max_gw,
                        d_model=self.d_model,
                        rank=self.pos_rank,
                    )
                else:
                    self.pos_emb[str(s)] = nn.Embedding(max_gh * max_gw, self.d_model)

        self.blocks = nn.ModuleList([
            FactorizedSTBlock(self.d_model, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(self.d_model)

        self.future_heads = nn.ModuleDict()
        self.token_to_patch = nn.ModuleDict()
        for s in self.scales:
            self.future_heads[str(s)] = nn.Linear(self.d_model, self.tout * self.d_model)
            self.token_to_patch[str(s)] = nn.Linear(self.d_model, self.in_chans * s * s)

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
        tok = self.tokenizers[str(s)](x_btchw)
        gh, gw = tok.shape[-2], tok.shape[-1]
        tok = tok.flatten(2).transpose(1, 2)
        return tok, gh, gw

    def _decode_scale(self, fut, s: int, gh: int, gw: int):
        B = fut.shape[0]
        patch_vec = self.token_to_patch[str(s)](fut)
        patch_vec = patch_vec.reshape(B, self.tout, gh, gw, self.in_chans, s, s)
        patch_vec = patch_vec.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        y = patch_vec.reshape(B, self.tout, self.in_chans, gh * s, gw * s)
        return y

    def _pool_scale_rep(self, tok):
        if self.time_pool == "last":
            rep = tok[:, -1].mean(dim=1)   # (B, D)
        else:
            rep = tok.mean(dim=(1, 2))     # (B, D)
        return rep

    def _get_time_emb(self, t_idx: torch.Tensor) -> torch.Tensor:
        return self.time_emb(t_idx)  # (Tin, D)

    def _get_scale_emb(self, s_idx: torch.Tensor) -> torch.Tensor:
        return self.scale_emb(s_idx)  # (D,) or (S,D) depending on idx

    def _get_pos_emb(self, scale: int, gh: int, gw: int, device: torch.device) -> torch.Tensor:
        mod = self.pos_emb[str(scale)]
        if isinstance(mod, LowRankGridEmbedding2D):
            return mod(h=gh, w=gw)  # (Ns, D)
        p_idx = torch.arange(gh * gw, device=device)
        return mod(p_idx)

    def forward(self, x):
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

        t_idx = torch.arange(Tin, device=x.device, dtype=torch.long)
        t_emb = self._get_time_emb(t_idx).view(1, Tin, 1, self.d_model)  # (1, Tin, 1, D)

        for sid, s in enumerate(self.scales):
            tok_s, gh, gw = self._tokenize_one_scale(xt, s)
            Ns = gh * gw
            tok_s = tok_s.reshape(B, Tin, Ns, self.d_model)

            # 1) time embedding
            tok_s = tok_s + t_emb

            # 2) scale embedding
            sid_tensor = torch.tensor(sid, device=x.device, dtype=torch.long)
            s_emb = self._get_scale_emb(sid_tensor).view(1, 1, 1, self.d_model)
            tok_s = tok_s + s_emb

            # 3) position embedding
            if self.use_pos_emb:
                p_emb = self._get_pos_emb(s, gh, gw, x.device).view(1, 1, Ns, self.d_model)
                tok_s = tok_s + p_emb

            scale_tokens.append(tok_s)
            scale_meta.append((s, Ns, gh, gw))

        if self.fusion == "gated":
            reps = [self._pool_scale_rep(tok_s) for tok_s in scale_tokens]
            cat = torch.cat(reps, dim=-1)
            w = torch.softmax(self.gate(cat), dim=-1)
        else:
            w = None

        z = torch.cat(scale_tokens, dim=2)  # (B, Tin, N_total, D)
        h = z
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)

        if self.fusion == "gated2":
            reps_h = []
            cursor = 0
            for _, Ns, _, _ in scale_meta:
                tok_h = h[:, :, cursor:cursor + Ns, :]
                cursor += Ns
                reps_h.append(self._pool_scale_rep(tok_h))
            cat_h = torch.cat(reps_h, dim=-1)
            w = torch.softmax(self.gate(cat_h), dim=-1)

        ys = []
        cursor = 0
        for i, (s, Ns, gh, gw) in enumerate(scale_meta):
            tok = h[:, :, cursor:cursor + Ns, :]
            cursor += Ns

            rep = tok[:, -1, :, :] if self.time_pool == "last" else tok.mean(dim=1)

            fut = self.future_heads[str(s)](rep).reshape(B, Ns, self.tout, self.d_model)
            fut = fut.permute(0, 2, 1, 3).contiguous()

            y_s = self._decode_scale(fut, s, gh, gw)

            if w is not None:
                y_s = y_s * w[:, i].view(B, 1, 1, 1, 1)

            ys.append(y_s)

        y = torch.stack(ys, dim=0).sum(dim=0)

        if self.use_pad and (pad_h > 0 or pad_w > 0):
            y = y[..., :H, :W]

        B2, T2, C2, H2, W2 = y.shape
        y2 = y.reshape(B2 * T2, C2, H2, W2)
        y2 = y2 + self.smoother(y2)
        y = y2.reshape(B2, T2, C2, H2, W2)
        return y


class MSFv3(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs.setdefault("fusion", "sum")
        kwargs.setdefault("use_pos_emb", True)

        kwargs.setdefault("factorize_time_emb", True)
        kwargs.setdefault("factorize_scale_emb", True)
        kwargs.setdefault("factorize_pos_emb", True)

        kwargs.setdefault("time_rank", 4)
        kwargs.setdefault("scale_rank", 4)
        kwargs.setdefault("pos_rank", 4)

        self.net = MSFormerV3(**kwargs)

    def forward(self, x):
        return self.net(x)


class MSFGv3(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs["fusion"] = "gated"
        kwargs.setdefault("use_pos_emb", True)

        kwargs.setdefault("factorize_time_emb", True)
        kwargs.setdefault("factorize_scale_emb", True)
        kwargs.setdefault("factorize_pos_emb", True)

        kwargs.setdefault("time_rank", 4)
        kwargs.setdefault("scale_rank", 4)
        kwargs.setdefault("pos_rank", 4)

        self.net = MSFormerV3(**kwargs)

    def forward(self, x):
        return self.net(x)


class MSFG2v3(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs["fusion"] = "gated2"
        kwargs.setdefault("use_pos_emb", True)

        kwargs.setdefault("factorize_time_emb", True)
        kwargs.setdefault("factorize_scale_emb", True)
        kwargs.setdefault("factorize_pos_emb", True)

        kwargs.setdefault("time_rank", 4)
        kwargs.setdefault("scale_rank", 4)
        kwargs.setdefault("pos_rank", 4)

        self.net = MSFormerV3(**kwargs)

    def forward(self, x):
        return self.net(x)