import torch
import torch.nn as nn
import torch.nn.functional as F


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
        b, t, n, d = x.shape

        xs = self.ln_s(x).reshape(b * t, n, d)
        ys, _ = self.attn_s(xs, xs, xs, need_weights=False)
        x = x + ys.reshape(b, t, n, d)

        xt = self.ln_t(x).permute(0, 2, 1, 3).reshape(b * n, t, d)
        yt, _ = self.attn_t(xt, xt, xt, need_weights=False)
        x = x + yt.reshape(b, n, t, d).permute(0, 2, 1, 3)

        x = x + self.mlp(self.ln_m(x))
        return x


class MSFormerV3ResDecode(nn.Module):
    """
    MSFv3 trunk + coarse-to-fine residual decoder.

    Decoder semantics (for scales sorted from coarse to fine):
        y_{s_coarse} = base_{s_coarse}
        y_s = y_{prev} + r_s

    Each scale branch still uses the original future head and token-to-patch decoder.
    The only change is how decoded branch maps are combined: additive residual hierarchy
    replaces flat summation over all scales.
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
        fusion="sum",          # "sum", "gated", "gated2"
        use_pad=True,
        use_pos_emb=False,
        time_pool="last",      # "mean" or "last"
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
        self.scales_desc = tuple(sorted(self.scales, reverse=True))
        self.scales_asc = tuple(sorted(self.scales))
        self.fusion = fusion
        assert self.fusion in ["sum", "gated", "gated2"]
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

        self.tokenizers = nn.ModuleDict()
        for s in self.scales:
            self.tokenizers[str(s)] = nn.Conv2d(
                self.in_chans, self.d_model, kernel_size=s, stride=s, bias=True
            )

        self.time_emb = nn.Embedding(self.tin, self.d_model)
        self.scale_emb = nn.Embedding(len(self.scales), self.d_model)

        self.pos_emb = nn.ModuleDict()
        if self.use_pos_emb:
            for s in self.scales:
                gs = self.image_size // s
                ns = gs * gs
                self.pos_emb[str(s)] = nn.Embedding(ns, self.d_model)

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
        b = fut.shape[0]
        patch_vec = self.token_to_patch[str(s)](fut)
        patch_vec = patch_vec.reshape(b, self.tout, gh, gw, self.in_chans, s, s)
        patch_vec = patch_vec.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        y = patch_vec.reshape(b, self.tout, self.in_chans, gh * s, gw * s)
        return y

    def _pool_scale_rep(self, tok):
        # tok: (B, Tin, Ns, D)
        if self.time_pool == "last":
            rep = tok[:, -1].mean(dim=1)   # (B,D)
        else:
            rep = tok.mean(dim=(1, 2))     # (B,D)
        return rep

    def forward(self, x):
        b, tin, c, h, w = x.shape
        assert c == self.in_chans
        assert tin == self.tin, f"Tin mismatch: got {tin}, expected {self.tin}"

        xt = x.reshape(b * tin, c, h, w)
        if self.use_pad:
            xt, (pad_h, pad_w) = self._pad_to_multiple(xt, max(self.scales))
        else:
            pad_h, pad_w = 0, 0

        scale_tokens = []
        scale_meta = []
        t_idx = torch.arange(tin, device=x.device)

        for sid, s in enumerate(self.scales):
            tok_s, gh, gw = self._tokenize_one_scale(xt, s)
            ns = gh * gw
            tok_s = tok_s.reshape(b, tin, ns, self.d_model)

            tok_s = tok_s + self.time_emb(t_idx).view(1, tin, 1, self.d_model)
            sid_tensor = torch.tensor(sid, device=x.device)
            tok_s = tok_s + self.scale_emb(sid_tensor).view(1, 1, 1, self.d_model)

            if self.use_pos_emb:
                p_idx = torch.arange(ns, device=x.device)
                tok_s = tok_s + self.pos_emb[str(s)](p_idx).view(1, 1, ns, self.d_model)

            scale_tokens.append(tok_s)
            scale_meta.append((s, ns, gh, gw))

        if self.fusion == "gated":
            reps = [self._pool_scale_rep(tok_s) for tok_s in scale_tokens]
            cat = torch.cat(reps, dim=-1)
            w_scale = torch.softmax(self.gate(cat), dim=-1)
        else:
            w_scale = None

        z = torch.cat(scale_tokens, dim=2)
        htok = z
        for blk in self.blocks:
            htok = blk(htok)
        htok = self.ln_f(htok)

        if self.fusion == "gated2":
            reps_h = []
            cursor = 0
            for _, ns, _, _ in scale_meta:
                tok_h = htok[:, :, cursor:cursor + ns, :]
                cursor += ns
                reps_h.append(self._pool_scale_rep(tok_h))
            cat_h = torch.cat(reps_h, dim=-1)
            w_scale = torch.softmax(self.gate(cat_h), dim=-1)

        # Decode each scale branch exactly as in MSFv3, but store maps by scale.
        decoded_by_scale = {}
        cursor = 0
        scale_to_index = {s: i for i, s in enumerate(self.scales)}
        for s, ns, gh, gw in scale_meta:
            tok = htok[:, :, cursor:cursor + ns, :]
            cursor += ns

            rep = tok[:, -1, :, :] if self.time_pool == "last" else tok.mean(dim=1)
            fut = self.future_heads[str(s)](rep).reshape(b, ns, self.tout, self.d_model)
            fut = fut.permute(0, 2, 1, 3).contiguous()
            y_s = self._decode_scale(fut, s, gh, gw)

            if w_scale is not None:
                idx = scale_to_index[s]
                y_s = y_s * w_scale[:, idx].view(b, 1, 1, 1, 1)

            decoded_by_scale[s] = y_s

        # Coarse-to-fine residual accumulation.
        running = None
        for s in self.scales_desc:
            branch = decoded_by_scale[s]
            if running is None:
                running = branch           # coarsest base
            else:
                running = running + branch # finer residual correction

        y = running

        if self.use_pad and (pad_h > 0 or pad_w > 0):
            y = y[..., :h, :w]

        b2, t2, c2, h2, w2 = y.shape
        y2 = y.reshape(b2 * t2, c2, h2, w2)
        y2 = y2 + self.smoother(y2)
        y = y2.reshape(b2, t2, c2, h2, w2)
        return y


class MSFv3ResDecode(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs.setdefault("fusion", "sum")
        self.net = MSFormerV3ResDecode(**kwargs)

    def forward(self, x):
        return self.net(x)


class MSFGv3ResDecode(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs["fusion"] = "gated"
        self.net = MSFormerV3ResDecode(**kwargs)

    def forward(self, x):
        return self.net(x)


class MSFG2v3ResDecode(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs["fusion"] = "gated2"
        self.net = MSFormerV3ResDecode(**kwargs)

    def forward(self, x):
        return self.net(x)
