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
        B, T, N, D = x.shape
        xs = self.ln_s(x).reshape(B * T, N, D)
        ys, _ = self.attn_s(xs, xs, xs, need_weights=False)
        x = x + ys.reshape(B, T, N, D)

        xt = self.ln_t(x).permute(0, 2, 1, 3).reshape(B * N, T, D)
        yt, _ = self.attn_t(xt, xt, xt, need_weights=False)
        x = x + yt.reshape(B, N, T, D).permute(0, 2, 1, 3)

        x = x + self.mlp(self.ln_m(x))
        return x


class MSFormerV2(nn.Module):

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
        time_pool="mean",      # "mean" or "last"
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
        assert self.fusion in ["sum", "gated"]
        self.use_pad = bool(use_pad)
        self.use_pos_emb = bool(use_pos_emb)
        self.time_pool = str(time_pool)
        assert self.time_pool in ["mean", "last"]

        # tokenizers per scale (same as v1)
        self.tokenizers = nn.ModuleDict()
        for s in self.scales:
            self.tokenizers[str(s)] = nn.Conv2d(self.in_chans, self.d_model, kernel_size=s, stride=s, bias=True)

        # embeddings (same as v1)
        self.time_emb = nn.Embedding(self.tin, self.d_model)
        self.scale_emb = nn.Embedding(len(self.scales), self.d_model)

        self.pos_emb = nn.ModuleDict()
        if self.use_pos_emb:
            for s in self.scales:
                gs = self.image_size // s
                Ns = gs * gs
                self.pos_emb[str(s)] = nn.Embedding(Ns, self.d_model)

        # transformer blocks (same as v1)
        self.blocks = nn.ModuleList([
            FactorizedSTBlock(self.d_model, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(self.d_model)

        # ✅ v2 핵심: scale별 head + scale별 decoder
        self.future_heads = nn.ModuleDict()
        self.token_to_patch = nn.ModuleDict()
        for s in self.scales:
            self.future_heads[str(s)] = nn.Linear(self.d_model, self.tout * self.d_model)
            self.token_to_patch[str(s)] = nn.Linear(self.d_model, self.in_chans * s * s)

        # ✅ fusion="gated"일 때: batch-wise scale weight
        if self.fusion == "gated":
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
        tok = self.tokenizers[str(s)](x_btchw)          # (B*T, D, gh, gw)
        gh, gw = tok.shape[-2], tok.shape[-1]
        tok = tok.flatten(2).transpose(1, 2)            # (B*T, N_s, D)
        return tok, gh, gw

    def _decode_scale(self, fut, s: int, Hp: int, Wp: int, gh: int, gw: int):
        """
        fut: (B, Tout, N_s, D)
        return y_s: (B, Tout, C, Hp, Wp)  (s 패치로 fold하면 Hp,Wp로 정확히 복원됨)
        """
        B = fut.shape[0]
        patch_vec = self.token_to_patch[str(s)](fut)  # (B, Tout, N_s, C*s*s)
        patch_vec = patch_vec.reshape(B, self.tout, gh, gw, self.in_chans, s, s)
        patch_vec = patch_vec.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        y = patch_vec.reshape(B, self.tout, self.in_chans, gh * s, gw * s)
        # 여기서 gh*s == Hp, gw*s == Wp 여야 함
        return y

    def forward(self, x):
        # x: (B,Tin,C,H,W)
        B, Tin, C, H, W = x.shape
        assert C == self.in_chans
        assert Tin == self.tin, f"Tin mismatch: got {Tin}, expected {self.tin}"

        # pad to max scale (same idea as v1)
        xt = x.reshape(B * Tin, C, H, W)
        if self.use_pad:
            xt, (pad_h, pad_w) = self._pad_to_multiple(xt, max(self.scales))
        else:
            pad_h, pad_w = 0, 0
        Hp, Wp = xt.shape[-2], xt.shape[-1]

        # tokenize per scale + emb add
        scale_tokens = []
        scale_meta = []  # (s, Ns, gh, gw)
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

            scale_tokens.append(tok_s)
            scale_meta.append((s, Ns, gh, gw))

        # concat and transformer
        z = torch.cat(scale_tokens, dim=2)  # (B, Tin, N_total, D)
        h = z
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)  # (B, Tin, N_total, D)

        # gated fusion weights (batch-wise)
        if self.fusion == "gated":
            reps = []
            for tok_s in scale_tokens:  # (B,T,Ns,D) before transformer도 가능하지만, 일관성 위해 h에서 뽑아도 됨
                reps.append(tok_s.mean(dim=(1, 2)))  # (B,D)
            cat = torch.cat(reps, dim=-1)  # (B, D*S)
            w = torch.softmax(self.gate(cat), dim=-1)  # (B,S)
        else:
            w = None

        # scale-wise: slice -> pool -> head -> decode -> fuse
        ys = []
        cursor = 0
        for i, (s, Ns, gh, gw) in enumerate(scale_meta):
            tok = h[:, :, cursor:cursor + Ns, :]  # (B,Tin,Ns,D)
            cursor += Ns

            if self.time_pool == "last":
                rep = tok[:, -1, :, :]            # (B,Ns,D)
            else:
                rep = tok.mean(dim=1)             # (B,Ns,D)

            fut = self.future_heads[str(s)](rep).reshape(B, Ns, self.tout, self.d_model)  # (B,Ns,Tout,D)
            fut = fut.permute(0, 2, 1, 3).contiguous()                                    # (B,Tout,Ns,D)

            y_s = self._decode_scale(fut, s, Hp, Wp, gh, gw)  # (B,Tout,C,Hp,Wp)

            if w is not None:
                y_s = y_s * w[:, i].view(B, 1, 1, 1, 1)
            ys.append(y_s)

        y = torch.stack(ys, dim=0).sum(dim=0)  # (B,Tout,C,Hp,Wp)

        # crop back
        if self.use_pad and (pad_h > 0 or pad_w > 0):
            y = y[..., :H, :W]
        return y


# short names
class MSFv2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs.setdefault("fusion", "sum")
        self.net = MSFormerV2(**kwargs)
    def forward(self, x): return self.net(x)

class MSFGv2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs["fusion"] = "gated"
        self.net = MSFormerV2(**kwargs)
    def forward(self, x): return self.net(x)
