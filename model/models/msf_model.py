# model/models/msformer_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedSTBlock(nn.Module):
    """
    Factorized Space-Time Transformer block:
      1) spatial self-attn within each frame (tokens length N)
      2) temporal self-attn for each token index across time (length T)
      3) MLP
    x: (B, T, N, D)
    """
    def __init__(self, d_model: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.0, attn_dropout: float = 0.0):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,N,D)
        B, T, N, D = x.shape

        # 1) spatial attn per frame: batch = B*T, seq = N
        xs = self.ln_s(x).reshape(B * T, N, D)
        ys, _ = self.attn_s(xs, xs, xs, need_weights=False)
        x = x + ys.reshape(B, T, N, D)

        # 2) temporal attn per token index: batch = B*N, seq = T
        xt = self.ln_t(x).permute(0, 2, 1, 3).reshape(B * N, T, D)
        yt, _ = self.attn_t(xt, xt, xt, need_weights=False)
        x = x + yt.reshape(B, N, T, D).permute(0, 2, 1, 3)

        # 3) MLP
        x = x + self.mlp(self.ln_m(x))
        return x


class MSFormer(nn.Module):
    """
    Multi-Scale Spatial Tokenization + Factorized ST Transformer.
    - scales: (2,4,8,16,32) each creates tokens via Conv2d(kernel=stride=s)
    - tokens are concatenated along N dimension
    - optional global scale gating (fusion="gated")
    - decode ONLY base_scale tokens (default=min(scales)) to (B,Tout,1,H,W)

    I/O:
      input : (B, Tin, 1, H, W)
      output: (B, Tout, 1, H, W)
    """
    def __init__(
        self,
        image_size: int = 32,
        in_chans: int = 1,
        tin: int = 14,
        tout: int = 7,
        d_model: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        spatial_scales=(2, 4, 8, 16, 32),
        base_scale: int = None,
        fusion: str = "concat",   # "concat" or "gated"
        use_pad: bool = True,
        use_pos_emb: bool = False,  # scale-specific spatial pos emb (optional)
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.in_chans = int(in_chans)
        self.tin = int(tin)
        self.tout = int(tout)
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        self.attn_dropout = float(attn_dropout)

        self.scales = tuple(int(s) for s in spatial_scales)
        assert len(self.scales) >= 1
        self.base_scale = int(base_scale) if base_scale is not None else min(self.scales)
        assert self.base_scale in self.scales, "base_scale must be one of spatial_scales"

        self.fusion = fusion
        assert self.fusion in ["concat", "gated"]
        self.use_pad = bool(use_pad)
        self.use_pos_emb = bool(use_pos_emb)

        # scale-specific tokenizers
        self.tokenizers = nn.ModuleDict()
        for s in self.scales:
            self.tokenizers[str(s)] = nn.Conv2d(
                self.in_chans, self.d_model, kernel_size=s, stride=s, bias=True
            )

        # embeddings
        self.time_emb = nn.Embedding(self.tin, self.d_model)
        self.scale_emb = nn.Embedding(len(self.scales), self.d_model)

        # optional: scale-specific spatial position embeddings (N_s differs per scale)
        self.pos_emb = nn.ModuleDict()
        if self.use_pos_emb:
            for s in self.scales:
                gs = self.image_size // s  # assumes divisible after pad/resize
                Ns = gs * gs
                self.pos_emb[str(s)] = nn.Embedding(Ns, self.d_model)

        # gated fusion (global scale weights)
        if self.fusion == "gated":
            self.gate = nn.Sequential(
                nn.Linear(self.d_model * len(self.scales), self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, len(self.scales)),
            )
        else:
            self.gate = None

        # transformer
        self.blocks = nn.ModuleList([
            FactorizedSTBlock(self.d_model, self.num_heads, self.mlp_ratio, self.dropout, self.attn_dropout)
            for _ in range(self.depth)
        ])
        self.ln_f = nn.LayerNorm(self.d_model)

        # forecast head for BASE tokens only: (B, N_base, D) -> (B, N_base, Tout*D)
        self.future_head = nn.Linear(self.d_model, self.tout * self.d_model)

        # decode base tokens to pixels
        ps = self.base_scale
        self.token_to_patch = nn.Linear(self.d_model, self.in_chans * ps * ps)

    def _pad_to_multiple(self, x: torch.Tensor, multiple: int):
        # x: (B*T,C,H,W)
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (pad_h, pad_w)

    def _tokenize_one_scale(self, x_btchw: torch.Tensor, s: int):
        # x_btchw: (B*T,C,H,W)
        tok = self.tokenizers[str(s)](x_btchw)          # (B*T, D, gh, gw)
        gh, gw = tok.shape[-2], tok.shape[-1]
        tok = tok.flatten(2).transpose(1, 2)            # (B*T, N_s, D)
        return tok, gh, gw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,Tin,1,H,W)
        B, Tin, C, H, W = x.shape
        assert C == self.in_chans
        assert Tin == self.tin, f"Tin mismatch: got {Tin}, expected {self.tin}"

        # pad to multiple of lcm-ish (just use max scale)
        xt = x.reshape(B * Tin, C, H, W)
        if self.use_pad:
            xt, (pad_h, pad_w) = self._pad_to_multiple(xt, max(self.scales))
        Hp, Wp = xt.shape[-2], xt.shape[-1]

        # tokenize per scale
        scale_tokens = []
        scale_meta = []   # (s, Ns, gh, gw)
        for sid, s in enumerate(self.scales):
            tok_s, gh, gw = self._tokenize_one_scale(xt, s)  # (B*Tin, Ns, D)
            Ns = gh * gw
            tok_s = tok_s.reshape(B, Tin, Ns, self.d_model)

            # add time + scale emb
            t_idx = torch.arange(Tin, device=x.device)
            tok_s = tok_s + self.time_emb(t_idx).view(1, Tin, 1, self.d_model)
            tok_s = tok_s + self.scale_emb(torch.tensor(sid, device=x.device)).view(1, 1, 1, self.d_model)

            # optional pos emb per scale
            if self.use_pos_emb:
                p_idx = torch.arange(Ns, device=x.device)
                tok_s = tok_s + self.pos_emb[str(s)](p_idx).view(1, 1, Ns, self.d_model)

            scale_tokens.append(tok_s)
            scale_meta.append((s, Ns, gh, gw))

        # optional global gated scaling BEFORE concat
        if self.fusion == "gated":
            reps = []
            for tok_s in scale_tokens:
                reps.append(tok_s.mean(dim=(1, 2)) )   # (B, D)
            cat = torch.cat(reps, dim=-1)              # (B, D*S)
            w = torch.softmax(self.gate(cat), dim=-1)  # (B, S)
            for i in range(len(scale_tokens)):
                scale_tokens[i] = scale_tokens[i] * w[:, i].view(B, 1, 1, 1)

        # concat tokens along N: (B,T, sum(Ns), D)
        z = torch.cat(scale_tokens, dim=2)
        N_total = z.shape[2]

        # factorized ST transformer
        h = z
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)

        # pick BASE tokens slice (assumes scales order and base token block exists)
        # We'll take the segment corresponding to base_scale in the concatenation order.
        base_start = 0
        base_N = None
        for (s, Ns, _, _) in scale_meta:
            if s == self.base_scale:
                base_N = Ns
                break
            base_start += Ns
        assert base_N is not None
        base_tok = h[:, :, base_start:base_start + base_N, :]     # (B,T,Nb,D)

        # pool over time (mean) -> per base token rep
        base_rep = base_tok.mean(dim=1)                           # (B,Nb,D)

        # one-shot future tokens for base tokens
        fut = self.future_head(base_rep).reshape(B, base_N, self.tout, self.d_model)  # (B,Nb,Tout,D)
        fut = fut.permute(0, 2, 1, 3).contiguous()                                   # (B,Tout,Nb,D)

        # decode base tokens back to frames using base_scale patch size
        ps = self.base_scale
        # base grid size
        ghb = Hp // ps
        gwb = Wp // ps
        assert ghb * gwb == base_N, "base_N mismatch with padded spatial size"

        patch_vec = self.token_to_patch(fut)  # (B,Tout,Nb,C*ps*ps)
        patch_vec = patch_vec.reshape(B, self.tout, ghb, gwb, self.in_chans, ps, ps)
        patch_vec = patch_vec.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        y = patch_vec.reshape(B, self.tout, self.in_chans, ghb * ps, gwb * ps)

        # remove pad
        if self.use_pad and (pad_h > 0 or pad_w > 0):
            y = y[..., :H, :W]
        return y


# short names (requested)
class MSF(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs["fusion"] = "concat"
        self.net = MSFormer(**kwargs)
    def forward(self, x): return self.net(x)


class MSFG(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs["fusion"] = "gated"
        self.net = MSFormer(**kwargs)
    def forward(self, x): return self.net(x)
