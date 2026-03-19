import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import VisionTransformer


class TorchVisionViT(nn.Module):
    """
    ViT + One-shot Head only (OpenSTL-style I/O)

    Input : x (B, Tin, C, H, W)
    Output: yhat (B, Tout, C, H, W)

    - Spatial encoder: torchvision VisionTransformer -> patch tokens (no CLS in output)
    - Temporal handling: simple aggregation over time (mean pooling)
    - Forecast head: one-shot Linear (D -> Tout*D) per patch token
    - Decoder: token -> pixel patch, then fold back to frames
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 1,
        tin: int = 14,
        tout: int = 7,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        freeze_vit: bool = False,
        # optional: how to aggregate temporal tokens ("mean" or "last")
        time_pool: str = "mean",
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.tin = tin
        self.tout = tout
        self.embed_dim = embed_dim
        self.time_pool = time_pool

        mlp_dim = int(embed_dim * mlp_ratio)

        vit_kwargs = dict(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=depth,
            num_heads=num_heads,
            hidden_dim=embed_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attn_dropout,
            num_classes=1,  # dummy
        )

        sig = inspect.signature(VisionTransformer.__init__)
        if "in_channels" in sig.parameters:
            vit_kwargs["in_channels"] = in_chans
        elif "in_chans" in sig.parameters:
            vit_kwargs["in_chans"] = in_chans

        self.vit = VisionTransformer(**vit_kwargs)

        # Ensure conv_proj matches in_chans (some builds default to 3)
        if hasattr(self.vit, "conv_proj") and isinstance(self.vit.conv_proj, nn.Conv2d):
            if self.vit.conv_proj.in_channels != in_chans:
                self.vit.conv_proj = nn.Conv2d(
                    in_chans,
                    embed_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                    bias=True,
                )

        # One-shot forecast head: per patch token (D -> Tout*D)
        self.future_head = nn.Linear(embed_dim, tout * embed_dim)

        # Decoder: token -> patch pixels
        self.token_to_patch = nn.Linear(embed_dim, in_chans * patch_size * patch_size)

        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

    def _pad_to_multiple(self, x: torch.Tensor, multiple: int):
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (pad_h, pad_w)

    def _forward_vit_patch_tokens(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B, C, H, W)
        return: patch tokens (B, N, D) (CLS dropped)
        """
        if not hasattr(self.vit, "_process_input"):
            raise RuntimeError("VisionTransformer API mismatch: missing _process_input().")

        x = self.vit._process_input(img)  # (B, N, D)
        cls = self.vit.class_token.expand(img.shape[0], -1, -1)  # (B, 1, D)
        x = torch.cat((cls, x), dim=1)  # (B, 1+N, D)
        x = self.vit.encoder(x)         # (B, 1+N, D)
        x = x[:, 1:, :]                 # (B, N, D)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Tin, C, H, W)
        b, tin, c, h, w = x.shape
        assert tin == self.tin, f"Tin mismatch: got {tin}, expected {self.tin}"

        xt = x.reshape(b * tin, c, h, w)
        xt, (pad_h, pad_w) = self._pad_to_multiple(xt, self.patch_size)

        hp, wp = xt.shape[-2], xt.shape[-1]
        gh, gw = hp // self.patch_size, wp // self.patch_size
        n = gh * gw

        # ViT patch tokens per frame
        patch_tokens = self._forward_vit_patch_tokens(xt)  # (B*Tin, N, D)
        d = patch_tokens.shape[-1]
        tokens = patch_tokens.reshape(b, tin, n, d)        # (B, Tin, N, D)

        # Temporal aggregation (no GRU)
        if self.time_pool == "last":
            fused = tokens[:, -1, :, :]        # (B, N, D)
        else:
            fused = tokens.mean(dim=1)         # (B, N, D)

        # One-shot future tokens per patch: (B, N, Tout, D)
        fut = self.future_head(fused).reshape(b, n, self.tout, d)

        # Reorder to (B, Tout, N, D)
        pred_tokens = fut.permute(0, 2, 1, 3).contiguous()

        # Token -> patch pixels
        patch_vec = self.token_to_patch(pred_tokens)  # (B, Tout, N, C*ps*ps)

        # Fold patches back
        ps = self.patch_size
        patch_vec = patch_vec.reshape(b, self.tout, n, c, ps, ps)
        patch_vec = patch_vec.permute(0, 1, 3, 2, 4, 5)  # (B, Tout, C, N, ps, ps)
        patch_vec = patch_vec.reshape(b, self.tout, c, gh, gw, ps, ps)
        patch_vec = patch_vec.permute(0, 1, 2, 3, 5, 4, 6).contiguous()
        yhat = patch_vec.reshape(b, self.tout, c, hp, wp)

        # Remove padding
        if pad_h > 0:
            yhat = yhat[..., :h, :]
        if pad_w > 0:
            yhat = yhat[..., :w]

        return yhat
