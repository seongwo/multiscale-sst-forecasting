import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.vision_transformer import VisionTransformer


class TorchVisionViTGRUForecast(nn.Module):
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
    ):
        super().__init__()

        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.tin = tin
        self.tout = tout
        self.embed_dim = embed_dim

        self.gh = image_size // patch_size
        self.gw = image_size // patch_size
        self.num_patches = self.gh * self.gw

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
            num_classes=1,
        )

        sig = inspect.signature(VisionTransformer.__init__)
        if "in_channels" in sig.parameters:
            vit_kwargs["in_channels"] = in_chans
        elif "in_chans" in sig.parameters:
            vit_kwargs["in_chans"] = in_chans

        self.vit = VisionTransformer(**vit_kwargs)

        if hasattr(self.vit, "conv_proj") and isinstance(self.vit.conv_proj, nn.Conv2d):
            if self.vit.conv_proj.in_channels != in_chans:
                self.vit.conv_proj = nn.Conv2d(
                    in_chans,
                    embed_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                    bias=True,
                )

        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.temporal_encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            batch_first=True,
        )

        self.temporal_decoder_cell = nn.GRUCell(
            input_size=embed_dim,
            hidden_size=embed_dim,
        )

        self.latent_to_tokens = nn.Linear(embed_dim, self.num_patches * embed_dim)
        self.patch_skip = nn.Linear(embed_dim, embed_dim)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        self.token_refine = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.token_to_patch = nn.Linear(
            embed_dim, in_chans * patch_size * patch_size
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        nn.init.xavier_uniform_(self.latent_to_tokens.weight)
        nn.init.zeros_(self.latent_to_tokens.bias)

        nn.init.xavier_uniform_(self.patch_skip.weight)
        nn.init.zeros_(self.patch_skip.bias)

        nn.init.xavier_uniform_(self.token_to_patch.weight)
        nn.init.zeros_(self.token_to_patch.bias)

        for m in self.token_refine:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _resize_input_if_needed(self, x: torch.Tensor):
        h, w = x.shape[-2], x.shape[-1]
        if h == self.image_size and w == self.image_size:
            return x, (h, w), False

        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return x, (h, w), True

    def _forward_vit_tokens(self, img: torch.Tensor):
        if not hasattr(self.vit, "_process_input"):
            raise RuntimeError(
                "torchvision VisionTransformer is missing _process_input in this version."
            )

        x = self.vit._process_input(img)

        cls_token = self.vit.class_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.encoder(x)

        cls_out = x[:, 0, :]
        patch_out = x[:, 1:, :]
        return cls_out, patch_out

    def _decode_tokens_to_frames(self, patch_tokens: torch.Tensor):
        b, t, n, d = patch_tokens.shape
        ps = self.patch_size
        gh, gw = self.gh, self.gw
        c = self.in_chans

        patch_vec = self.token_to_patch(patch_tokens)

        patch_vec = patch_vec.reshape(b, t, n, c, ps, ps)
        patch_vec = patch_vec.reshape(b, t, gh, gw, c, ps, ps)
        patch_vec = patch_vec.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        frames = patch_vec.reshape(b, t, c, gh * ps, gw * ps)

        return frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, tin, c, h, w = x.shape
        assert tin == self.tin, f"Expected Tin={self.tin}, got {tin}"
        assert c == self.in_chans, f"Expected C={self.in_chans}, got {c}"

        xt = x.reshape(b * tin, c, h, w)
        xt, orig_hw, resized = self._resize_input_if_needed(xt)

        cls_tokens, patch_tokens = self._forward_vit_tokens(xt)

        cls_tokens = cls_tokens.reshape(b, tin, self.embed_dim)
        patch_tokens = patch_tokens.reshape(b, tin, self.num_patches, self.embed_dim)

        last_patch_tokens = patch_tokens[:, -1]

        _, h_n = self.temporal_encoder(cls_tokens)
        h_state = h_n[-1]

        future_latents = []
        inp = h_state
        for _ in range(self.tout):
            h_state = self.temporal_decoder_cell(inp, h_state)
            future_latents.append(h_state)
            inp = h_state

        future_latents = torch.stack(future_latents, dim=1)

        future_tokens = self.latent_to_tokens(future_latents)
        future_tokens = future_tokens.view(b, self.tout, self.num_patches, self.embed_dim)

        last_patch_skip = self.patch_skip(last_patch_tokens).unsqueeze(1)
        future_tokens = future_tokens + last_patch_skip + self.decoder_pos_embed.unsqueeze(1)

        future_tokens = self.token_refine(future_tokens)

        yhat = self._decode_tokens_to_frames(future_tokens)

        if resized:
            yhat = yhat.reshape(b * self.tout, c, self.image_size, self.image_size)
            yhat = F.interpolate(
                yhat,
                size=orig_hw,
                mode="bilinear",
                align_corners=False,
            )
            yhat = yhat.reshape(b, self.tout, c, orig_hw[0], orig_hw[1])

        return yhat