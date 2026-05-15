import torch
import torch.nn as nn
from timm.models.swin_transformer import (
    SwinTransformerBlock,
    window_reverse,
    PatchEmbed,
    PatchMerging,
    window_partition,
)
from timm.layers import to_2tuple


def _to_tokens(x: torch.Tensor):
    """
    x를 (B, L, C) 토큰 형태로 통일해서 반환.
    timm PatchEmbed 출력이 (B,L,C) / (B,H,W,C) / (B,C,H,W) 어떤 경우든 처리.
    return: tokens (B,L,C), (H,W)
    """
    if x.ndim == 3:
        B, L, C = x.shape
        return x, (None, None)

    if x.ndim != 4:
        raise ValueError(f"Unsupported tensor ndim={x.ndim}")

    B = x.shape[0]

    # NHWC
    if x.shape[-1] <= 4096 and x.shape[-1] != x.shape[1]:
        H, W, C = x.shape[1], x.shape[2], x.shape[3]
        return x.view(B, H * W, C), (H, W)

    # NCHW
    C, H, W = x.shape[1], x.shape[2], x.shape[3]
    x = x.permute(0, 2, 3, 1).contiguous()  # -> NHWC
    return x.view(B, H * W, C), (H, W)


def _tokens_to_nhwc(tokens: torch.Tensor, hw):
    """
    tokens: (B,L,C)
    hw: (H,W)
    return: (B,H,W,C)
    """
    H, W = hw
    if H is None or W is None:
        raise ValueError("H/W is None, cannot reshape tokens to NHWC.")
    B, L, C = tokens.shape
    if L != H * W:
        raise ValueError(f"Token length mismatch: L={L}, H*W={H*W}")
    return tokens.view(B, H, W, C)


def _nhwc_to_tokens(x: torch.Tensor):
    """
    x: (B,H,W,C) -> (B,L,C), (H,W)
    """
    if x.ndim != 4:
        raise ValueError(f"Expected NHWC 4D, got {x.ndim}D")
    B, H, W, C = x.shape
    return x.view(B, H * W, C), (H, W)


class SwinLSTMCell(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size,
        depth,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        flag=None,
    ):
        super().__init__()
        self.STBs = nn.ModuleList(
            STB(
                i,
                dim=dim,
                input_resolution=input_resolution,
                depth=depth,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                norm_layer=norm_layer,
                flag=flag,
            )
            for i in range(depth)
        )

    def forward(self, xt, hidden_states):
        if hidden_states is None:
            B, L, C = xt.shape
            hx = torch.zeros(B, L, C, device=xt.device)
            cx = torch.zeros(B, L, C, device=xt.device)
        else:
            hx, cx = hidden_states

        outputs = []
        for index, layer in enumerate(self.STBs):
            if index == 0:
                x = layer(xt, hx)
                outputs.append(x)
            else:
                if index % 2 == 0:
                    x = layer(outputs[-1], xt)
                    outputs.append(x)
                else:
                    x = layer(outputs[-1], None)
                    outputs.append(x)

        o_t = outputs[-1]
        Ft = torch.sigmoid(o_t)
        cell = torch.tanh(o_t)

        Ct = Ft * (cx + cell)
        Ht = Ft * torch.tanh(Ct)

        return Ht, (Ht, Ct)


class STB(SwinTransformerBlock):
    def __init__(
        self,
        index,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        flag=None,
    ):
        if flag == 0:
            drop_path = drop_path[depth - index - 1]
        elif flag == 1:
            drop_path = drop_path[index]
        else:
            drop_path = drop_path

        super().__init__(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0 if (index % 2 == 0) else window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
        )

        self.red = nn.Linear(2 * dim, dim)

    def forward(self, x, hx=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size: L={L}, H*W={H*W}"

        shortcut = x
        x = self.norm1(x)

        if hx is not None:
            hx = self.norm1(hx)
            x = torch.cat((x, hx), dim=-1)
            x = self.red(x)

        x = x.view(B, H, W, C)

        win = self.window_size if isinstance(self.window_size, tuple) else to_2tuple(self.window_size)
        shift = self.shift_size if isinstance(self.shift_size, tuple) else to_2tuple(self.shift_size)
        has_shift = (shift[0] != 0) or (shift[1] != 0)

        if has_shift:
            shifted_x = torch.roll(x, shifts=(-shift[0], -shift[1]), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, win)  # (nW*B, Wh, Ww, C)
        x_windows = x_windows.view(-1, win[0] * win[1], C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, win[0], win[1], C)

        shifted_x = window_reverse(attn_windows, win, H, W)

        if has_shift:
            x = torch.roll(shifted_x, shifts=shift, dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path1(x)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x


class PatchInflated(nn.Module):
    def __init__(self, in_chans, embed_dim, input_resolution, stride=2, padding=1, output_padding=1):
        super().__init__()
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        output_padding = to_2tuple(output_padding)
        self.input_resolution = input_resolution

        self.Conv = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=in_chans,
            kernel_size=(3, 3),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size: L={L}, H*W={H*W}"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.Conv(x)
        return x


class PatchExpanding(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size: L={L}, H*W={H*W}"

        x = x.view(B, H, W, C)
        x = x.reshape(B, H, W, 2, 2, C // 4)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H * 2, W * 2, C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x


class UpSample(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        depths_upsample,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        flag=0,
    ):
        super().__init__()

        self.img_size = img_size
        self.num_layers = len(depths_upsample)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )
        self.patches_resolution = self.patch_embed.grid_size
        self.Unembed = PatchInflated(in_chans=in_chans, embed_dim=embed_dim, input_resolution=patches_resolution)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_upsample))]

        self.layers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.resolutions = []

        for i_layer in range(self.num_layers):
            res_h = self.patches_resolution[0] // (2 ** (self.num_layers - i_layer))
            res_w = self.patches_resolution[1] // (2 ** (self.num_layers - i_layer))
            self.resolutions.append((res_h, res_w))

            dimension = int(embed_dim * 2 ** (self.num_layers - i_layer))
            upsample = PatchExpanding(input_resolution=(res_h, res_w), dim=dimension)

            layer = SwinLSTMCell(
                dim=dimension,
                input_resolution=(res_h, res_w),
                depth=depths_upsample[(self.num_layers - 1 - i_layer)],
                num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths_upsample[: (self.num_layers - 1 - i_layer)]) : sum(
                        depths_upsample[: (self.num_layers - 1 - i_layer) + 1]
                    )
                ],
                norm_layer=norm_layer,
                flag=flag,
            )

            self.layers.append(layer)
            self.upsample.append(upsample)

    def forward(self, x, y):
        hidden_states_up = []
        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, y[index])
            x = self.upsample[index](x)
            hidden_states_up.append(hidden_state)

        x = self.Unembed(x)
        return hidden_states_up, x


class DownSample(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        depths_downsample,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        flag=1,
    ):
        super().__init__()

        self.num_layers = len(depths_downsample)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )
        self.patches_resolution = self.patch_embed.grid_size  # (Hp, Wp)

        self.resolutions = []
        for i_layer in range(self.num_layers):
            res_h = self.patches_resolution[0] // (2 ** i_layer)
            res_w = self.patches_resolution[1] // (2 ** i_layer)
            self.resolutions.append((res_h, res_w))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_downsample))]

        self.layers = nn.ModuleList()
        self.downsample = nn.ModuleList()

        for i_layer in range(self.num_layers):
            downsample = PatchMerging(
                dim=int(embed_dim * 2 ** i_layer),
                norm_layer=norm_layer,
            )

            layer = SwinLSTMCell(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=self.resolutions[i_layer],
                depth=depths_downsample[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths_downsample[:i_layer]) : sum(depths_downsample[: i_layer + 1])],
                norm_layer=norm_layer,
                flag=flag,
            )

            self.layers.append(layer)
            self.downsample.append(downsample)

    def forward(self, x, y):
        x = self.patch_embed(x)
        x, hw = _to_tokens(x)

        if hw[0] is None or hw[1] is None:
            hw = self.resolutions[0]

        hidden_states_down = []

        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, y[index])
            hidden_states_down.append(hidden_state)

            res_h, res_w = self.resolutions[index]
            x4 = _tokens_to_nhwc(x, (res_h, res_w))
            x4 = self.downsample[index](x4)  # (B, H/2, W/2, C')
            x, _ = _nhwc_to_tokens(x4)

        return hidden_states_down, x


class STconvert(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        depths,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        flag=2,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        self.patches_resolution = self.patch_embed.grid_size

        self.patch_inflated = PatchInflated(in_chans=in_chans, embed_dim=embed_dim, input_resolution=self.patches_resolution)

        self.layer = SwinLSTMCell(
            dim=embed_dim,
            input_resolution=(self.patches_resolution[0], self.patches_resolution[1]),
            depth=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
            norm_layer=norm_layer,
            flag=flag,
        )

    def forward(self, x, h=None):
        x = self.patch_embed(x)
        x, hw = _to_tokens(x)
        if hw[0] is None or hw[1] is None:
            hw = self.patches_resolution

        x, hidden_state = self.layer(x, h)
        x = self.patch_inflated(x)
        return x, hidden_state
