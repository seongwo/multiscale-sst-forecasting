import torch
import torch.nn as nn

from model.modules import DownSample, UpSample, STconvert


class SwinLSTM_D_Model(nn.Module):
    def __init__(self, depths_downsample, depths_upsample, num_heads, configs, **kwargs):
        super().__init__()
        T, C, H, W = configs["in_shape"]
        assert H == W, "Only support H = W for image input"

        self.configs = configs
        self.depths_downsample = depths_downsample
        self.depths_upsample = depths_upsample

        self.Downsample = DownSample(
            img_size=H,
            patch_size=configs["patch_size"],
            in_chans=C,
            embed_dim=configs["embed_dim"],
            depths_downsample=depths_downsample,
            num_heads=num_heads,
            window_size=configs["window_size"],
        )

        self.Upsample = UpSample(
            img_size=H,
            patch_size=configs["patch_size"],
            in_chans=C,
            embed_dim=configs["embed_dim"],
            depths_upsample=depths_upsample,
            num_heads=num_heads,
            window_size=configs["window_size"],
        )

    def forward(self, frames_tensor, **kwargs):
        """
        Input : frames_tensor (B, T, C, H, W)  (T can be Tin or Tin+Tout depending on dataset)
        Output: preds        (B, Tout, C, H, W)

        Fix:
        - Warm-up includes Tin-1 (i.e., range(Tin))
        - last_frame is the last observed frame (Tin-1), NOT frames[:, -1] to avoid leakage
        - Pure autoregressive rollout 유지
        """
        Tin = int(self.configs["input_len"])
        Tout = int(self.configs["pred_len"])

        frames = frames_tensor.contiguous()
        input_frames = frames[:, :Tin]  # ✅ prevent future leakage

        states_down = [None] * len(self.depths_downsample)
        states_up = [None] * len(self.depths_upsample)

        for i in range(Tin):
            states_down, x = self.Downsample(input_frames[:, i], states_down)
            states_up, _ = self.Upsample(x, states_up)

        preds = []
        last_frame = input_frames[:, -1]  


        for _ in range(Tout):
            states_down, x = self.Downsample(last_frame, states_down)
            states_up, out = self.Upsample(x, states_up)
            preds.append(out)
            last_frame = out

        return torch.stack(preds, dim=1).contiguous()


class SwinLSTM_B_Model(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()
        T, C, H, W = configs["in_shape"]
        assert H == W, "Only support H = W for image input"
        self.configs = configs

        self.ST = STconvert(
            img_size=H,
            patch_size=configs["patch_size"],
            in_chans=C,
            embed_dim=configs["embed_dim"],
            depths=configs["depths"],
            num_heads=configs["num_heads"],
            window_size=configs["window_size"],
        )

    def forward(self, frames_tensor):
        """
        Input : frames_tensor (B, T, C, H, W)  (T can be Tin or Tin+Tout depending on dataset)
        Output: preds        (B, Tout, C, H, W)

        Fix:
        - Warm-up includes Tin-1 (i.e., range(Tin))
        - last_frame is frames[:, Tin-1] (via input_frames), NOT frames[:, -1] to avoid leakage
        - Pure autoregressive rollout 유지
        """
        Tin = int(self.configs["input_len"])
        Tout = int(self.configs["pred_len"])

        frames = frames_tensor.contiguous()
        input_frames = frames[:, :Tin]  

        states = None

        for i in range(Tin):
            _, states = self.ST(input_frames[:, i], states)

        preds = []
        last_frame = input_frames[:, -1]  

        for _ in range(Tout):
            out, states = self.ST(last_frame, states)
            preds.append(out)
            last_frame = out

        return torch.stack(preds, dim=1).contiguous()
