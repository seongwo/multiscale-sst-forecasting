import torch
import torch.nn as nn

from model.modules import SpatioTemporalLSTMCell, MIMBlock, MIMN


class MIM_Model(nn.Module):
    r"""
    MIM Model

    Memory In Memory: A Predictive Neural Network for Learning
    Higher-Order Non-Stationarity from Spatiotemporal Dynamics
    https://arxiv.org/abs/1811.07490
    """

    def __init__(self, configs, **kwargs):
        super(MIM_Model, self).__init__()
        T, C, H, W = configs["in_shape"]

        self.configs = configs
        ps = int(configs.get("patch_size", 1))

        self.frame_channel = ps * ps * C
        self.num_layers = int(configs["num_layers"])
        self.num_hidden = list(configs["num_hidden"])

        if self.num_layers < 2:
            raise ValueError("MIM requires num_layers >= 2")
        if len(self.num_hidden) != self.num_layers:
            raise ValueError("len(num_hidden) must equal num_layers")

        height = H // ps
        width = W // ps

        stlstm_layer = []
        stlstm_layer_diff = []

        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else self.num_hidden[i - 1]
            if i == 0:
                stlstm_layer.append(
                    SpatioTemporalLSTMCell(
                        in_channel,
                        self.num_hidden[i],
                        height,
                        width,
                        configs["filter_size"],
                        configs["stride"],
                        configs["layer_norm"],
                    )
                )
            else:
                stlstm_layer.append(
                    MIMBlock(
                        in_channel,
                        self.num_hidden[i],
                        height,
                        width,
                        configs["filter_size"],
                        configs["stride"],
                        configs["layer_norm"],
                    )
                )

        for i in range(self.num_layers - 1):
            stlstm_layer_diff.append(
                MIMN(
                    self.num_hidden[i],
                    self.num_hidden[i + 1],
                    height,
                    width,
                    configs["filter_size"],
                    configs["stride"],
                    configs["layer_norm"],
                )
            )

        self.stlstm_layer = nn.ModuleList(stlstm_layer)
        self.stlstm_layer_diff = nn.ModuleList(stlstm_layer_diff)

        self.conv_last = nn.Conv2d(
            self.num_hidden[-1],
            self.frame_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x, **kwargs):
        """
        x: (B, Tin, 1, H, W)
        return: (B, Tout, 1, H, W)
        """
        device = x.device
        x = x.contiguous()

        Tin = int(self.configs["input_len"])
        Tout = int(self.configs["pred_len"])

        if x.dim() != 5 or x.shape[1] != Tin or x.shape[2] != 1:
            raise ValueError(f"Expected x shape (B,{Tin},1,H,W), got {tuple(x.shape)}")

        ps = int(self.configs.get("patch_size", 1))
        if ps != 1:
            raise ValueError("MIM free-run expects patch_size=1")

        B, _, _, H, W = x.shape

        # ======================================================
        # IMPORTANT: reset persistent states in MIMBlock
        # (prevents batch-size mismatch in val/test)
        # ======================================================
        for m in self.stlstm_layer:
            if hasattr(m, "convlstm_c"):
                m.convlstm_c = None

        # initialize states
        h_t, c_t = [], []
        hidden_state_diff = []
        cell_state_diff = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                (B, self.num_hidden[i], H, W),
                device=device,
                dtype=x.dtype,
            )
            h_t.append(zeros)
            c_t.append(zeros)
            hidden_state_diff.append(None)
            cell_state_diff.append(None)

        st_memory = torch.zeros(
            (B, self.num_hidden[0], H, W),
            device=device,
            dtype=x.dtype,
        )

        next_frames = []
        x_gen = None

        total_steps = Tin + Tout - 1

        for t in range(total_steps):
            # free-run input
            if t < Tin:
                net = x[:, t]
            else:
                net = x_gen

            # layer 0 (ST-LSTM)
            preh = h_t[0]
            h_t[0], c_t[0], st_memory = self.stlstm_layer[0](
                net, h_t[0], c_t[0], st_memory
            )

            # higher layers (MIM)
            for i in range(1, self.num_layers):
                if t > 0:
                    if i == 1:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = \
                            self.stlstm_layer_diff[i - 1](
                                h_t[i - 1] - preh,
                                hidden_state_diff[i - 1],
                                cell_state_diff[i - 1],
                            )
                    else:
                        hidden_state_diff[i - 1], cell_state_diff[i - 1] = \
                            self.stlstm_layer_diff[i - 1](
                                hidden_state_diff[i - 2],
                                hidden_state_diff[i - 1],
                                cell_state_diff[i - 1],
                            )
                else:
                    # t == 0: initialize diff module internal states only
                    self.stlstm_layer_diff[i - 1](
                        torch.zeros_like(h_t[i - 1]), None, None
                    )
                    hidden_state_diff[i - 1] = None
                    cell_state_diff[i - 1] = None

                h_t[i], c_t[i], st_memory = self.stlstm_layer[i](
                    h_t[i - 1],
                    hidden_state_diff[i - 1],
                    h_t[i],
                    c_t[i],
                    st_memory,
                )

            x_gen = self.conv_last(h_t[-1])  # (B,1,H,W)
            next_frames.append(x_gen)

        outputs = torch.stack(next_frames, dim=1)                 # (B, Tin+Tout-1, 1, H, W)
        outputs = outputs[:, Tin-1:Tin-1+Tout].contiguous()       # (B, Tout, 1, H, W)
        return outputs
