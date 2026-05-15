import torch
import torch.nn as nn

from model.modules import SpatioTemporalLSTMCell


class PredRNN_Model(nn.Module):
    r"""PredRNN

    Implementation of `PredRNN: A Recurrent Neural Network for Spatiotemporal
    Predictive Learning <https://dl.acm.org/doi/abs/10.5555/3294771.3294855>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(PredRNN_Model, self).__init__()
        T, C, H, W = configs["in_shape"]

        self.configs = configs
        self.frame_channel = configs["patch_size"] * configs["patch_size"] * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        height = H // configs["patch_size"]
        width = W // configs["patch_size"]
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], height, width,
                                       configs["filter_size"], configs["stride"], configs["layer_norm"]))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        
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

        ps = int(self.configs["patch_size"])
        if ps != 1:
            raise ValueError("Free-run + current dataset expects patch_size=1 (no patchify).")

        B, _, C, H, W = x.shape

        next_frames = []
        h_t, c_t = [], []

        for i in range(self.num_layers):
            zeros = torch.zeros((B, self.num_hidden[i], H, W), device=device, dtype=x.dtype)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros((B, self.num_hidden[0], H, W), device=device, dtype=x.dtype)

        x_gen = None
        total_steps = Tin + Tout - 1

        for t in range(total_steps):
            if t < Tin:
                net = x[:, t]            
            else:
                net = x_gen             

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i-1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[-1])   # (B,1,H,W)
            next_frames.append(x_gen)

        outputs = torch.stack(next_frames, dim=1)                # (B, Tin+Tout-1, 1, H, W)
        outputs = outputs[:, Tin-1:Tin-1+Tout].contiguous()      # (B, Tout, 1, H, W)
        return outputs
