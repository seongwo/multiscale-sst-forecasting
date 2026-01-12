import torch
import torch.nn as nn

from model.modules.convlstm_modules import ConvLSTMCell


class ConvLSTM_Model(nn.Module):

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super().__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden

        height = H // configs.patch_size
        width = W // configs.patch_size

        cell_list = []
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    in_channel,
                    num_hidden[i],
                    height,
                    width,
                    configs.filter_size,
                    configs.stride,
                    configs.layer_norm,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(
            num_hidden[-1],
            self.frame_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, frames_tensor):
        
        device = frames_tensor.device
        frames = frames_tensor.contiguous()

        B, _, _, H, W = frames.shape

        h_t = []
        c_t = []
        for i in range(self.num_layers):
            h_t.append(torch.zeros(B, self.num_hidden[i], H, W, device=device))
            c_t.append(torch.zeros(B, self.num_hidden[i], H, W, device=device))

        outputs = []

        total_steps = self.configs.pre_seq_length + self.configs.aft_seq_length - 1

        for t in range(total_steps):
            if t < self.configs.pre_seq_length:
                net = frames[:, t]
            else:
                net = x_gen

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[-1])
            outputs.append(x_gen)

        # [B, T_out, H, W, C]
        outputs = torch.stack(outputs, dim=1).permute(0, 1, 3, 4, 2).contiguous()
        return outputs
