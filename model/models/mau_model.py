import math
import torch
import torch.nn as nn

from model.modules import MAUCell


class MAU_Model(nn.Module):
    r"""MAU Model

    Implementation of `MAU: A Motion-Aware Unit for Video Prediction and Beyond
    <https://openreview.net/forum?id=qwtfY-3ibt7>`_.

    """

    def __init__(self, configs, **kwargs):
        super(MAU_Model, self).__init__()
        T, C, H, W = configs["in_shape"]
        
        self.configs = configs
        self.frame_channel = configs["patch_size"] * configs["patch_size"] * C
        self.num_layers = configs["num_layers"]
        self.num_hidden = configs["num_hidden"]
        self.tau = configs["tau"]
        self.cell_mode = configs["cell_mode"]
        self.states = ['recall', 'normal']
        if not self.configs["model_mode"] in self.states:
            raise AssertionError
        cell_list = []

        width = W // configs["patch_size"] // configs["sr_size"]
        height = H // configs["patch_size"] // configs["sr_size"]
        self.MSE_criterion = nn.MSELoss()

        for i in range(self.num_layers):
            in_channel = self.num_hidden[0] if i == 0 else self.num_hidden[i - 1]
            cell_list.append(
                MAUCell(in_channel, self.num_hidden[i], height, width, configs["filter_size"],
                        configs["stride"], self.tau, self.cell_mode)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Encoder
        n = int(math.log2(configs["sr_size"]))
        encoders = []
        encoder = nn.Sequential()
        encoder.add_module(name='encoder_t_conv{0}'.format(-1),
                           module=nn.Conv2d(in_channels=self.frame_channel,
                                            out_channels=self.num_hidden[0],
                                            stride=1,
                                            padding=0,
                                            kernel_size=1))
        encoder.add_module(name='relu_t_{0}'.format(-1),
                           module=nn.LeakyReLU(0.2))
        encoders.append(encoder)
        for i in range(n):
            encoder = nn.Sequential()
            encoder.add_module(name='encoder_t{0}'.format(i),
                               module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                out_channels=self.num_hidden[0],
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                kernel_size=(3, 3)
                                                ))
            encoder.add_module(name='encoder_t_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # Decoder
        decoders = []

        for i in range(n - 1):
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(i),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoder.add_module(name='c_decoder_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            decoders.append(decoder)

        if n > 0:
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(n - 1),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        self.srcnn = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        self.merge = nn.Conv2d(
            self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)
        self.conv_last_sr = nn.Conv2d(
            self.frame_channel * 2, self.frame_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x, **kwargs):
        device = x.device
        x = x.contiguous()

        Tin = int(self.configs["input_len"])
        Tout = int(self.configs["pred_len"])

        if x.dim() != 5 or x.shape[1] != Tin or x.shape[2] != 1:
            raise ValueError(f"Expected x shape (B,{Tin},1,H,W), got {tuple(x.shape)}")

        B, _, C, H, W = x.shape
        sr = int(self.configs.get("sr_size", 1))
        if sr < 1:
            raise ValueError("sr_size must be >= 1")

        # low-res spatial size after encoder downsample
        h_lr = H // sr
        w_lr = W // sr

        # init temporal states
        T_t = []
        T_pre = []
        S_pre = []
        for layer_idx in range(self.num_layers):
            tmp_t, tmp_s = [], []
            in_ch = self.num_hidden[layer_idx] if layer_idx == 0 else self.num_hidden[layer_idx - 1]
            for _ in range(self.tau):
                tmp_t.append(torch.zeros((B, in_ch, h_lr, w_lr), device=device, dtype=x.dtype))
                tmp_s.append(torch.zeros((B, in_ch, h_lr, w_lr), device=device, dtype=x.dtype))
            T_pre.append(tmp_t)
            S_pre.append(tmp_s)

        # initialize T_t for each layer
        for i in range(self.num_layers):
            T_t.append(torch.zeros((B, self.num_hidden[i], h_lr, w_lr), device=device, dtype=x.dtype))

        next_frames = []
        x_gen = None
        total_steps = Tin + Tout - 1

        for t in range(total_steps):
            if t < Tin:
                net = x[:, t]     # (B,1,H,W)
            else:
                net = x_gen       # (B,1,H,W)

            frames_feature = net
            frames_feature_encoded = []
            for enc in self.encoders:
                frames_feature = enc(frames_feature)
                frames_feature_encoded.append(frames_feature)

            S_t = frames_feature  # (B, hidden0, h_lr, w_lr) after encoder

            for i in range(self.num_layers):
                t_att = torch.stack(T_pre[i][-self.tau:], dim=0)
                s_att = torch.stack(S_pre[i][-self.tau:], dim=0)
                S_pre[i].append(S_t)

                T_t[i], S_t = self.cell_list[i](T_t[i], S_t, t_att, s_att)
                T_pre[i].append(T_t[i])

            out = S_t
            for k, dec in enumerate(self.decoders):
                out = dec(out)
                if self.configs.get("model_mode", "normal") == "recall":
                    out = out + frames_feature_encoded[-2 - k]

            x_gen = self.srcnn(out)            # (B,1,H,W)
            next_frames.append(x_gen)

        outputs = torch.stack(next_frames, dim=1)                 # (B, Tin+Tout-1, 1, H, W)
        outputs = outputs[:, Tin-1:Tin-1+Tout].contiguous()       # (B, Tout, 1, H, W)
        return outputs

