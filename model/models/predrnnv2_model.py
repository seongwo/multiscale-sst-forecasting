import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import SpatioTemporalLSTMCellv2


class PredRNNv2_Model(nn.Module):
    r"""PredRNNv2

    Free-running autoregressive version for:
        x: (B, Tin, 1, H, W)
        y: (B, Tout, 1, H, W)

    If return_aux_loss=True:
        return outputs, aux_loss
    else:
        return outputs
    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(PredRNNv2_Model, self).__init__()

        T, C, H, W = configs["in_shape"]

        self.configs = configs
        self.num_layers = num_layers
        self.num_hidden = num_hidden

        patch_size = int(configs["patch_size"])
        self.frame_channel = patch_size * patch_size * C

        height = H // patch_size
        width = W // patch_size

        cell_list = []
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell = SpatioTemporalLSTMCellv2(
                in_channel,
                num_hidden[i],
                height,
                width,
                configs["filter_size"],
                configs["stride"],
                configs["layer_norm"],
            )
            cell_list.append(cell)

        self.cell_list = nn.ModuleList(cell_list)

        self.conv_last = nn.Conv2d(
            num_hidden[num_layers - 1],
            self.frame_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(
            adapter_num_hidden,
            adapter_num_hidden,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x, **kwargs):
        """
        Args:
            x: (B, Tin, 1, H, W)

        Returns:
            outputs: (B, Tout, 1, H, W)

            If return_aux_loss=True:
                outputs, aux_loss
        """
        device = x.device
        x = x.contiguous()

        Tin = int(self.configs["input_len"])
        Tout = int(self.configs["pred_len"])
        patch_size = int(self.configs["patch_size"])

        if x.dim() != 5 or x.shape[1] != Tin or x.shape[2] != 1:
            raise ValueError(f"Expected x shape (B,{Tin},1,H,W), got {tuple(x.shape)}")

        if patch_size != 1:
            raise ValueError("Free-run + current dataset expects patch_size=1 (no patchify).")

        B, _, C, H, W = x.shape

        next_frames = []
        h_t, c_t = [], []
        delta_c_list, delta_m_list = [], []
        decouple_loss_list = []

        for i in range(self.num_layers):
            zeros = torch.zeros((B, self.num_hidden[i], H, W), device=device, dtype=x.dtype)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros((B, self.num_hidden[0], H, W), device=device, dtype=x.dtype)

        x_gen = None
        total_steps = Tin + Tout - 1

        for t in range(total_steps):
            if t < Tin:
                net = x[:, t]
            else:
                net = x_gen

            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](
                net, h_t[0], c_t[0], memory
            )

            delta_c_list[0] = F.normalize(
                self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1),
                dim=2
            )
            delta_m_list[0] = F.normalize(
                self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1),
                dim=2
            )

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](
                    h_t[i - 1], h_t[i], c_t[i], memory
                )

                delta_c_list[i] = F.normalize(
                    self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1),
                    dim=2
                )
                delta_m_list[i] = F.normalize(
                    self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1),
                    dim=2
                )

            x_gen = self.conv_last(h_t[-1])   # (B,1,H,W)
            next_frames.append(x_gen)

            if kwargs.get("return_aux_loss", False):
                for i in range(self.num_layers):
                    decouple_loss_list.append(
                        torch.mean(
                            torch.abs(
                                torch.cosine_similarity(
                                    delta_c_list[i], delta_m_list[i], dim=2
                                )
                            )
                        )
                    )

        outputs = torch.stack(next_frames, dim=1)                  # (B, Tin+Tout-1, 1, H, W)
        outputs = outputs[:, Tin - 1:Tin - 1 + Tout].contiguous() # (B, Tout, 1, H, W)

        if kwargs.get("return_aux_loss", False):
            if len(decouple_loss_list) == 0:
                aux_loss = torch.tensor(0.0, device=device, dtype=x.dtype)
            else:
                aux_loss = torch.mean(torch.stack(decouple_loss_list, dim=0))

            beta = float(self.configs.get("decouple_beta", 0.0))
            aux_loss = beta * aux_loss
            return outputs, aux_loss

        return outputs