import torch
import torch.nn as nn


class SSTConvLSTM(nn.Module):
    def __init__(self, encoder, convlstm, decoder):
        super().__init__()
        self.encoder = encoder
        self.convlstm = convlstm
        self.decoder = decoder

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        z = self.encoder(x)

        _, Cz, Hz, Wz = z.shape
        z = z.view(B, T, Cz, Hz, Wz)
        z = z.contiguous()

        pred_z = self.convlstm(z)
        
        pred_z = pred_z.permute(0, 1, 4, 2, 3).contiguous()
        pred_z = pred_z.view(-1, Cz, Hz, Wz)

        out = self.decoder(pred_z)
        out = out.view(B, -1, 1, H, W)

        return out
