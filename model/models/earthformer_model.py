"""Absolute-SST adapter around the official Earthformer core."""
from torch import nn

from model.vendor.earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel


class EarthformerSST(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.core = CuboidTransformerModel(**settings)
        self.tin, self.height, self.width, self.channels = settings["input_shape"]
        self.tout = settings["target_shape"][0]

    def forward(self, x):
        if x.ndim != 5 or x.shape[1:] != (
            self.tin, self.channels, self.height, self.width
        ):
            raise ValueError("Expected (B,14,1,32,32) normalized SST inputs")
        output = self.core(x.permute(0, 1, 3, 4, 2).contiguous())
        return output.permute(0, 1, 4, 2, 3).contiguous()
