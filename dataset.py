import torch
import numpy as np
from torch.utils.data import Dataset


class OISSTDataset(Dataset):
    def __init__(self, ds, mask, mean, std, tin=14, tout=7):
        self.sst = ds["sst"]
        self.mask = mask
        self.mean = mean
        self.std = std
        self.tin = tin
        self.tout = tout

    def __len__(self):
        return len(self.sst.time) - self.tin - self.tout

    def __getitem__(self, idx):
        seq = self.sst.isel(
            time=slice(idx, idx + self.tin + self.tout)
        ).values.astype("float32")

        x = seq[: self.tin]
        y = seq[self.tin :]

        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        x = np.where(self.mask, x, 0.0)
        y = np.where(self.mask, y, 0.0)

        # x = x * self.mask
        # y = y * self.mask

        x = torch.from_numpy(x).unsqueeze(1)
        y = torch.from_numpy(y).unsqueeze(1)

        return x, y
