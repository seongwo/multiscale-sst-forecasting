import numpy as np
import torch
from torch.utils.data import Dataset

MISSING_VALUE = -9.96921e36

class OISSTDataset(Dataset):
    def __init__(
        self,
        ds,
        mask: np.ndarray,
        mean: float,
        std: float,
        tin: int = 14,
        tout: int = 7,
        var_name: str = "sst",
    ):
        self.tin = int(tin)
        self.tout = int(tout)

        self.mask = mask.astype(bool)  # (H, W)
        self.mean = float(mean)
        self.std = float(std) if float(std) != 0.0 else 1e-12

        arr = ds[var_name].values.astype(np.float32)  # (T, H, W)
        self.arr = arr

        self.T = arr.shape[0]
        self.H = arr.shape[1]
        self.W = arr.shape[2]

        if self.mask.shape != (self.H, self.W):
            raise ValueError(f"mask shape {self.mask.shape} != data spatial shape {(self.H, self.W)}")

        self.n = self.T - self.tin - self.tout + 1
        if self.n <= 0:
            raise ValueError("Not enough timesteps for the requested (tin, tout).")

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        seq = self.arr[idx : idx + self.tin + self.tout]  # (tin+tout, H, W)

        x = seq[: self.tin]
        y = seq[self.tin :]

        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        ocean = self.mask[None, :, :]
        x = np.where(ocean, x, 0.0)
        y = np.where(ocean, y, 0.0)

        x = torch.from_numpy(x).unsqueeze(1)  # (Tin, 1, H, W)
        y = torch.from_numpy(y).unsqueeze(1)  # (Tout, 1, H, W)

        return x, y
