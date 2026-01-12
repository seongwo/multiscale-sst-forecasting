import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
import xarray as xr

from tqdm import tqdm
from dataset import OISSTDataset
from losses import masked_mse, masked_rmse
from model.models.convlstm_model import ConvLSTM_Model
from model.modules.coder.cnn_coder import CNNEncoder, CNNDecoder
from model.models.sst_convlstm_model import SSTConvLSTM


device = "cuda" if torch.cuda.is_available() else "cpu"

pre_seq_length = 14
aft_seq_length = 7
total_length = pre_seq_length + aft_seq_length
in_shape = (total_length, 1, 720, 1440)
encoder_out_ch = 32 * 4

class Config:
    pass

configs = Config()
configs.pre_seq_length = pre_seq_length
configs.aft_seq_length = aft_seq_length
configs.total_length = total_length
configs.patch_size = 1
configs.filter_size = 5
configs.stride = 1
configs.layer_norm = 0
configs.reverse_scheduled_sampling = 0


ds = xr.open_zarr("data/oisst_sst.zarr")
mask = np.load("data/oisst_spatial_mask.npy")

train_ds = ds.sel(time=slice("1985-01-01", "1990-01-01"))
val_ds   = ds.sel(time=slice("1990-01-01", "1992-01-01"))

train_mean = 13.5897
train_std = 11.5540

train_dataset = OISSTDataset(train_ds, mask, train_mean, train_std)
val_dataset   = OISSTDataset(val_ds, mask, train_mean, train_std)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False)

mask_torch = torch.from_numpy(mask).float().to(device)[None, None, None]


encoder = CNNEncoder(in_ch=1, base_ch=32).to(device)
decoder = CNNDecoder(out_ch=1, base_ch=32).to(device)

with torch.no_grad():
    dummy = torch.zeros(1, 1, 720, 1440, device=device)  # [B,C,H,W]
    z = encoder(dummy)                                   # [B,Cz,Hz,Wz]
    _, Cz, Hz, Wz = z.shape

configs.in_shape = (total_length, Cz, Hz, Wz)


convlstm = ConvLSTM_Model(
    num_layers=3,
    num_hidden=[128, 128, 128],
    configs=configs
)

model = SSTConvLSTM(encoder, convlstm, decoder).to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)


x, y = next(iter(train_loader))
x = x.to(device)
y = y.to(device)

with torch.no_grad():
    pred = model(x)


def validate():
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            pred = pred[:, -y.shape[1]:]
            rmse = masked_rmse(pred, y, mask_torch)
            total += rmse.item()
    return total / len(val_loader)


for epoch in range(10):
    model.train()
    total = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}")

    for it, (x, y) in enumerate(pbar):
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        pred = pred[:, -y.shape[1]:]

        loss = masked_mse(pred, y, mask_torch)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item()

        # 실시간 표시
        pbar.set_postfix({
            "loss": f"{loss.item():.5f}",
            "grad": f"{grad_norm:.3f}"
        })

    print(
        f"Epoch {epoch:03d} | "
        f"train MSE {total / len(train_loader):.6f} | "
        f"val RMSE {validate():.6f}"
    )
