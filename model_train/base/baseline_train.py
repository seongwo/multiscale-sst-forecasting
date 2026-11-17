"""Train PiFormer, Earthformer, ViT-GRU or Single-scale using paper presets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from build_datasets import build_datasets
from model_train.base.baseline_models import CONFIG_FOLDERS, build_model, config_path, training_config
from model_train.base.evaluate_predictions import evaluate
from trainer import set_seed, train


def export_test(model, dataset, mask, mean, std, cfg, output):
    """Save complete predictions and score physical-unit errors after selection."""
    shape = (len(dataset), cfg["pred_len"], 1, *mask.shape)
    pred = np.lib.format.open_memmap(output / "test_predictions_normalized.npy",
                                   mode="w+", dtype=np.float32, shape=shape)
    target = np.lib.format.open_memmap(output / "test_targets_normalized.npy",
                                     mode="w+", dtype=np.float32, shape=shape)
    model.eval()
    offset = 0
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False,
                               drop_last=False, num_workers=cfg["num_workers"]):
            prediction = model(x.to(cfg["device"]))
            if prediction.shape != y.shape:
                raise ValueError("Model output shape does not match the SST targets")
            last = offset + len(x)
            pred[offset:last] = prediction.cpu().numpy()
            target[offset:last] = y.numpy()
            offset = last
    if offset != len(dataset):
        raise RuntimeError("Test export omitted samples")
    pred.flush()
    target.flush()
    return evaluate(pred, target, mask, mean, std, batch_size=cfg["batch_size"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CONFIG_FOLDERS), required=True)
    parser.add_argument("--seed", type=int, choices=range(42, 47), default=42)
    parser.add_argument("--config", type=Path, help="Override the preset JSON")
    parser.add_argument("--piformer-root", type=Path, help="Pinned official PiFormer checkout")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--data-path")
    parser.add_argument("--mask-path")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--check-model", action="store_true",
                        help="Check parameters and one forward pass without SST data or training")
    args = parser.parse_args(argv)
    reference = args.config or config_path(args.model, args.seed)
    settings = json.loads(reference.read_text())["settings"]
    cfg = training_config(settings)
    cfg["seed"] = args.seed
    cfg["device"] = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg["exp_name"] = f"{args.model}_seed{args.seed}"
    if args.data_path:
        cfg["data_path"] = args.data_path
    if args.mask_path:
        cfg["mask_path"] = args.mask_path
    output = args.output_root or Path("outputs") / cfg["exp_name"]
    cfg["ckpt_dir"] = str(output / "checkpoints")
    cfg["results_dir"] = str(output)
    cfg["stats_cache"] = str(output / "normalization.json")
    if args.print_config:
        print(json.dumps(cfg, indent=2))
        return
    set_seed(cfg["seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = build_model(args.model, settings, args.piformer_root).to(cfg["device"])
    if args.check_model:
        model.eval()
        with torch.no_grad():
            prediction = model(torch.zeros(1, cfg["input_len"], 1, 32, 32,
                                           device=cfg["device"]))
        if prediction.shape != (1, cfg["pred_len"], 1, 32, 32) or not torch.isfinite(prediction).all():
            raise ValueError("Model shape/finite check failed")
        print(json.dumps({"model": args.model,
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "output_shape": list(prediction.shape),
                          "objective": cfg["objective"], "lr": cfg["lr"],
                          "lambda_gdl": cfg["lambda_gdl"]}, indent=2))
        return
    # A run owns a fresh output directory; reference tables/checkpoints are never overwritten.
    output.mkdir(parents=True, exist_ok=False)
    mask = np.load(cfg["mask_path"], allow_pickle=False).astype(bool)
    train_set, val_set, test_set, mean, std, torch_mask = build_datasets(
        data_path=cfg["data_path"], mask=mask, tin=cfg["input_len"], tout=cfg["pred_len"],
        var_name=cfg["var_name"], train_start=cfg["train_start"], train_end=cfg["train_end"],
        val_start=cfg["val_start"], val_end=cfg["val_end"],
        test_start=cfg["test_start"], test_end=cfg["test_end"],
        stats_cache_json=cfg["stats_cache"],
    )
    cfg.update(train_mean_c=mean, train_std_c=std)
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    checkpoint = output / "checkpoints" / "best.pt"
    _, best_epoch, best_val, history = train(
        model, train_set, val_set, test_set, torch_mask, cfg, str(checkpoint)
    )
    # train() has restored the Validation-selected checkpoint before returning.
    metrics = export_test(model, test_set, mask, mean, std, cfg, output)
    metrics.update(best_epoch_zero_based=best_epoch, validation_rmse_c=best_val * std)
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
