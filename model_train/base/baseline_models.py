"""Construct the paper's added comparison models from their saved settings."""
from pathlib import Path

from configs import get_default_cfg


ROOT = Path(__file__).resolve().parents[2]
CONFIG_FOLDERS = {
    "piformer": "PiFormer",
    "earthformer": "Earthformer",
    "vit_gru": "ViT_GRU",
    "single_scale": "Single_D152",
}


def config_path(model, seed):
    return ROOT / "docs/run_configs" / CONFIG_FOLDERS[model] / f"seed{seed}.json"


def training_config(settings):
    cfg = get_default_cfg()
    cfg.update(settings)
    cfg["input_len"] = settings["tin"]
    cfg["pred_len"] = settings["tout"]
    cfg["evaluation_aggregation"] = "global"
    cfg.setdefault("objective", "masked_mse")
    return cfg


def build_model(model, settings, piformer_root=None):
    if model == "piformer":
        from model.models.piformer_model import PiFormerSST
        return PiFormerSST(
            settings["architecture"], piformer_root or ROOT / "external/PiFormer",
            node_chunk=settings.get("feedforward_node_chunk", 32),
            checkpoint_enabled=settings.get("activation_checkpointing", True),
        )
    if model == "earthformer":
        from model.models.earthformer_model import EarthformerSST
        return EarthformerSST(settings["architecture"])
    if model == "vit_gru":
        from model.models.vit_gru_model import TorchVisionViTGRUForecast
        return TorchVisionViTGRUForecast(**settings["architecture"])
    if model == "single_scale":
        from model.models.msf3_model import MSFv3
        keys = ("image_size", "tin", "tout", "d_model", "depth", "num_heads",
                "mlp_ratio", "dropout", "attn_dropout", "spatial_scales",
                "fusion", "use_pad", "use_pos_emb", "time_pool")
        return MSFv3(**{key: settings[key] for key in keys})
    raise ValueError(f"Unknown model: {model}")
