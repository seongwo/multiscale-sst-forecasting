"""SST adapter for the pinned official PiFormer implementation."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

from .piformer_memory import install


SOURCE_COMMIT = "fc922dca8199eafe062e7b24cb09913aeaffc77c"
SOURCE_HASHES = {
    "models/PiFormer.py": "9900be066dc8b69650218971c2a4edd2f3bb2b75039cdbbd3efdb77c30af0eb3",
    "layers/Embed.py": "917bf86e7f9c99feff1e2cad1dc3adf980de4cb8f61f121468e075678e86df8e",
}


def load_core(source_root):
    """Load only the required official files without retaining global aliases."""
    root = Path(source_root).resolve()
    for name, expected in SOURCE_HASHES.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name}. Supply --piformer-root for official PiFormer "
                f"commit {SOURCE_COMMIT}; see docs/additional_models.md."
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"PiFormer source differs from the pinned revision: {name}")
    # Upstream imports layers.Embed as a top-level module. Limit that alias to
    # loading so it cannot shadow another model's layers package afterwards.
    names = ("layers", "layers.Embed")
    previous = {name: sys.modules.get(name) for name in names}
    parent = ModuleType("layers")
    parent.__path__ = [str(root / "layers")]
    try:
        sys.modules["layers"] = parent
        for name, path in (("layers.Embed", root / "layers/Embed.py"),
                           ("msf_official_piformer", root / "models/PiFormer.py")):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return module.Model
    finally:
        for name in names:
            if previous[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous[name]


class PiFormerSST(nn.Module):
    def __init__(self, settings, source_root, node_chunk=32, checkpoint_enabled=True):
        super().__init__()
        self.settings = dict(settings)
        self.core = load_core(source_root)(SimpleNamespace(**settings))
        self.tin, self.tout = settings["seq_len"], settings["pred_len"]
        self.height, self.width = settings["length"], settings["width"]
        if settings["num_nodes"] != self.height * self.width + 1:
            raise ValueError("PiFormer requires one extra terminal padding node")
        if node_chunk < 1:
            raise ValueError("node_chunk must be positive")
        install(self, node_chunk=node_chunk, checkpoint_enabled=checkpoint_enabled)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1:] != (self.tin, 1, self.height, self.width):
            raise ValueError("Expected (B,14,1,32,32) normalized SST inputs")
        cells = self.height * self.width
        grid = x[:, :, 0].reshape(len(x), self.tin, cells)
        ghost = grid.new_zeros((len(x), self.tin, 1))
        output = self.core(torch.cat((grid, ghost), dim=-1))
        return output[:, :, :cells].reshape(len(x), self.tout, 1, self.height, self.width)
