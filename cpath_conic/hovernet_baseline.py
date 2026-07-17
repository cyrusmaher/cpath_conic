from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _install_tiatoolbox_compatibility() -> None:
    """Provide the two tiny TIAToolbox classes used by official net_desc.py."""
    import numpy as np
    import torch
    import torch.nn as nn

    class UpSample2x(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("unpool_mat", torch.from_numpy(np.ones((2, 2), dtype="float32")))

        def forward(self, inputs):
            shape = list(inputs.shape)
            result = torch.tensordot(inputs.unsqueeze(-1), self.unpool_mat.unsqueeze(0), dims=1)
            result = result.permute(0, 1, 2, 4, 3, 5)
            return result.reshape((-1, shape[1], shape[2] * 2, shape[3] * 2))

    class HoVerNetCompatibility:
        pass

    modules = {
        "tiatoolbox": types.ModuleType("tiatoolbox"),
        "tiatoolbox.models": types.ModuleType("tiatoolbox.models"),
        "tiatoolbox.models.abc": types.ModuleType("tiatoolbox.models.abc"),
        "tiatoolbox.models.architecture": types.ModuleType("tiatoolbox.models.architecture"),
        "tiatoolbox.models.architecture.hovernet": types.ModuleType("tiatoolbox.models.architecture.hovernet"),
        "tiatoolbox.models.architecture.utils": types.ModuleType("tiatoolbox.models.architecture.utils"),
    }
    modules["tiatoolbox.models.abc"].ModelABC = nn.Module
    modules["tiatoolbox.models.architecture.hovernet"].HoVerNet = HoVerNetCompatibility
    modules["tiatoolbox.models.architecture.utils"].UpSample2x = UpSample2x
    sys.modules.update(modules)


def load_official_hovernet(checkpoint: Path, conic_root: Path, device: str):
    """Load either the public baseline or one of our provenance-tagged fits."""
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        isinstance(state, dict)
        and "desc" in state
        and str(state.get("initialization", "")).startswith("ImageNet ")
        and str(state.get("initialization", "")).endswith("only; no CoNIC checkpoint")
    ):
        hovernet_root = conic_root.parent / "hover_net"
        if str(hovernet_root) not in sys.path:
            sys.path.insert(0, str(hovernet_root))
        from models.hovernet.net_desc import HoVerNetExt

        architecture = state.get("backbone_architecture", state.get("args", {}).get("backbone_architecture", "resnet50"))
        model = HoVerNetExt(
            num_types=7,
            pretrained_backbone=None,
            backbone_name=architecture,
        )
        model.load_state_dict(state["desc"], strict=True)
        return model.to(device).eval()

    _install_tiatoolbox_compatibility()
    spec = importlib.util.spec_from_file_location("official_conic_net_desc", conic_root / "net_desc.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import official CoNIC model from {conic_root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.HoVerNetConic(num_types=7)
    if isinstance(state, dict) and "desc" in state:
        state = state["desc"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
