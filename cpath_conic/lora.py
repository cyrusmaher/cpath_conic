from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper that leaves the pretrained linear layer frozen."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = (self.dropout(inputs) @ self.lora_A.t()) @ self.lora_B.t()
        return self.base(inputs) + self.scale * update


def inject_sam_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    last_n_blocks: int | None = None,
    target_projection: bool = True,
) -> list[str]:
    """Inject LoRA into SAM attention QKV and output projections."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    blocks = model.encoder.blocks
    first_block = 0 if last_n_blocks is None else max(0, len(blocks) - last_n_blocks)
    injected = []
    for block_index in range(first_block, len(blocks)):
        attention = blocks[block_index].attn
        attention.qkv = LoRALinear(attention.qkv, rank, alpha, dropout)
        injected.append(f"encoder.blocks.{block_index}.attn.qkv")
        if target_projection:
            attention.proj = LoRALinear(attention.proj, rank, alpha, dropout)
            injected.append(f"encoder.blocks.{block_index}.attn.proj")
    return injected


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if ".lora_A" in name or ".lora_B" in name
    }
    header = model.hv_map_decoder.decoder0_header[-1]
    if getattr(header, "out_channels", 2) == 4:
        from cpath_conic.directional import directional_header_state_dict

        state.update(directional_header_state_dict(model))
    return state


def set_lora_train_mode(model: nn.Module) -> None:
    """Train LoRA paths without mutating frozen normalization buffers."""
    model.eval()
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.dropout.train()


def save_lora_adapter(model: nn.Module, path: Path, configuration: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter_state_dict": lora_state_dict(model), "configuration": configuration}, path)


def load_lora_adapter(model: nn.Module, path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    configuration = checkpoint["configuration"]
    inject_sam_lora(
        model,
        rank=int(configuration["rank"]),
        alpha=float(configuration["alpha"]),
        dropout=float(configuration.get("dropout", 0.0)),
        last_n_blocks=configuration.get("last_n_blocks"),
        target_projection=bool(configuration.get("target_projection", True)),
    )
    if configuration.get("directional_maps", {}).get("enabled", False):
        from cpath_conic.directional import HV_HEADER_PREFIX, expand_hv_head

        expand_hv_head(model)
        if not any(name.startswith(HV_HEADER_PREFIX) for name in checkpoint["adapter_state_dict"]):
            raise RuntimeError("Directional adapter checkpoint does not contain its four-map header")
    message = model.load_state_dict(checkpoint["adapter_state_dict"], strict=False)
    missing_adapter = [name for name in message.missing_keys if ".lora_" in name]
    if missing_adapter or message.unexpected_keys:
        raise RuntimeError(f"Could not load LoRA adapter: {message}")
    return configuration
