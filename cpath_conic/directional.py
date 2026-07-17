from __future__ import annotations

import math

import numpy as np


HV_HEADER_PREFIX = "hv_map_decoder.decoder0_header.2"


def instance_directional_map(instance_map: np.ndarray) -> np.ndarray:
    """Generate horizontal, vertical, and two diagonal centroid maps."""
    from cpath_conic.segmentation import instance_hv_map

    hv = instance_hv_map(instance_map)
    diagonal = (hv[0] + hv[1]) / math.sqrt(2.0)
    anti_diagonal = (hv[0] - hv[1]) / math.sqrt(2.0)
    for instance_id in np.unique(instance_map):
        if instance_id == 0:
            continue
        mask = instance_map == instance_id
        for channel in (diagonal, anti_diagonal):
            values = channel[mask]
            negative = values < 0
            positive = values > 0
            if negative.any():
                values[negative] /= -values[negative].min()
            if positive.any():
                values[positive] /= values[positive].max()
            channel[mask] = values
    return np.stack((hv[0], hv[1], diagonal, anti_diagonal)).astype(np.float32)


def expand_hv_head(model) -> None:
    """Expand a pretrained two-map CellViT header to four directional maps."""
    import torch
    import torch.nn as nn

    old = model.hv_map_decoder.decoder0_header[-1]
    if not isinstance(old, nn.Conv2d) or old.kernel_size != (1, 1):
        raise TypeError("Expected the CellViT HV decoder to end in a 1x1 convolution")
    if old.out_channels == 4:
        return
    if old.out_channels != 2:
        raise ValueError(f"Expected a two-channel HV header, found {old.out_channels}")
    new = nn.Conv2d(old.in_channels, 4, kernel_size=1, stride=1, padding=0, bias=old.bias is not None)
    new = new.to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight[:2].copy_(old.weight)
        new.weight[2].copy_((old.weight[0] + old.weight[1]) / math.sqrt(2.0))
        new.weight[3].copy_((old.weight[0] - old.weight[1]) / math.sqrt(2.0))
        if old.bias is not None:
            new.bias[:2].copy_(old.bias)
            new.bias[2].copy_((old.bias[0] + old.bias[1]) / math.sqrt(2.0))
            new.bias[3].copy_((old.bias[0] - old.bias[1]) / math.sqrt(2.0))
    model.hv_map_decoder.decoder0_header[-1] = new
    model.branches_output["hv_map"] = 4


def directional_header_state_dict(model) -> dict:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith(HV_HEADER_PREFIX)
    }


def set_directional_header_trainable(model) -> list:
    header = model.hv_map_decoder.decoder0_header[-1]
    for parameter in header.parameters():
        parameter.requires_grad = True
    return list(header.parameters())
