from __future__ import annotations


def invert_spatial_rotation(tensor, k: int):
    """Undo a torch.rot90 image-space augmentation."""
    import torch

    return torch.rot90(tensor, -int(k) % 4, dims=(-2, -1))


def invert_hv_rotation(hv, k: int):
    """Undo spatial rotation for H/V or four-direction centroid maps."""
    import torch

    aligned = invert_spatial_rotation(hv, k)
    k = int(k) % 4
    horizontal = aligned[:, 0]
    vertical = aligned[:, 1]
    if k == 0:
        return aligned
    if k == 1:
        components = [-vertical, horizontal]
    elif k == 2:
        components = [-horizontal, -vertical]
    else:
        components = [vertical, -horizontal]
    if aligned.shape[1] == 2:
        return torch.stack(components, dim=1)
    if aligned.shape[1] != 4:
        raise ValueError("Expected two or four directional channels")
    diagonal = aligned[:, 2]
    anti_diagonal = aligned[:, 3]
    if k == 1:
        components.extend([anti_diagonal, -diagonal])
    elif k == 2:
        components.extend([-diagonal, -anti_diagonal])
    else:
        components.extend([-anti_diagonal, diagonal])
    return torch.stack(components, dim=1)


def invert_hv_horizontal_flip(hv):
    """Undo a horizontal flip for H/V or four-direction centroid maps."""
    import torch

    aligned = hv.flip(-1)
    components = [-aligned[:, 0], aligned[:, 1]]
    if aligned.shape[1] == 4:
        components.extend([-aligned[:, 3], -aligned[:, 2]])
    elif aligned.shape[1] != 2:
        raise ValueError("Expected two or four directional channels")
    return torch.stack(components, dim=1)


def invert_hv_vertical_flip(hv):
    """Undo a vertical flip for H/V or four-direction centroid maps."""
    import torch

    aligned = hv.flip(-2)
    components = [aligned[:, 0], -aligned[:, 1]]
    if aligned.shape[1] == 4:
        components.extend([aligned[:, 3], aligned[:, 2]])
    elif aligned.shape[1] != 2:
        raise ValueError("Expected two or four directional channels")
    return torch.stack(components, dim=1)
