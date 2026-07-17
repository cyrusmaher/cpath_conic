from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass


def instance_hv_map(instance_map: np.ndarray) -> np.ndarray:
    """Generate HoVer-Net-style horizontal/vertical targets in [-1, 1]."""
    x_map = np.zeros(instance_map.shape, dtype=np.float32)
    y_map = np.zeros(instance_map.shape, dtype=np.float32)
    for instance_id in np.unique(instance_map):
        if instance_id == 0:
            continue
        ys, xs = np.where(instance_map == instance_id)
        if not len(ys):
            continue
        y0, y1 = max(0, int(ys.min()) - 2), min(instance_map.shape[0], int(ys.max()) + 3)
        x0, x1 = max(0, int(xs.min()) - 2), min(instance_map.shape[1], int(xs.max()) + 3)
        mask = (instance_map[y0:y1, x0:x1] == instance_id).astype(np.uint8)
        if min(mask.shape) < 2:
            continue
        center_y, center_x = center_of_mass(mask)
        center_y, center_x = int(center_y + 0.5), int(center_x + 0.5)
        x_values, y_values = np.meshgrid(
            np.arange(1, mask.shape[1] + 1) - center_x,
            np.arange(1, mask.shape[0] + 1) - center_y,
        )
        x_values = x_values.astype(np.float32)
        y_values = y_values.astype(np.float32)
        x_values[mask == 0] = 0
        y_values[mask == 0] = 0
        negative = x_values < 0
        positive = x_values > 0
        if negative.any():
            x_values[negative] /= -x_values[negative].min()
        if positive.any():
            x_values[positive] /= x_values[positive].max()
        negative = y_values < 0
        positive = y_values > 0
        if negative.any():
            y_values[negative] /= -y_values[negative].min()
        if positive.any():
            y_values[positive] /= y_values[positive].max()
        x_box = x_map[y0:y1, x0:x1]
        y_box = y_map[y0:y1, x0:x1]
        x_box[mask > 0] = x_values[mask > 0]
        y_box[mask > 0] = y_values[mask > 0]
    return np.stack([x_map, y_map])

