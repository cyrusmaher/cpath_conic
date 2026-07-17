from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


DEFAULT_HV_20X = {
    "binary_threshold": 0.5,
    "edge_threshold": 0.4,
    "object_size": 3,
    "ksize": 11,
    "opening_size": 5,
    "min_nucleus_size": 10,
    "directional_weight": 1.0,
}


def load_decoder_config(path: Path | None) -> dict:
    if path is None:
        return DEFAULT_HV_20X.copy()
    payload = json.loads(path.read_text())
    if "best" in payload:
        payload = payload["best"].get("config", payload["best"])
    elif "config" in payload:
        payload = payload["config"]
    config = DEFAULT_HV_20X.copy()
    config.update({key: payload[key] for key in config if key in payload})
    if "scale" in payload:
        config["scale"] = float(payload["scale"])
    return config


def decode_hv(binary_probability: np.ndarray, hv_map: np.ndarray, scale: float = 1.0, **config) -> np.ndarray:
    """Decode CellViT foreground/HV maps with explicit, tunable 20× parameters."""
    parameters = DEFAULT_HV_20X.copy()
    parameters.update(config)
    binary_threshold = float(parameters["binary_threshold"])
    edge_threshold = float(parameters["edge_threshold"])
    object_size = int(parameters["object_size"])
    ksize = int(parameters["ksize"])
    opening_size = int(parameters["opening_size"])
    min_nucleus_size = int(parameters["min_nucleus_size"])
    directional_weight = float(parameters["directional_weight"])
    if ksize <= 0 or ksize % 2 == 0 or ksize > 31:
        raise ValueError("ksize must be a positive odd OpenCV Sobel kernel no larger than 31")
    if opening_size <= 0 or opening_size % 2 == 0:
        raise ValueError("opening_size must be a positive odd integer")
    if not 0.0 <= directional_weight <= 1.0:
        raise ValueError("directional_weight must be between zero and one")
    if binary_probability.ndim != 2 or hv_map.shape not in {
        (*binary_probability.shape, 2),
        (*binary_probability.shape, 4),
    }:
        raise ValueError("Expected binary_probability HxW and hv_map HxWx2 or HxWx4")
    if scale <= 0:
        raise ValueError("scale must be positive")
    if not np.isclose(scale, 1.0):
        height, width = binary_probability.shape
        target_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        resized_binary = cv2.resize(np.asarray(binary_probability, dtype=np.float32), target_size, interpolation=cv2.INTER_LINEAR)
        resized_hv = cv2.resize(np.asarray(hv_map, dtype=np.float32), target_size, interpolation=cv2.INTER_LINEAR)
        high_resolution = decode_hv(resized_binary, resized_hv, scale=1.0, **parameters)
        decoded = cv2.resize(high_resolution, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        old_ids = np.unique(decoded)
        old_ids = old_ids[old_ids != 0]
        remapped = np.zeros_like(decoded)
        for new_id, old_id in enumerate(old_ids, start=1):
            remapped[decoded == old_id] = new_id
        return remapped

    foreground = np.asarray(binary_probability >= binary_threshold, dtype=bool)
    foreground = remove_small_objects(foreground, min_size=min_nucleus_size)
    binary = foreground.astype(np.int32)
    if not foreground.any():
        return np.zeros(binary_probability.shape, dtype=np.int32)

    horizontal = cv2.normalize(np.asarray(hv_map[..., 0], dtype=np.float32), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    vertical = cv2.normalize(np.asarray(hv_map[..., 1], dtype=np.float32), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    sobel_h = cv2.Sobel(horizontal, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_v = cv2.Sobel(vertical, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel_h = 1 - cv2.normalize(sobel_h, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    sobel_v = 1 - cv2.normalize(sobel_v, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    edge = np.maximum(sobel_h, sobel_v)
    if hv_map.shape[-1] == 4:
        root_two = np.sqrt(2.0)
        diagonal = cv2.normalize(np.asarray(hv_map[..., 2], dtype=np.float32), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        anti_diagonal = cv2.normalize(np.asarray(hv_map[..., 3], dtype=np.float32), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        diagonal_dx = cv2.Sobel(diagonal, cv2.CV_64F, 1, 0, ksize=ksize)
        diagonal_dy = cv2.Sobel(diagonal, cv2.CV_64F, 0, 1, ksize=ksize)
        anti_dx = cv2.Sobel(anti_diagonal, cv2.CV_64F, 1, 0, ksize=ksize)
        anti_dy = cv2.Sobel(anti_diagonal, cv2.CV_64F, 0, 1, ksize=ksize)
        sobel_diagonal = (diagonal_dx + diagonal_dy) / root_two
        sobel_anti = (anti_dx - anti_dy) / root_two
        sobel_diagonal = 1 - cv2.normalize(sobel_diagonal, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        sobel_anti = 1 - cv2.normalize(sobel_anti, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        directional_edge = np.maximum(sobel_diagonal, sobel_anti)
        edge = edge + directional_weight * np.maximum(directional_edge - edge, 0.0)
    edge = np.maximum(edge - (1 - binary), 0)
    distance = -cv2.GaussianBlur((1.0 - edge) * binary, (3, 3), 0)

    marker = binary - np.asarray(edge >= edge_threshold, dtype=np.int32)
    marker[marker < 0] = 0
    marker = ndi.binary_fill_holes(marker).astype(np.uint8)
    if opening_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_size, opening_size))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker, _ = ndi.label(marker)
    if object_size > 1 and marker.max() > 0:
        sizes = np.bincount(marker.ravel())
        remove = sizes < object_size
        remove[0] = False
        marker[remove[marker]] = 0
    decoded = watershed(distance, markers=marker, mask=foreground).astype(np.int32)
    old_ids = np.unique(decoded)
    old_ids = old_ids[old_ids != 0]
    if len(old_ids) and not np.array_equal(old_ids, np.arange(1, len(old_ids) + 1)):
        remapped = np.zeros_like(decoded)
        for new_id, old_id in enumerate(old_ids, start=1):
            remapped[decoded == old_id] = new_id
        decoded = remapped
    return decoded


def fast_binary_pq_stats(true_inst: np.ndarray, pred_inst: np.ndarray, threshold: float = 0.5) -> tuple[int, int, int, float]:
    """Exact binary PQ counts; IoU > 0.5 guarantees one-to-one matches."""
    true_flat = np.asarray(true_inst, dtype=np.int64).ravel()
    pred_flat = np.asarray(pred_inst, dtype=np.int64).ravel()
    n_true = int(true_flat.max()) + 1
    n_pred = int(pred_flat.max()) + 1
    true_area = np.bincount(true_flat, minlength=n_true)
    pred_area = np.bincount(pred_flat, minlength=n_pred)
    joint = np.bincount(true_flat * n_pred + pred_flat, minlength=n_true * n_pred).reshape(n_true, n_pred)
    gt_ids, pred_ids = np.nonzero(joint[1:, 1:])
    gt_ids = gt_ids + 1
    pred_ids = pred_ids + 1
    intersections = joint[gt_ids, pred_ids]
    unions = true_area[gt_ids] + pred_area[pred_ids] - intersections
    ious = intersections / np.maximum(unions, 1)
    matched = ious > threshold
    tp = int(matched.sum())
    return tp, n_pred - 1 - tp, n_true - 1 - tp, float(ious[matched].sum())
