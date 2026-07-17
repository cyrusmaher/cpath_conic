from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .constants import CLASS_NAMES, COUNT_COLUMNS


def _decode_image(value: Any) -> np.ndarray:
    """Decode common Hugging Face image-column representations."""
    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"), dtype=np.uint8)
    if isinstance(value, dict):
        value = value.get("bytes", value.get("path", value))
    if isinstance(value, str):
        return np.asarray(Image.open(value).convert("RGB"), dtype=np.uint8)
    if isinstance(value, (bytes, bytearray, memoryview)):
        from io import BytesIO

        return np.asarray(Image.open(BytesIO(bytes(value))).convert("RGB"), dtype=np.uint8)
    arr = np.asarray(value)
    if arr.dtype == object and arr.size == 1:
        return _decode_image(arr.item())
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=-1).astype(np.uint8)
    return arr.astype(np.uint8)


def _decode_mask(value: Any, dtype: np.dtype) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("bytes", value.get("path", value))
    if isinstance(value, (bytes, bytearray, memoryview)):
        from io import BytesIO

        return np.asarray(Image.open(BytesIO(bytes(value))), dtype=dtype)
    if isinstance(value, str):
        return np.asarray(Image.open(value), dtype=dtype)
    arr = np.asarray(value)
    if arr.dtype == object and arr.size == 1:
        return _decode_mask(arr.item(), dtype)
    return arr.astype(dtype)


def source_group(patch_info: str, source: str = "") -> str:
    """Return a source-image key, removing the public tile suffix."""
    token = str(patch_info or "")
    token = re.sub(r"-[0-9]{4,}$", "", token)
    return f"{source}:{token}" if source else token


def stable_split(group: str, seed: int = 20260715) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
    value = int(digest[:8], 16) / 2**32
    if value < 0.80:
        return "train"
    if value < 0.90:
        return "val"
    return "test"


def official_hovernet_fold(metadata: pd.DataFrame, fold_index: int = 0, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the public HoVer-Net baseline's group-stratified split."""
    from sklearn.model_selection import StratifiedShuffleSplit

    ordered = metadata.sort_values("patch_id")
    patch_info = ordered.patch_info.astype(str).tolist()
    image_sources = np.unique([value.split("-")[0] for value in patch_info])
    cohort_sources = [value.split("_")[0] for value in image_sources]
    _, strata = np.unique(cohort_sources, return_inverse=True)
    splitter = StratifiedShuffleSplit(n_splits=10, train_size=0.8, test_size=0.2, random_state=seed)
    splits = list(splitter.split(image_sources, strata))
    if not 0 <= fold_index < len(splits):
        raise ValueError(f"fold_index must lie in [0, {len(splits) - 1}]")
    train_group_indexes, validation_group_indexes = splits[fold_index]
    train_groups = set(image_sources[train_group_indexes])
    validation_groups = set(image_sources[validation_group_indexes])
    groups_by_patch = np.asarray([value.split("-")[0] for value in patch_info])
    patch_ids = ordered.patch_id.to_numpy(dtype=np.int32)
    train_ids = patch_ids[np.isin(groups_by_patch, list(train_groups))]
    validation_ids = patch_ids[np.isin(groups_by_patch, list(validation_groups))]
    if len(np.intersect1d(train_ids, validation_ids)) or train_groups & validation_groups:
        raise RuntimeError("Official HoVer-Net fold reconstruction is not group-disjoint")
    return train_ids, validation_ids


def load_metadata(prepared: Path) -> pd.DataFrame:
    return pd.read_csv(prepared / "metadata.csv")


def load_patch(prepared: Path, patch_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    image = np.asarray(Image.open(prepared / "images" / f"{patch_id:05d}.png").convert("RGB"))
    label = np.load(prepared / "labels" / f"{patch_id:05d}.npy", allow_pickle=True).item()
    meta = load_metadata(prepared)
    row = meta.loc[meta.patch_id == patch_id].iloc[0].to_dict()
    return image, label["inst_map"], label["class_map"], row


def patch_count_from_maps(inst_map: np.ndarray, class_map: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for inst_id in np.unique(inst_map):
        if inst_id == 0:
            continue
        pixels = class_map[inst_map == inst_id]
        if len(pixels):
            cls = int(np.bincount(pixels.astype(np.int64), minlength=7).argmax())
            if 1 <= cls <= 6:
                counts[cls - 1] += 1
    return counts


def central_crop_counts(inst_map: np.ndarray, class_map: np.ndarray, margin: int = 16) -> np.ndarray:
    """Count instances represented anywhere in the official 224x224 crop."""
    h, w = inst_map.shape
    crop = np.s_[margin : h - margin, margin : w - margin]
    return patch_count_from_maps(inst_map[crop], class_map[crop])


def prediction_to_class_map(inst_map: np.ndarray, instance_classes: np.ndarray) -> np.ndarray:
    class_map = np.zeros_like(inst_map, dtype=np.uint8)
    for inst_id, cls in enumerate(instance_classes, start=1):
        class_map[inst_map == inst_id] = int(cls)
    return class_map
