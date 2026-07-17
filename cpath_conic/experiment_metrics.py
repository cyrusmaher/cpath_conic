from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import CLASS_NAMES, COUNT_COLUMNS
from .data import patch_count_from_maps


def _feature_signature(feature_patch_ids: np.ndarray, feature_instance_ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(feature_patch_ids, dtype=np.int32).view(np.uint8))
    digest.update(np.ascontiguousarray(feature_instance_ids, dtype=np.int32).view(np.uint8))
    return digest.hexdigest()


def _file_signature(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    return f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"


def build_instance_cache(
    prepared: Path,
    feature_patch_ids: np.ndarray,
    feature_instance_ids: np.ndarray,
    instance_maps_path: Path,
) -> dict[str, np.ndarray]:
    """Cache fixed-mask metadata needed by metric-aligned experiments."""
    metadata = pd.read_csv(prepared / "metadata.csv").sort_values("patch_id")
    n_patches = int(metadata.patch_id.max()) + 1
    if len(metadata) != n_patches or not np.array_equal(metadata.patch_id, np.arange(n_patches)):
        raise ValueError("Expected contiguous patch IDs in prepared metadata")

    instance_maps = np.load(instance_maps_path, mmap_mode="r")
    if len(instance_maps) != n_patches:
        raise ValueError("Instance-map and metadata patch counts differ")

    central = np.zeros(len(feature_patch_ids), dtype=bool)
    for patch_id in np.unique(feature_patch_ids):
        rows = np.flatnonzero(feature_patch_ids == patch_id)
        represented = np.unique(instance_maps[int(patch_id), 16:-16, 16:-16])
        central[rows] = np.isin(feature_instance_ids[rows], represented[represented > 0])

    gt_full_counts = np.zeros((n_patches, len(CLASS_NAMES)), dtype=np.int32)
    for patch_id in range(n_patches):
        label = np.load(prepared / "labels" / f"{patch_id:05d}.npy", allow_pickle=True).item()
        gt_full_counts[patch_id] = patch_count_from_maps(label["inst_map"], label["class_map"])

    split_codes = metadata.split.map({"train": 0, "val": 1, "test": 2})
    if split_codes.isna().any():
        raise ValueError("Unknown split name in metadata")
    return {
        "central": central,
        "gt_full_counts": gt_full_counts,
        "split_codes": split_codes.to_numpy(dtype=np.int8),
        "n_records": np.asarray([len(feature_patch_ids)], dtype=np.int64),
        "feature_signature": np.asarray([_feature_signature(feature_patch_ids, feature_instance_ids)]),
        "instance_maps_signature": np.asarray([_file_signature(instance_maps_path)]),
        "prepared_signature": np.asarray([str(prepared.resolve())]),
    }


def load_or_build_instance_cache(
    cache_path: Path,
    prepared: Path,
    feature_patch_ids: np.ndarray,
    feature_instance_ids: np.ndarray,
    instance_maps_path: Path,
) -> dict[str, np.ndarray]:
    if cache_path.exists():
        cached = np.load(cache_path)
        valid = (
            "feature_signature" in cached.files
            and "instance_maps_signature" in cached.files
            and "prepared_signature" in cached.files
            and int(cached["n_records"][0]) == len(feature_patch_ids)
            and str(cached["feature_signature"][0]) == _feature_signature(feature_patch_ids, feature_instance_ids)
            and str(cached["instance_maps_signature"][0]) == _file_signature(instance_maps_path)
            and str(cached["prepared_signature"][0]) == str(prepared.resolve())
        )
        if valid:
            return {name: cached[name] for name in cached.files}
    values = build_instance_cache(prepared, feature_patch_ids, feature_instance_ids, instance_maps_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **values)
    return values


def evaluate_fixed_masks(
    assignments: np.ndarray,
    labels: np.ndarray,
    ious: np.ndarray,
    feature_patch_ids: np.ndarray,
    central: np.ndarray,
    metadata: pd.DataFrame,
    gt_full_counts: np.ndarray,
    split: str,
    match_iou: float = 0.5,
) -> dict:
    """Evaluate class/reject assignments while keeping predicted masks fixed.

    Assignments use 0 for a rejected instance and 1..6 for CoNIC classes.
    The pooled-PQ calculation is exact when each predicted instance's stored
    best-overlap match is unique above the 0.5 IoU threshold, as it must be for
    disjoint instance masks.
    """
    assignments = np.asarray(assignments, dtype=np.int8)
    if len(assignments) != len(feature_patch_ids):
        raise ValueError("Assignment and feature record lengths differ")
    split_rows = metadata.loc[metadata.split == split].sort_values("patch_id")
    split_patch_ids = split_rows.patch_id.to_numpy(dtype=np.int32)
    feature_mask = np.isin(feature_patch_ids, split_patch_ids)

    patch_lookup = np.full(int(metadata.patch_id.max()) + 1, -1, dtype=np.int32)
    patch_lookup[split_patch_ids] = np.arange(len(split_patch_ids), dtype=np.int32)
    count_mask = feature_mask & central & (assignments > 0)
    pred_counts = np.zeros((len(split_patch_ids), len(CLASS_NAMES)), dtype=np.int32)
    np.add.at(
        pred_counts,
        (patch_lookup[feature_patch_ids[count_mask]], assignments[count_mask] - 1),
        1,
    )
    true_counts = split_rows[COUNT_COLUMNS].to_numpy(dtype=np.float64)
    per_class_r2 = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        target = true_counts[:, class_index]
        residual = float(np.square(target - pred_counts[:, class_index]).sum())
        total = float(np.square(target - target.mean()).sum())
        per_class_r2[class_name] = 1.0 - residual / total if total > 0 else float("nan")

    assigned = assignments[feature_mask]
    target_labels = labels[feature_mask]
    target_ious = ious[feature_mask]
    matched = (assigned > 0) & (assigned == target_labels) & (target_ious > match_iou)
    gt_totals = gt_full_counts[split_patch_ids].sum(axis=0).astype(np.int64)
    per_class_pq = {}
    for class_id, class_name in enumerate(CLASS_NAMES, start=1):
        class_matched = matched & (assigned == class_id)
        tp = int(class_matched.sum())
        pred_total = int((assigned == class_id).sum())
        fp = pred_total - tp
        fn = int(gt_totals[class_id - 1]) - tp
        sum_iou = float(target_ious[class_matched].sum())
        denominator = tp + 0.5 * fp + 0.5 * fn
        dq = tp / denominator if denominator else 0.0
        sq = sum_iou / tp if tp else 0.0
        per_class_pq[class_name] = {
            "pq": sum_iou / denominator if denominator else 0.0,
            "dq": dq,
            "sq": sq,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "sum_iou": sum_iou,
        }

    finite_r2 = [value for value in per_class_r2.values() if np.isfinite(value)]
    return {
        "split": split,
        "n_patches": int(len(split_patch_ids)),
        "R2": float(np.mean(finite_r2)),
        "per_class_R2": per_class_r2,
        "mPQ+": float(np.mean([value["pq"] for value in per_class_pq.values()])),
        "per_class_pq": per_class_pq,
        "rejected_fraction": float((assigned == 0).mean()),
        "predicted_counts": pred_counts,
    }
