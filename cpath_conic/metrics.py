from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import r2_score
from skimage.segmentation import find_boundaries

from .constants import CLASS_NAMES, SHORT_COUNT_COLUMNS


def remap_label(inst: np.ndarray) -> np.ndarray:
    out = np.zeros_like(inst, dtype=np.int32)
    for new_id, old_id in enumerate(np.unique(inst)[1:], start=1):
        out[inst == old_id] = new_id
    return out


def pq_stats(true_inst: np.ndarray, pred_inst: np.ndarray, threshold: float = 0.5) -> tuple[int, int, int, float]:
    """Return exact PQ sufficient statistics with vectorized pixel counting."""
    true_values = np.asarray(true_inst, dtype=np.int64)
    pred_values = np.asarray(pred_inst, dtype=np.int64)
    if true_values.shape != pred_values.shape:
        raise ValueError("true and predicted instance maps must have matching shapes")
    true_ids, true_counts = np.unique(true_values[true_values > 0], return_counts=True)
    pred_ids, pred_counts = np.unique(pred_values[pred_values > 0], return_counts=True)
    overlap = (true_values > 0) & (pred_values > 0)
    candidates: list[tuple[float, int, int]] = []
    if overlap.any():
        pairs, intersections = np.unique(
            np.stack([true_values[overlap], pred_values[overlap]], axis=1),
            axis=0,
            return_counts=True,
        )
        true_area = true_counts[np.searchsorted(true_ids, pairs[:, 0])]
        pred_area = pred_counts[np.searchsorted(pred_ids, pairs[:, 1])]
        unions = true_area + pred_area - intersections
        ious = intersections / np.maximum(unions, 1)
        keep = ious > threshold
        # Match the legacy reverse tuple ordering exactly: IoU, GT ID, then
        # prediction ID, all descending.  Only overlapping instance pairs are
        # iterated; the previous implementation iterated every image pixel.
        for index in np.lexsort((-pairs[keep, 1], -pairs[keep, 0], -ious[keep])):
            pair = pairs[keep][index]
            candidates.append((float(ious[keep][index]), int(pair[0]), int(pair[1])))
    matched_gt, matched_pred = set(), set()
    sum_iou = 0.0
    for iou, gt, pred in candidates:
        if gt not in matched_gt and pred not in matched_pred:
            matched_gt.add(gt)
            matched_pred.add(pred)
            sum_iou += iou
    tp = len(matched_gt)
    return tp, len(pred_ids) - tp, len(true_ids) - tp, sum_iou


def instance_type_confusion(true: np.ndarray, pred: np.ndarray, threshold: float = 0.5) -> dict:
    """Class-agnostically match nuclei, then tabulate typing, misses, and spurious detections.

    Rows are six ground-truth classes plus ``spurious_prediction``; columns are
    six predicted classes plus ``missed_truth``. This deliberately differs
    from typed PQ matching: an epithelial nucleus predicted as connective is a
    geometric match and an off-diagonal typing error, not one opaque FP+FN pair.
    """
    true = np.asarray(true)
    pred = np.asarray(pred)
    if true.shape != pred.shape or true.ndim != 4 or true.shape[-1] != 2:
        raise ValueError("true and pred must be matching NxHxWx2 instance/class maps")
    labels = [*CLASS_NAMES, "unmatched"]
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)

    def instance_classes(instance_map: np.ndarray, class_map: np.ndarray) -> dict[int, int]:
        valid = (instance_map > 0) & (class_map >= 1) & (class_map <= len(CLASS_NAMES))
        if not valid.any():
            return {}
        instance_ids, inverse = np.unique(instance_map[valid], return_inverse=True)
        joint = inverse * len(CLASS_NAMES) + (class_map[valid].astype(np.int64) - 1)
        votes = np.bincount(joint, minlength=len(instance_ids) * len(CLASS_NAMES)).reshape(
            len(instance_ids), len(CLASS_NAMES)
        )
        classes = votes.argmax(axis=1) + 1
        return {int(instance_id): int(class_id) for instance_id, class_id in zip(instance_ids, classes)}

    for true_patch, pred_patch in zip(true, pred):
        true_inst, true_cls = true_patch[..., 0], true_patch[..., 1]
        pred_inst, pred_cls = pred_patch[..., 0], pred_patch[..., 1]
        true_types = instance_classes(true_inst, true_cls)
        pred_types = instance_classes(pred_inst, pred_cls)
        true_ids, true_counts = np.unique(true_inst[true_inst > 0], return_counts=True)
        pred_ids, pred_counts = np.unique(pred_inst[pred_inst > 0], return_counts=True)
        true_area = {int(instance_id): int(count) for instance_id, count in zip(true_ids, true_counts)}
        pred_area = {int(instance_id): int(count) for instance_id, count in zip(pred_ids, pred_counts)}
        intersections: dict[tuple[int, int], int] = {}
        overlap = (true_inst > 0) & (pred_inst > 0)
        if overlap.any():
            pairs, counts = np.unique(
                np.stack([true_inst[overlap], pred_inst[overlap]], axis=1), axis=0, return_counts=True
            )
            intersections = {
                (int(pair[0]), int(pair[1])): int(count)
                for pair, count in zip(pairs, counts)
                if int(pair[0]) in true_types and int(pair[1]) in pred_types
            }
        candidates = []
        for (true_id, pred_id), intersection in intersections.items():
            union = true_area[true_id] + pred_area[pred_id] - intersection
            iou = intersection / union if union else 0.0
            if iou > threshold:
                candidates.append((iou, true_id, pred_id))
        candidates.sort(reverse=True)
        matched_true: set[int] = set()
        matched_pred: set[int] = set()
        for _, true_id, pred_id in candidates:
            if true_id in matched_true or pred_id in matched_pred:
                continue
            matched_true.add(true_id)
            matched_pred.add(pred_id)
            matrix[true_types[true_id] - 1, pred_types[pred_id] - 1] += 1
        for true_id, class_id in true_types.items():
            if true_id not in matched_true:
                matrix[class_id - 1, -1] += 1
        for pred_id, class_id in pred_types.items():
            if pred_id not in matched_pred:
                matrix[-1, class_id - 1] += 1

    matched = int(matrix[:-1, :-1].sum())
    correct = int(np.trace(matrix[:-1, :-1]))
    return {
        "labels": labels,
        "matrix": matrix.tolist(),
        "geometry_matched": matched,
        "correctly_typed": correct,
        "matched_type_accuracy": float(correct / matched) if matched else 0.0,
        "missed_truth": int(matrix[:-1, -1].sum()),
        "spurious_prediction": int(matrix[-1, :-1].sum()),
    }


def multiclass_pq_plus(true: np.ndarray, pred: np.ndarray, nr_classes: int = 6) -> dict:
    """Exact CoNIC-style pooled mPQ+ over all patches."""
    totals = np.zeros((nr_classes, 4), dtype=np.float64)
    for true_patch, pred_patch in zip(true, pred):
        true_inst, true_cls = true_patch[..., 0], true_patch[..., 1]
        pred_inst, pred_cls = pred_patch[..., 0], pred_patch[..., 1]
        for cls in range(1, nr_classes + 1):
            gt_mask = true_cls == cls
            pr_mask = pred_cls == cls
            # pq_stats supports sparse/non-contiguous IDs, so remapping each
            # class mask would be pure overhead.
            totals[cls - 1] += pq_stats(true_inst * gt_mask, pred_inst * pr_mask)

    per_class = []
    for tp, fp, fn, sum_iou in totals:
        denom = tp + 0.5 * fp + 0.5 * fn
        dq = tp / (denom + 1e-6)
        sq = sum_iou / (tp + 1e-6)
        per_class.append({"pq": float(dq * sq), "dq": float(dq), "sq": float(sq), "tp": int(tp), "fp": int(fp), "fn": int(fn), "sum_iou": float(sum_iou)})
    return {
        "mPQ+": float(np.mean([x["pq"] for x in per_class])),
        "mDQ+": float(np.mean([x["dq"] for x in per_class])),
        "mSQ+": float(np.mean([x["sq"] for x in per_class])),
        "per_class": dict(zip(CLASS_NAMES, per_class)),
    }


def _aji_plus_stats(true_inst: np.ndarray, pred_inst: np.ndarray) -> tuple[int, int]:
    """Return AJI+ intersection and union terms for one patch."""
    true_map = remap_label(np.asarray(true_inst))
    pred_map = remap_label(np.asarray(pred_inst))
    n_true = int(true_map.max())
    n_pred = int(pred_map.max())
    true_area = np.bincount(true_map.ravel(), minlength=n_true + 1)[1:]
    pred_area = np.bincount(pred_map.ravel(), minlength=n_pred + 1)[1:]
    if n_true == 0 or n_pred == 0:
        return 0, int(true_area.sum() + pred_area.sum())

    joint = np.bincount(
        true_map.ravel() * (n_pred + 1) + pred_map.ravel(),
        minlength=(n_true + 1) * (n_pred + 1),
    ).reshape(n_true + 1, n_pred + 1)[1:, 1:]
    unions = true_area[:, None] + pred_area[None, :] - joint
    iou = joint / np.maximum(unions, 1)
    true_match, pred_match = linear_sum_assignment(-iou)
    overlapping = joint[true_match, pred_match] > 0
    true_match = true_match[overlapping]
    pred_match = pred_match[overlapping]
    numerator = int(joint[true_match, pred_match].sum())
    denominator = int(unions[true_match, pred_match].sum())
    denominator += int(true_area[np.setdiff1d(np.arange(n_true), true_match)].sum())
    denominator += int(pred_area[np.setdiff1d(np.arange(n_pred), pred_match)].sum())
    return numerator, denominator


def _instance_boundaries(instance_map: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(instance_map), 1, mode="constant")
    return find_boundaries(padded, mode="thick")[1:-1, 1:-1]


def binary_instance_segmentation_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    boundary_tolerance: int = 2,
) -> dict:
    """Pooled class-agnostic semantic, instance, and boundary diagnostics.

    These metrics are explanatory diagnostics; CoNIC mPQ+ and macro-R² remain
    the model-selection objectives.
    """
    true_values = np.asarray(true)
    pred_values = np.asarray(pred)
    if true_values.ndim == 4:
        true_values = true_values[..., 0]
    if pred_values.ndim == 4:
        pred_values = pred_values[..., 0]
    if true_values.shape != pred_values.shape or true_values.ndim != 3:
        raise ValueError("true and pred must be matching NxHxW instance maps")
    if boundary_tolerance < 0:
        raise ValueError("boundary_tolerance must be non-negative")

    foreground_intersection = 0
    foreground_true = 0
    foreground_pred = 0
    pq_totals = np.zeros(4, dtype=np.float64)
    aji_intersection = 0
    aji_union = 0
    boundary_predicted = 0
    boundary_true = 0
    boundary_matched_predicted = 0
    boundary_matched_true = 0
    structure = ndi.generate_binary_structure(2, 1)
    for true_patch, pred_patch in zip(true_values, pred_values):
        true_foreground = true_patch > 0
        pred_foreground = pred_patch > 0
        foreground_intersection += int(np.logical_and(true_foreground, pred_foreground).sum())
        foreground_true += int(true_foreground.sum())
        foreground_pred += int(pred_foreground.sum())
        pq_totals += pq_stats(true_patch, pred_patch)
        numerator, denominator = _aji_plus_stats(true_patch, pred_patch)
        aji_intersection += numerator
        aji_union += denominator

        true_boundary = _instance_boundaries(true_patch)
        pred_boundary = _instance_boundaries(pred_patch)
        if boundary_tolerance:
            true_neighborhood = ndi.binary_dilation(true_boundary, structure=structure, iterations=boundary_tolerance)
            pred_neighborhood = ndi.binary_dilation(pred_boundary, structure=structure, iterations=boundary_tolerance)
        else:
            true_neighborhood = true_boundary
            pred_neighborhood = pred_boundary
        boundary_predicted += int(pred_boundary.sum())
        boundary_true += int(true_boundary.sum())
        boundary_matched_predicted += int(np.logical_and(pred_boundary, true_neighborhood).sum())
        boundary_matched_true += int(np.logical_and(true_boundary, pred_neighborhood).sum())

    foreground_union = foreground_true + foreground_pred - foreground_intersection
    foreground_iou = foreground_intersection / foreground_union if foreground_union else 1.0
    foreground_dice = (
        2.0 * foreground_intersection / (foreground_true + foreground_pred)
        if foreground_true + foreground_pred else 1.0
    )
    tp, fp, fn, sum_iou = pq_totals
    pq_denominator = tp + 0.5 * fp + 0.5 * fn
    dq = tp / pq_denominator if pq_denominator else 1.0
    sq = sum_iou / tp if tp else 0.0
    boundary_precision = boundary_matched_predicted / boundary_predicted if boundary_predicted else float(boundary_true == 0)
    boundary_recall = boundary_matched_true / boundary_true if boundary_true else float(boundary_predicted == 0)
    boundary_f1 = (
        2.0 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
        if boundary_precision + boundary_recall else 0.0
    )
    return {
        "foreground_jaccard": float(foreground_iou),
        "foreground_jaccard_distance": float(1.0 - foreground_iou),
        "foreground_dice": float(foreground_dice),
        "bPQ": float(sum_iou / pq_denominator) if pq_denominator else 1.0,
        "binary_DQ": float(dq),
        "binary_SQ": float(sq),
        "binary_tp": int(tp),
        "binary_fp": int(fp),
        "binary_fn": int(fn),
        "AJI+": float(aji_intersection / aji_union) if aji_union else 1.0,
        "boundary_F1": float(boundary_f1),
        "boundary_precision": float(boundary_precision),
        "boundary_recall": float(boundary_recall),
        "boundary_tolerance_pixels": int(boundary_tolerance),
    }


def multiclass_r2(true_counts: pd.DataFrame, pred_counts: pd.DataFrame) -> dict:
    per_class = {}
    for name in SHORT_COUNT_COLUMNS:
        y_true = true_counts[name].to_numpy(dtype=float)
        y_pred = pred_counts[name].to_numpy(dtype=float)
        per_class[name] = float(r2_score(y_true, y_pred)) if np.var(y_true) > 0 else float("nan")
    values = [x for x in per_class.values() if np.isfinite(x)]
    return {"R2": float(np.mean(values)), "per_class": per_class}
