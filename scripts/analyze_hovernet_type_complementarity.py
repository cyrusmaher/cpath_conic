#!/usr/bin/env python3
"""Validation-select type-probability blends while keeping control geometry fixed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS
from cpath_conic.data import central_crop_counts
from cpath_conic.metrics import instance_type_confusion, multiclass_pq_plus, multiclass_r2, pq_stats
from scripts.analyze_hovernet_count_complementarity import zero_truth_overcount_summary


def overlap_pairs_and_counts(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return positive-ID overlap pairs and intersections without pixel sorting."""
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    overlap = (first > 0) & (second > 0)
    if not overlap.any():
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    first_values = first[overlap]
    second_values = second[overlap]
    first_size = int(first_values.max()) + 1
    second_size = int(second_values.max()) + 1
    if first_size * second_size <= 5_000_000:
        joint = np.bincount(
            first_values * second_size + second_values,
            minlength=first_size * second_size,
        ).reshape(first_size, second_size)
        first_ids, second_ids = np.nonzero(joint[1:, 1:])
        first_ids += 1
        second_ids += 1
        pairs = np.stack([first_ids, second_ids], axis=1).astype(np.int64)
        return pairs, joint[first_ids, second_ids].astype(np.int64)
    return np.unique(np.stack([first_values, second_values], axis=1), axis=0, return_counts=True)


def match_instances(control: np.ndarray, candidate: np.ndarray, threshold: float = 0.5) -> dict[int, int]:
    """Greedily one-to-one match decoded instances by IoU, as in PQ."""
    control_ids, control_area = np.unique(control[control > 0], return_counts=True)
    candidate_ids, candidate_area = np.unique(candidate[candidate > 0], return_counts=True)
    control_area = dict(zip(control_ids.astype(int), control_area.astype(int)))
    candidate_area = dict(zip(candidate_ids.astype(int), candidate_area.astype(int)))
    candidates: list[tuple[float, int, int]] = []
    pairs, intersections = overlap_pairs_and_counts(control, candidate)
    if len(pairs):
        for pair, intersection in zip(pairs, intersections):
            control_id, candidate_id = map(int, pair)
            union = control_area[control_id] + candidate_area[candidate_id] - int(intersection)
            iou = float(intersection / union) if union else 0.0
            if iou > threshold:
                candidates.append((iou, control_id, candidate_id))
    candidates.sort(reverse=True)
    matched_control: set[int] = set()
    matched_candidate: set[int] = set()
    matches = {}
    for _, control_id, candidate_id in candidates:
        if control_id not in matched_control and candidate_id not in matched_candidate:
            matches[control_id] = candidate_id
            matched_control.add(control_id)
            matched_candidate.add(candidate_id)
    return matches


def probability_lookup(artifact: np.lib.npyio.NpzFile) -> dict[tuple[int, int], np.ndarray]:
    return {
        (int(patch_id), int(instance_id)): probability.astype(np.float64)
        for patch_id, instance_id, probability in zip(
            artifact["probability_patch_ids"], artifact["probability_instance_ids"], artifact["class_probs"]
        )
    }


def decoder_aligned_probability(probability: np.ndarray, decoded_class: int) -> np.ndarray:
    """Minimally make a pooled probability agree with the decoder endpoint."""
    values = np.asarray(probability, dtype=np.float64).copy()
    if values.shape != (len(CLASS_NAMES),) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("instance probability must contain six finite non-negative values")
    if decoded_class < 1 or decoded_class > len(CLASS_NAMES):
        raise ValueError("decoded class must be a CoNIC class ID in 1..6")
    total = float(values.sum())
    values = values / total if total > 0 else np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES))
    target = decoded_class - 1
    if int(np.argmax(values)) != target:
        values[target] = float(values.max()) + np.finfo(np.float64).eps
        values /= values.sum()
    return values


def serialized_assignment_probability(probability: np.ndarray, assigned_class: int) -> np.ndarray:
    """Serialize tooltip probabilities without changing selection arithmetic."""
    values = np.asarray(probability, dtype=np.float32).copy()
    total = float(values.sum())
    values = values / total if total > 0 else np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES), dtype=np.float32)
    if assigned_class > 0 and int(np.argmax(values)) + 1 != assigned_class:
        target = assigned_class - 1
        values[target] = values.max() + np.float32(1.0e-6)
        values /= values.sum()
    return values.astype(np.float32)


def decoded_instance_class(prediction: np.ndarray, instance_id: int) -> int:
    values = prediction[..., 1][prediction[..., 0] == instance_id]
    values = values[(values >= 1) & (values <= len(CLASS_NAMES))]
    if not len(values):
        # HoVer-Net can decode foreground geometry while assigning background
        # type 0.  Keep that explicit dustbin state unless a matched candidate
        # contributes a valid decoded cell type.
        return 0
    return int(np.bincount(values.astype(np.int64), minlength=len(CLASS_NAMES) + 1)[1:].argmax()) + 1


def decoded_class_lookup(prediction: np.ndarray) -> dict[int, int]:
    instance_ids = np.unique(prediction[..., 0][prediction[..., 0] > 0]).astype(np.int64)
    classes = instance_classes_for_ids(prediction[..., 0], prediction[..., 1], instance_ids)
    return {int(instance_id): int(class_id) for instance_id, class_id in zip(instance_ids, classes, strict=True)}


def patch_pq_statistics(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Compute additive CoNIC PQ statistics once per patch and class."""
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 4 or truth.shape[-1] != 2:
        raise ValueError("truth and prediction must be matching NxHxWx2 maps")
    totals = np.zeros((len(truth), len(CLASS_NAMES), 4), dtype=np.float64)
    for patch_index, (true_patch, predicted_patch) in enumerate(zip(truth, prediction, strict=True)):
        for class_index in range(len(CLASS_NAMES)):
            class_id = class_index + 1
            totals[patch_index, class_index] = pq_stats(
                true_patch[..., 0] * (true_patch[..., 1] == class_id),
                predicted_patch[..., 0] * (predicted_patch[..., 1] == class_id),
            )
    return totals


def pq_summary_from_statistics(statistics: np.ndarray) -> dict:
    """Aggregate additive patch/class statistics using exact CoNIC formulas."""
    values = np.asarray(statistics, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (len(CLASS_NAMES), 4):
        raise ValueError("PQ statistics must have shape patches-by-six-by-four")
    totals = values.sum(axis=0)
    per_class = {}
    for name, (tp, fp, fn, sum_iou) in zip(CLASS_NAMES, totals, strict=True):
        denominator = tp + 0.5 * fp + 0.5 * fn
        dq = tp / (denominator + 1.0e-6)
        sq = sum_iou / (tp + 1.0e-6)
        per_class[name] = {
            "pq": float(dq * sq),
            "dq": float(dq),
            "sq": float(sq),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "sum_iou": float(sum_iou),
        }
    return {
        "mPQ+": float(np.mean([row["pq"] for row in per_class.values()])),
        "mDQ+": float(np.mean([row["dq"] for row in per_class.values()])),
        "mSQ+": float(np.mean([row["sq"] for row in per_class.values()])),
        "per_class": per_class,
    }


def instance_classes_for_ids(instance_map: np.ndarray, class_map: np.ndarray, instance_ids: np.ndarray) -> np.ndarray:
    """Return the single decoded class (including dustbin 0) for each ID."""
    instance_ids = np.asarray(instance_ids, dtype=np.int64)
    if not len(instance_ids):
        return np.empty(0, dtype=np.int16)
    valid = instance_map > 0
    observed_ids = instance_map[valid].astype(np.int64)
    positions = np.searchsorted(instance_ids, observed_ids)
    if np.any(positions >= len(instance_ids)) or not np.array_equal(instance_ids[positions], observed_ids):
        raise ValueError("instance map contains an ID absent from the supplied ID list")
    observed_classes = class_map[valid].astype(np.int64)
    if np.any((observed_classes < 0) | (observed_classes > len(CLASS_NAMES))):
        raise ValueError("class map contains an invalid CoNIC class")
    votes = np.bincount(
        positions * (len(CLASS_NAMES) + 1) + observed_classes,
        minlength=len(instance_ids) * (len(CLASS_NAMES) + 1),
    ).reshape(len(instance_ids), len(CLASS_NAMES) + 1)
    if np.any(np.sum(votes > 0, axis=1) != 1):
        raise ValueError("fixed-geometry PQ requires one decoded class per instance")
    return votes.argmax(axis=1).astype(np.int16)


def fixed_geometry_pq_cache(truth: np.ndarray, control_prediction: np.ndarray) -> list[dict]:
    """Cache IoU>0.5 geometry matches shared by every type-blend weight."""
    cache = []
    for true_patch, predicted_patch in zip(truth, control_prediction, strict=True):
        true_inst = true_patch[..., 0].astype(np.int64)
        pred_inst = predicted_patch[..., 0].astype(np.int64)
        true_ids, true_areas = np.unique(true_inst[true_inst > 0], return_counts=True)
        pred_ids, pred_areas = np.unique(pred_inst[pred_inst > 0], return_counts=True)
        true_classes = instance_classes_for_ids(true_inst, true_patch[..., 1], true_ids)
        true_count_by_class = np.bincount(
            true_classes[(true_classes >= 1) & (true_classes <= len(CLASS_NAMES))],
            minlength=len(CLASS_NAMES) + 1,
        )[1:].astype(np.int32)
        match_true_class = []
        match_pred_index = []
        match_iou = []
        pairs, intersections = overlap_pairs_and_counts(true_inst, pred_inst)
        if len(pairs):
            true_positions = np.searchsorted(true_ids, pairs[:, 0])
            pred_positions = np.searchsorted(pred_ids, pairs[:, 1])
            unions = true_areas[true_positions] + pred_areas[pred_positions] - intersections
            ious = intersections / np.maximum(unions, 1)
            keep = ious > 0.5
            match_true_class = true_classes[true_positions[keep]].astype(np.int16)
            match_pred_index = pred_positions[keep].astype(np.int32)
            match_iou = ious[keep].astype(np.float64)
            if len(np.unique(pairs[keep, 0])) != int(keep.sum()) or len(np.unique(pairs[keep, 1])) != int(keep.sum()):
                raise RuntimeError("IoU>0.5 geometry matches were unexpectedly not one-to-one")
        cache.append({
            "pred_ids": pred_ids.astype(np.int64),
            "true_count_by_class": true_count_by_class,
            "match_true_class": np.asarray(match_true_class, dtype=np.int16),
            "match_pred_index": np.asarray(match_pred_index, dtype=np.int32),
            "match_iou": np.asarray(match_iou, dtype=np.float64),
        })
    return cache


def fixed_geometry_patch_pq_statistics(cache: list[dict], prediction: np.ndarray) -> np.ndarray:
    """Evaluate typed PQ from cached geometry and candidate instance classes."""
    if len(cache) != len(prediction):
        raise ValueError("fixed-geometry cache and prediction patch counts differ")
    statistics = np.zeros((len(prediction), len(CLASS_NAMES), 4), dtype=np.float64)
    for patch_index, (item, patch) in enumerate(zip(cache, prediction, strict=True)):
        pred_ids = item["pred_ids"]
        current_ids = np.unique(patch[..., 0][patch[..., 0] > 0]).astype(np.int64)
        if not np.array_equal(pred_ids, current_ids):
            raise RuntimeError("type blend changed fixed prediction geometry")
        pred_classes = instance_classes_for_ids(patch[..., 0], patch[..., 1], pred_ids)
        pred_count_by_class = np.bincount(
            pred_classes[(pred_classes >= 1) & (pred_classes <= len(CLASS_NAMES))],
            minlength=len(CLASS_NAMES) + 1,
        )[1:]
        tp = np.zeros(len(CLASS_NAMES), dtype=np.int32)
        sum_iou = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        if len(item["match_iou"]):
            matched_pred_class = pred_classes[item["match_pred_index"]]
            matched_true_class = item["match_true_class"]
            correct = (matched_true_class == matched_pred_class) & (matched_true_class > 0)
            if correct.any():
                classes = matched_true_class[correct].astype(np.int64) - 1
                tp += np.bincount(classes, minlength=len(CLASS_NAMES)).astype(np.int32)
                np.add.at(sum_iou, classes, item["match_iou"][correct])
        statistics[patch_index, :, 0] = tp
        statistics[patch_index, :, 1] = pred_count_by_class - tp
        statistics[patch_index, :, 2] = item["true_count_by_class"] - tp
        statistics[patch_index, :, 3] = sum_iou
    return statistics


def fixed_geometry_lookup_pq_statistics(cache: list[dict], class_lookups: list[dict[int, int]]) -> np.ndarray:
    """Evaluate typed PQ directly from compact per-instance class assignments."""
    if len(cache) != len(class_lookups):
        raise ValueError("fixed-geometry cache and class lookup counts differ")
    statistics = np.zeros((len(cache), len(CLASS_NAMES), 4), dtype=np.float64)
    for patch_index, (item, lookup) in enumerate(zip(cache, class_lookups, strict=True)):
        pred_ids = item["pred_ids"]
        if set(map(int, pred_ids)) != set(lookup):
            raise RuntimeError("type lookup changed fixed prediction geometry")
        pred_classes = np.asarray([lookup[int(instance_id)] for instance_id in pred_ids], dtype=np.int16)
        pred_count_by_class = np.bincount(
            pred_classes[(pred_classes >= 1) & (pred_classes <= len(CLASS_NAMES))],
            minlength=len(CLASS_NAMES) + 1,
        )[1:]
        tp = np.zeros(len(CLASS_NAMES), dtype=np.int32)
        sum_iou = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        if len(item["match_iou"]):
            matched_pred_class = pred_classes[item["match_pred_index"]]
            matched_true_class = item["match_true_class"]
            correct = (matched_true_class == matched_pred_class) & (matched_true_class > 0)
            if correct.any():
                classes = matched_true_class[correct].astype(np.int64) - 1
                tp += np.bincount(classes, minlength=len(CLASS_NAMES)).astype(np.int32)
                np.add.at(sum_iou, classes, item["match_iou"][correct])
        statistics[patch_index, :, 0] = tp
        statistics[patch_index, :, 1] = pred_count_by_class - tp
        statistics[patch_index, :, 2] = item["true_count_by_class"] - tp
        statistics[patch_index, :, 3] = sum_iou
    return statistics


def central_counts_from_lookup(
    instance_map: np.ndarray,
    class_lookup: dict[int, int],
    margin: int = 16,
) -> np.ndarray:
    """Count central-crop instances without materializing a class map."""
    central_ids = np.unique(instance_map[margin:-margin, margin:-margin])
    central_ids = central_ids[central_ids > 0].astype(np.int64)
    classes = np.asarray([class_lookup[int(instance_id)] for instance_id in central_ids], dtype=np.int16)
    return np.bincount(
        classes[(classes >= 1) & (classes <= len(CLASS_NAMES))],
        minlength=len(CLASS_NAMES) + 1,
    )[1:].astype(np.int32)


def blended_class_lookup_for_patch(
    patch_id: int,
    control_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    control_probabilities: dict[tuple[int, int], np.ndarray],
    candidate_probabilities: dict[tuple[int, int], np.ndarray],
    candidate_weight: float,
    instance_matches: dict[int, int] | None = None,
    control_decoded_classes: dict[int, int] | None = None,
    candidate_decoded_classes: dict[int, int] | None = None,
    assignment_probabilities: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, int], int]:
    """Return compact relabeled control instances without changing geometry."""
    control_instances = control_prediction[..., 0].astype(np.int32)
    control_classes = control_decoded_classes or decoded_class_lookup(control_prediction)
    if candidate_weight == 0.0:
        if assignment_probabilities is not None:
            for instance_id, control_class in control_classes.items():
                probability = control_probabilities.get((patch_id, instance_id))
                if probability is None:
                    probability = (
                        np.eye(len(CLASS_NAMES))[control_class - 1]
                        if control_class > 0
                        else np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES))
                    )
                elif control_class > 0:
                    probability = decoder_aligned_probability(probability, control_class)
                assignment_probabilities[instance_id] = serialized_assignment_probability(
                    probability, control_class
                )
        return dict(control_classes), 0
    matches = (
        match_instances(control_instances, candidate_prediction[..., 0])
        if instance_matches is None
        else instance_matches
    )
    candidate_classes = candidate_decoded_classes or decoded_class_lookup(candidate_prediction)
    assignments = {}
    blended = 0
    for instance_id, control_class in control_classes.items():
        control_key = (patch_id, instance_id)
        control_probability = control_probabilities.get(control_key)
        if control_probability is None:
            control_probability = (
                np.eye(len(CLASS_NAMES))[control_class - 1]
                if control_class > 0
                else np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES))
            )
        elif control_class > 0:
            control_probability = decoder_aligned_probability(control_probability, control_class)
        candidate_id = matches.get(instance_id)
        candidate_probability = (
            candidate_probabilities.get((patch_id, candidate_id)) if candidate_id is not None else None
        )
        if candidate_probability is None:
            assignments[instance_id] = control_class
            if assignment_probabilities is not None:
                assignment_probabilities[instance_id] = serialized_assignment_probability(
                    control_probability, control_class
                )
            continue
        candidate_class = candidate_classes.get(candidate_id, 0)
        if candidate_class == 0:
            assignments[instance_id] = control_class
            if assignment_probabilities is not None:
                assignment_probabilities[instance_id] = serialized_assignment_probability(
                    control_probability, control_class
                )
            continue
        candidate_probability = decoder_aligned_probability(candidate_probability, candidate_class)
        probability = (1.0 - candidate_weight) * control_probability + candidate_weight * candidate_probability
        blended += 1
        assignments[instance_id] = int(np.argmax(probability)) + 1
        if assignment_probabilities is not None:
            assignment_probabilities[instance_id] = serialized_assignment_probability(
                probability, assignments[instance_id]
            )
    return assignments, blended


def blended_classes_for_patch(
    patch_id: int,
    control_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    control_probabilities: dict[tuple[int, int], np.ndarray],
    candidate_probabilities: dict[tuple[int, int], np.ndarray],
    candidate_weight: float,
    instance_matches: dict[int, int] | None = None,
    control_decoded_classes: dict[int, int] | None = None,
    candidate_decoded_classes: dict[int, int] | None = None,
) -> tuple[np.ndarray, int]:
    """Relabel control instances; never alter their pixels or instance IDs."""
    assignments, blended = blended_class_lookup_for_patch(
        patch_id,
        control_prediction,
        candidate_prediction,
        control_probabilities,
        candidate_probabilities,
        candidate_weight,
        instance_matches=instance_matches,
        control_decoded_classes=control_decoded_classes,
        candidate_decoded_classes=candidate_decoded_classes,
    )
    control_instances = control_prediction[..., 0].astype(np.int32)
    class_lut = np.zeros(int(control_instances.max()) + 1, dtype=np.uint8)
    for instance_id, class_id in assignments.items():
        class_lut[instance_id] = class_id
    return class_lut[control_instances], blended


def metric_summary(
    truth: np.ndarray,
    prediction: np.ndarray,
    true_counts: np.ndarray,
    *,
    include_confusion: bool = True,
    predicted_counts: np.ndarray | None = None,
    pq_patch_statistics: np.ndarray | None = None,
) -> dict:
    pq = (
        multiclass_pq_plus(truth, prediction)
        if pq_patch_statistics is None
        else pq_summary_from_statistics(pq_patch_statistics)
    )
    if predicted_counts is None:
        predicted_counts = np.asarray(
            [central_crop_counts(patch[..., 0], patch[..., 1]) for patch in prediction], dtype=np.int32
        )
    else:
        predicted_counts = np.asarray(predicted_counts, dtype=np.int32)
    r2 = multiclass_r2(
        pd.DataFrame(true_counts, columns=CLASS_NAMES),
        pd.DataFrame(predicted_counts, columns=CLASS_NAMES),
    )
    residual = predicted_counts.astype(np.float64) - true_counts.astype(np.float64)
    payload = {
        "R2": float(r2["R2"]),
        "mPQ+": float(pq["mPQ+"]),
        "mDQ+": float(pq["mDQ+"]),
        "mSQ+": float(pq["mSQ+"]),
        "per_class_R2": r2["per_class"],
        "per_class_PQ": {name: values["pq"] for name, values in pq["per_class"].items()},
        "per_class_DQ": {name: values["dq"] for name, values in pq["per_class"].items()},
        "per_class_SQ": {name: values["sq"] for name, values in pq["per_class"].items()},
        "per_class_PQ_stats": {
            name: {key: values[key] for key in ("tp", "fp", "fn", "sum_iou")}
            for name, values in pq["per_class"].items()
        },
        "count_error": {
            "mean_signed_error": float(residual.mean()),
            "MAE": float(np.abs(residual).mean()),
            **{
                f"under_error_lt_minus_{threshold}_fraction": float(np.mean(residual < -threshold))
                for threshold in (5, 10, 20)
            },
            **{
                f"over_error_gt_{threshold}_fraction": float(np.mean(residual > threshold))
                for threshold in (5, 10, 20)
            },
            **{
                f"absolute_error_gt_{threshold}_fraction": float(np.mean(np.abs(residual) > threshold))
                for threshold in (5, 10, 20)
            },
        },
        "zero_truth_overcount": zero_truth_overcount_summary(true_counts, predicted_counts),
    }
    if include_confusion:
        payload["instance_type_confusion"] = instance_type_confusion(truth, prediction)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--control-name", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    weights = sorted(set(args.weights))
    if not weights or min(weights) < 0 or max(weights) > 1:
        parser.error("--weights must be a nonempty subset of [0, 1]")
    if 0 not in weights:
        parser.error("--weights must include 0 to preserve the fixed-geometry control")

    control = np.load(args.control_artifact)
    candidate = np.load(args.candidate_artifact)
    # NPZ members are decompressed on every ``artifact[key]`` access.  Hold
    # each dense map array once; per-patch indexing into the NpzFile would
    # otherwise inflate the same hundreds of megabytes repeatedly.
    control_predictions = control["predictions"]
    candidate_predictions = candidate["predictions"]
    patch_ids = control["patch_ids"].astype(np.int32)
    if not np.array_equal(patch_ids, candidate["patch_ids"]):
        raise ValueError("control and candidate patch IDs differ")
    metadata = pd.read_csv(args.prepared / "metadata.csv").set_index("patch_id").loc[patch_ids]
    if metadata.split.eq("test").any():
        raise RuntimeError("type-complementarity selection refuses locked-test patches")
    sources = metadata.source.astype(str).to_numpy()
    true_counts = metadata[COUNT_COLUMNS].to_numpy(dtype=np.int32)
    truth = np.zeros_like(control_predictions, dtype=np.int32)
    for index, patch_id in enumerate(patch_ids):
        label = np.load(args.prepared / "labels" / f"{int(patch_id):05d}.npy", allow_pickle=True).item()
        truth[index, ..., 0] = label["inst_map"]
        truth[index, ..., 1] = label["class_map"]

    control_probabilities = probability_lookup(control)
    candidate_probabilities = probability_lookup(candidate)
    geometry_cache = fixed_geometry_pq_cache(truth, control_predictions)
    control_candidate_matches = [
        match_instances(control_patch[..., 0], candidate_patch[..., 0])
        for control_patch, candidate_patch in zip(
            control_predictions, candidate_predictions, strict=True
        )
    ]
    control_decoded_classes = [decoded_class_lookup(patch) for patch in control_predictions]
    candidate_decoded_classes = [decoded_class_lookup(patch) for patch in candidate_predictions]
    rows = []
    for weight in weights:
        class_lookups = []
        blended_instances = 0
        for index, patch_id in enumerate(patch_ids):
            lookup, blended = blended_class_lookup_for_patch(
                int(patch_id), control_predictions[index], candidate_predictions[index],
                control_probabilities, candidate_probabilities, weight,
                instance_matches=control_candidate_matches[index],
                control_decoded_classes=control_decoded_classes[index],
                candidate_decoded_classes=candidate_decoded_classes[index],
            )
            class_lookups.append(lookup)
            blended_instances += blended
        predicted_counts = np.asarray(
            [
                central_counts_from_lookup(patch[..., 0], lookup)
                for patch, lookup in zip(control_predictions, class_lookups, strict=True)
            ],
            dtype=np.int32,
        )
        pq_patch_stats = fixed_geometry_lookup_pq_statistics(geometry_cache, class_lookups)
        # Confusion matrices do not enter weight selection and are expensive
        # to recompute for every candidate.  Materialize one for the final
        # locked recipe/dashboard instead of multiplying it across this grid.
        overall = metric_summary(
            truth,
            control_predictions,
            true_counts,
            include_confusion=False,
            predicted_counts=predicted_counts,
            pq_patch_statistics=pq_patch_stats,
        )
        by_source = {}
        excluded_source = {}
        for source in sorted(set(sources)):
            mask = sources == source
            by_source[source] = metric_summary(
                truth[mask], control_predictions[mask], true_counts[mask], include_confusion=False,
                predicted_counts=predicted_counts[mask], pq_patch_statistics=pq_patch_stats[mask],
            )
            excluded_source[source] = metric_summary(
                truth[~mask], control_predictions[~mask], true_counts[~mask], include_confusion=False,
                predicted_counts=predicted_counts[~mask], pq_patch_statistics=pq_patch_stats[~mask],
            )
        rows.append({
            "candidate_type_weight": weight,
            "matched_control_instances": blended_instances,
            "overall": overall,
            "by_source": by_source,
            "excluding_source": excluded_source,
        })
        print(f"candidate TP weight {weight:g}: mPQ+={overall['mPQ+']:.6f} R2={overall['R2']:.6f}", flush=True)

    selected = max(rows, key=lambda row: (row["overall"]["mPQ+"], -row["candidate_type_weight"]))
    source_excluded_weights = {
        source: max(
            rows,
            key=lambda row: (row["excluding_source"][source]["mPQ+"], -row["candidate_type_weight"]),
        )["candidate_type_weight"]
        for source in sorted(set(sources))
    }
    baseline = next(row for row in rows if row["candidate_type_weight"] == 0)["overall"]
    report = {
        "protocol": "development-validation-only instance-matched type-probability blend on exactly fixed control geometry",
        "evaluation_set": f"{len(patch_ids)}-patch source-group-disjoint development validation",
        "control": args.control_name,
        "candidate": args.candidate_name,
        "selection_metric": "pooled validation mPQ+; ties prefer less candidate weight",
        "geometry": "control NP/HV decoded instances are bitwise unchanged for every candidate weight",
        "selected": selected,
        "selected_delta_vs_control": {
            key: selected["overall"][key] - baseline[key] for key in ("R2", "mPQ+", "mDQ+", "mSQ+")
        },
        "selected_count_error_delta_vs_control": {
            key: selected["overall"]["count_error"][key] - baseline["count_error"][key]
            for key in baseline["count_error"]
        },
        "selected_candidate_weight_excluding_each_source": source_excluded_weights,
        "weight_stable_across_source_exclusions": len(set(source_excluded_weights.values())) == 1,
        "candidates": rows,
        "guardrail": (
            "Reject if pooled mPQ+/mDQ+ fail to improve, mSQ+ falls by more than 0.005, "
            "source-excluded weights are unstable, or a major source has an mPQ+/mDQ+/mSQ+ regression below "
            "-0.01. Directional and supported source/class true-zero count tails are reported for the separate "
            "R2/reliability track and do not veto fixed-geometry mPQ composition."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in (
        "evaluation_set", "selected_delta_vs_control", "selected_count_error_delta_vs_control",
        "selected_candidate_weight_excluding_each_source",
        "weight_stable_across_source_exclusions",
    )}, indent=2))


if __name__ == "__main__":
    main()
