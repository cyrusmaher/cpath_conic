#!/usr/bin/env python3
"""Decompose HoVer DQ failures into class-agnostic detection and typing errors."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES
from cpath_conic.data import load_metadata


def instance_classes(instance_map: np.ndarray, class_map: np.ndarray, instance_ids: np.ndarray) -> np.ndarray:
    classes = np.zeros(len(instance_ids), dtype=np.int16)
    for index, instance_id in enumerate(instance_ids):
        values = class_map[instance_map == instance_id]
        values = values[values > 0]
        if values.size:
            classes[index] = int(np.bincount(values.astype(np.int64)).argmax())
    return classes


def match_patch(truth: np.ndarray, prediction: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt_instances, gt_classes_map = truth[..., 0].astype(np.int64), truth[..., 1].astype(np.int16)
    pred_instances, pred_classes_map = prediction[..., 0].astype(np.int64), prediction[..., 1].astype(np.int16)
    gt_ids = np.unique(gt_instances)
    pred_ids = np.unique(pred_instances)
    gt_ids, pred_ids = gt_ids[gt_ids > 0], pred_ids[pred_ids > 0]
    gt_classes = instance_classes(gt_instances, gt_classes_map, gt_ids)
    pred_classes = instance_classes(pred_instances, pred_classes_map, pred_ids)
    if not gt_ids.size or not pred_ids.size:
        return np.empty((0, 2), dtype=np.int64), gt_classes, pred_classes, np.empty(0, dtype=np.float64)

    pred_stride = int(pred_instances.max()) + 1
    pair_codes, intersections = np.unique(
        gt_instances.astype(np.int64) * pred_stride + pred_instances.astype(np.int64),
        return_counts=True,
    )
    gt_pair_ids, pred_pair_ids = pair_codes // pred_stride, pair_codes % pred_stride
    valid = (gt_pair_ids > 0) & (pred_pair_ids > 0)
    gt_pair_ids, pred_pair_ids, intersections = gt_pair_ids[valid], pred_pair_ids[valid], intersections[valid]
    gt_areas = np.bincount(gt_instances.ravel())
    pred_areas = np.bincount(pred_instances.ravel())
    unions = gt_areas[gt_pair_ids] + pred_areas[pred_pair_ids] - intersections
    ious = intersections / np.maximum(unions, 1)
    keep = ious > threshold
    gt_lookup = {int(value): index for index, value in enumerate(gt_ids)}
    pred_lookup = {int(value): index for index, value in enumerate(pred_ids)}
    matches = np.asarray(
        [[gt_lookup[int(gt_id)], pred_lookup[int(pred_id)]] for gt_id, pred_id in zip(gt_pair_ids[keep], pred_pair_ids[keep])],
        dtype=np.int64,
    )
    if matches.size == 0:
        matches = np.empty((0, 2), dtype=np.int64)
    # At IoU > 0.5, disjoint instance maps imply a unique one-to-one match.
    if len(np.unique(matches[:, 0])) != len(matches) or len(np.unique(matches[:, 1])) != len(matches):
        raise AssertionError("IoU>0.5 matching unexpectedly produced a non-unique pair")
    return matches, gt_classes, pred_classes, ious[keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-plot", type=Path, required=True)
    args = parser.parse_args()

    metadata = load_metadata(args.prepared)
    patch_ids = metadata.loc[metadata.split == args.split, "patch_id"].sort_values().to_numpy(dtype=np.int32)
    predictions = np.load(args.predictions, mmap_mode="r")
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    missed = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    spurious = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    matched_ious: list[float] = []
    raw_match_count = 0
    raw_missed_count = 0
    raw_spurious_count = 0
    invalid_matched_class_count = 0
    invalid_missed_gt_class_count = 0
    invalid_spurious_pred_class_count = 0

    for patch_id in patch_ids:
        label = np.load(args.prepared / "labels" / f"{int(patch_id):05d}.npy", allow_pickle=True).item()
        truth = np.stack([label["inst_map"], label["class_map"]], axis=-1)
        matches, gt_classes, pred_classes, ious = match_patch(truth, predictions[int(patch_id)], args.iou_threshold)
        matched_gt = set(matches[:, 0].tolist())
        matched_pred = set(matches[:, 1].tolist())
        raw_match_count += len(matches)
        raw_missed_count += len(gt_classes) - len(matched_gt)
        raw_spurious_count += len(pred_classes) - len(matched_pred)
        for gt_index, pred_index in matches:
            gt_class, pred_class = int(gt_classes[gt_index]), int(pred_classes[pred_index])
            if 1 <= gt_class <= len(CLASS_NAMES) and 1 <= pred_class <= len(CLASS_NAMES):
                confusion[gt_class - 1, pred_class - 1] += 1
            else:
                invalid_matched_class_count += 1
        for gt_index, gt_class in enumerate(gt_classes):
            if gt_index not in matched_gt and 1 <= gt_class <= len(CLASS_NAMES):
                missed[int(gt_class) - 1] += 1
            elif gt_index not in matched_gt:
                invalid_missed_gt_class_count += 1
        for pred_index, pred_class in enumerate(pred_classes):
            if pred_index not in matched_pred and 1 <= pred_class <= len(CLASS_NAMES):
                spurious[int(pred_class) - 1] += 1
            elif pred_index not in matched_pred:
                invalid_spurious_pred_class_count += 1
        matched_ious.extend(ious.tolist())

    rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        detected = int(confusion[class_index].sum())
        correct = int(confusion[class_index, class_index])
        gt_total = detected + int(missed[class_index])
        predicted_total = int(confusion[:, class_index].sum()) + int(spurious[class_index])
        rows.append(
            {
                "class": class_name,
                "gt_instances": gt_total,
                "predicted_instances": predicted_total,
                "class_agnostic_matched_gt": detected,
                "correctly_typed_matches": correct,
                "missed_gt": int(missed[class_index]),
                "spurious_predictions": int(spurious[class_index]),
                "detection_recall": detected / gt_total if gt_total else None,
                "typing_accuracy_given_detection": correct / detected if detected else None,
                "typed_recall": correct / gt_total if gt_total else None,
            }
        )

    matched = int(confusion.sum())
    report = {
        "split": args.split,
        "n_patches": int(len(patch_ids)),
        "iou_threshold_strictly_greater_than": args.iou_threshold,
        "confusion_rows_gt_columns_prediction": confusion.tolist(),
        "missed_gt_by_class": dict(zip(CLASS_NAMES, missed.tolist())),
        "spurious_prediction_by_class": dict(zip(CLASS_NAMES, spurious.tolist())),
        "overall": {
            "raw_binary_matches": raw_match_count,
            "raw_binary_missed_gt": raw_missed_count,
            "raw_binary_spurious_predictions": raw_spurious_count,
            "valid_six_class_matches": matched,
            "valid_six_class_missed_gt": int(missed.sum()),
            "valid_six_class_spurious_predictions": int(spurious.sum()),
            "invalid_or_zero_class_matches": invalid_matched_class_count,
            "invalid_or_zero_class_missed_gt": invalid_missed_gt_class_count,
            "invalid_or_zero_class_spurious_predictions": invalid_spurious_pred_class_count,
            "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else None,
            "typing_accuracy_given_detection": float(np.trace(confusion) / matched) if matched else None,
        },
        "per_class": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    matrix = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    image = axes[0].imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    axes[0].set(xticks=np.arange(len(CLASS_NAMES)), yticks=np.arange(len(CLASS_NAMES)),
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="predicted type", ylabel="GT type",
                title="Type confusion among class-agnostic IoU>0.5 matches")
    axes[0].tick_params(axis="x", labelrotation=35)
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            axes[0].text(column, row, f"{matrix[row, column]:.1%}\n({confusion[row, column]:,})",
                         ha="center", va="center", fontsize=7,
                         color="white" if matrix[row, column] > 0.55 else "black")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    positions = np.arange(len(CLASS_NAMES))
    detection = [row["detection_recall"] for row in rows]
    typing = [row["typing_accuracy_given_detection"] for row in rows]
    typed_recall = [row["typed_recall"] for row in rows]
    width = 0.25
    axes[1].bar(positions - width, detection, width, label="detection recall")
    axes[1].bar(positions, typing, width, label="typing accuracy | detected")
    axes[1].bar(positions + width, typed_recall, width, label="correctly typed recall")
    axes[1].set(title="Where typed DQ is lost", ylabel="fraction", ylim=(0, 1), xticks=positions, xticklabels=CLASS_NAMES)
    axes[1].tick_params(axis="x", labelrotation=35)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False)
    fig.suptitle("HoVer-Net validation detection versus typing decomposition")
    fig.tight_layout()
    args.out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_plot, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
