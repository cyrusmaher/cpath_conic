#!/usr/bin/env python
"""Validation-select per-class top-1 confidence rejection for fixed masks."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES
from cpath_conic.experiment_metrics import evaluate_fixed_masks


def apply_class_thresholds(probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    assignments = probabilities.argmax(axis=1).astype(np.int8) + 1
    top = probabilities.max(axis=1)
    reject = top < thresholds[assignments - 1]
    assignments[reject] = 0
    return assignments


def compact(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "predicted_counts"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--max-threshold", type=float, default=0.95)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    features = np.load(args.features)
    probability_data = np.load(args.probabilities)
    cache = np.load(args.cache)
    patch_ids = features["patch_ids"].astype(np.int32)
    instance_ids = features["instance_ids"].astype(np.int32)
    if not np.array_equal(patch_ids, probability_data["patch_ids"]) or not np.array_equal(instance_ids, probability_data["instance_ids"]):
        raise ValueError("Feature and probability IDs are not aligned")
    probabilities = probability_data["class_probs"].astype(np.float32)
    labels = features["labels"].astype(np.int8)
    ious = features["ious"].astype(np.float32)
    central = cache["central"].astype(bool)
    gt_full_counts = cache["gt_full_counts"].astype(np.int32)
    if len(central) != len(probabilities):
        raise ValueError("Cache and probability records have different lengths")
    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    grid = np.arange(0.0, args.max_threshold + args.step / 2, args.step)
    thresholds = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    rows = []

    def evaluate(values: np.ndarray, split: str) -> dict:
        assignments = apply_class_thresholds(probabilities, values)
        return evaluate_fixed_masks(assignments, labels, ious, patch_ids, central, metadata, gt_full_counts, split)

    for class_index, class_name in enumerate(CLASS_NAMES):
        candidates = []
        for threshold in grid:
            values = np.zeros(len(CLASS_NAMES), dtype=np.float64)
            values[class_index] = threshold
            metrics = evaluate(values, "val")
            row = {
                "class": class_name,
                "threshold": float(threshold),
                "val_class_PQ": float(metrics["per_class_pq"][class_name]["pq"]),
                "val_class_DQ": float(metrics["per_class_pq"][class_name]["dq"]),
                "val_class_TP": int(metrics["per_class_pq"][class_name]["tp"]),
                "val_class_FP": int(metrics["per_class_pq"][class_name]["fp"]),
                "val_class_FN": int(metrics["per_class_pq"][class_name]["fn"]),
            }
            rows.append(row)
            candidates.append(row)
        best = max(candidates, key=lambda row: (row["val_class_PQ"], -row["threshold"]))
        thresholds[class_index] = best["threshold"]

    raw_thresholds = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    raw_val = evaluate(raw_thresholds, "val")
    selected_val = evaluate(thresholds, "val")
    raw_test = evaluate(raw_thresholds, "test")
    selected_test = evaluate(thresholds, "test")
    val_ids = metadata.loc[metadata.split == "val", "patch_id"].to_numpy(dtype=np.int32)
    on_val = np.isin(patch_ids, val_ids)
    raw_assignments = probabilities.argmax(axis=1).astype(np.int8) + 1
    top = probabilities.max(axis=1)
    diagnostics = {}
    for class_id, class_name in enumerate(CLASS_NAMES, start=1):
        predicted = on_val & (raw_assignments == class_id)
        matched = predicted & (labels == class_id) & (ious > 0.5)
        unmatched = predicted & ~matched
        diagnostics[class_name] = {
            "validation_predictions": int(predicted.sum()),
            "matched_predictions": int(matched.sum()),
            "unmatched_predictions": int(unmatched.sum()),
            "matched_median_top1": float(np.median(top[matched])) if matched.any() else None,
            "unmatched_median_top1": float(np.median(top[unmatched])) if unmatched.any() else None,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "selection_split": "val",
        "rule": "Reject a top-1 class prediction when its raw top-1 probability is below that class's validation-PQ-selected threshold.",
        "grid": {"step": args.step, "max_threshold": args.max_threshold},
        "selected_thresholds": {name: float(value) for name, value in zip(CLASS_NAMES, thresholds)},
        "raw_validation": compact(raw_val),
        "selected_validation": compact(selected_val),
        "raw_test": compact(raw_test),
        "selected_test": compact(selected_test),
        "diagnostics": diagnostics,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "thresholds": report["selected_thresholds"],
        "raw_validation": {key: report["raw_validation"][key] for key in ("R2", "mPQ+", "rejected_fraction")},
        "selected_validation": {key: report["selected_validation"][key] for key in ("R2", "mPQ+", "rejected_fraction")},
        "raw_test": {key: report["raw_test"][key] for key in ("R2", "mPQ+", "rejected_fraction")},
        "selected_test": {key: report["selected_test"][key] for key in ("R2", "mPQ+", "rejected_fraction")},
    }, indent=2))


if __name__ == "__main__":
    main()
