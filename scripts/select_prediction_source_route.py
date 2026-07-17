#!/usr/bin/env python3
"""Validation-select whole-source routing between arbitrary CoNIC prediction maps."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS
from cpath_conic.metrics import multiclass_pq_plus, multiclass_r2


def pooled_pq_from_stats(stats: np.ndarray) -> dict:
    """Convert additive [class, TP/FP/FN/sum-IoU] statistics to pooled PQ."""
    per_class = {}
    for class_name, (tp, fp, fn, sum_iou) in zip(CLASS_NAMES, np.asarray(stats, dtype=np.float64)):
        denominator = tp + 0.5 * fp + 0.5 * fn
        per_class[class_name] = {
            "pq": float(sum_iou / denominator) if denominator else 0.0,
            "dq": float(tp / denominator) if denominator else 0.0,
            "sq": float(sum_iou / tp) if tp else 0.0,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "sum_iou": float(sum_iou),
        }
    return {"mPQ+": float(np.mean([item["pq"] for item in per_class.values()])), "per_class": per_class}


def pq_stats_array(metrics: dict) -> np.ndarray:
    return np.asarray(
        [
            [
                metrics["per_class"][name]["tp"],
                metrics["per_class"][name]["fp"],
                metrics["per_class"][name]["fn"],
                metrics["per_class"][name]["sum_iou"],
            ]
            for name in CLASS_NAMES
        ],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--predictions-a", type=Path, required=True)
    parser.add_argument("--counts-a", type=Path, required=True)
    parser.add_argument("--predictions-b", type=Path, required=True)
    parser.add_argument("--counts-b", type=Path, required=True)
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--selection-split", default="val")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    selected = metadata.loc[metadata.split == args.selection_split].sort_values("patch_id")
    patch_ids = selected.patch_id.to_numpy(dtype=np.int32)
    sources = sorted(selected.source.dropna().unique().tolist())
    predictions = {
        "a": np.load(args.predictions_a, mmap_mode="r"),
        "b": np.load(args.predictions_b, mmap_mode="r"),
    }
    counts = {
        "a": np.load(args.counts_a, mmap_mode="r"),
        "b": np.load(args.counts_b, mmap_mode="r"),
    }
    expected_shape = (len(metadata), 256, 256, 2)
    for name, values in predictions.items():
        if values.shape != expected_shape:
            raise ValueError(f"prediction {name} has shape {values.shape}, expected {expected_shape}")
    for name, values in counts.items():
        if values.shape != (len(metadata), len(CLASS_NAMES)):
            raise ValueError(f"counts {name} have shape {values.shape}")

    truth = np.zeros((len(selected), 256, 256, 2), dtype=np.int32)
    for index, patch_id in enumerate(patch_ids):
        label = np.load(args.prepared / "labels" / f"{int(patch_id):05d}.npy", allow_pickle=True).item()
        truth[index, ..., 0] = label["inst_map"]
        truth[index, ..., 1] = label["class_map"]
    true_counts = selected[COUNT_COLUMNS].to_numpy(dtype=np.float64)

    source_stats: dict[str, dict[str, np.ndarray]] = {method: {} for method in predictions}
    source_metrics: dict[str, dict[str, dict]] = {method: {} for method in predictions}
    for method, method_predictions in predictions.items():
        for source in sources:
            mask = selected.source.to_numpy() == source
            metrics = multiclass_pq_plus(truth[mask], method_predictions[patch_ids[mask]])
            source_stats[method][source] = pq_stats_array(metrics)
            source_metrics[method][source] = {
                "n_patches": int(mask.sum()),
                "mPQ+": metrics["mPQ+"],
                "per_class": metrics["per_class"],
            }

    combinations = []
    source_values = selected.source.to_numpy()
    for route_values in itertools.product(("a", "b"), repeat=len(sources)):
        route = dict(zip(sources, route_values))
        pooled = pooled_pq_from_stats(sum(source_stats[route[source]][source] for source in sources))
        routed_counts = np.asarray(counts["a"][patch_ids]).copy()
        use_b = np.asarray([route[source] == "b" for source in source_values])
        routed_counts[use_b] = counts["b"][patch_ids[use_b]]
        r2 = multiclass_r2(
            pd.DataFrame(true_counts, columns=CLASS_NAMES),
            pd.DataFrame(routed_counts, columns=CLASS_NAMES),
        )
        combinations.append({"route": route, "mPQ+": pooled["mPQ+"], "per_class_pq": pooled["per_class"], **r2})

    winner = max(combinations, key=lambda item: (item["mPQ+"], -sum(value == "b" for value in item["route"].values())))
    endpoints = {}
    for method in ("a", "b"):
        pooled = pooled_pq_from_stats(sum(source_stats[method][source] for source in sources))
        r2 = multiclass_r2(
            pd.DataFrame(true_counts, columns=CLASS_NAMES),
            pd.DataFrame(np.asarray(counts[method][patch_ids]), columns=CLASS_NAMES),
        )
        endpoints[method] = {
            "R2": r2["R2"],
            "per_class_R2": r2["per_class"],
            "mPQ+": pooled["mPQ+"],
            "per_class_pq": pooled["per_class"],
        }
    report = {
        "selection_split": args.selection_split,
        "test_evaluated": False,
        "method_names": {"a": args.name_a, "b": args.name_b},
        "selection_metric": "exact pooled validation mPQ+ from additive source TP/FP/FN/sum-IoU statistics",
        "selected_route": winner["route"],
        "selected": {
            "R2": winner["R2"],
            "per_class_R2": winner["per_class"],
            "mPQ+": winner["mPQ+"],
            "per_class_pq": winner["per_class_pq"],
        },
        "endpoints": endpoints,
        "by_source": {
            source: {method: source_metrics[method][source] for method in ("a", "b")}
            for source in sources
        },
        "grid": [{"route": item["route"], "R2": item["R2"], "mPQ+": item["mPQ+"]} for item in combinations],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"route": winner["route"], "R2": winner["R2"], "mPQ+": winner["mPQ+"]}, indent=2))


if __name__ == "__main__":
    main()
