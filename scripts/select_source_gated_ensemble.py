#!/usr/bin/env python
"""Validation-select whole-source routing between two fixed-mask methods."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS
from cpath_conic.experiment_metrics import evaluate_fixed_masks
from cpath_conic.metrics import multiclass_r2


def pooled_pq(source_metrics: list[dict]) -> dict:
    per_class = {}
    for name in CLASS_NAMES:
        tp = sum(item["per_class_pq"][name]["tp"] for item in source_metrics)
        fp = sum(item["per_class_pq"][name]["fp"] for item in source_metrics)
        fn = sum(item["per_class_pq"][name]["fn"] for item in source_metrics)
        sum_iou = sum(item["per_class_pq"][name]["sum_iou"] for item in source_metrics)
        denominator = tp + 0.5 * fp + 0.5 * fn
        per_class[name] = {
            "pq": sum_iou / denominator if denominator else 0.0,
            "dq": tp / denominator if denominator else 0.0,
            "sq": sum_iou / tp if tp else 0.0,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "sum_iou": float(sum_iou),
        }
    return {"mPQ+": float(np.mean([value["pq"] for value in per_class.values()])), "per_class_pq": per_class}


def load_method(features_path: Path, probabilities_path: Path, cache_path: Path) -> dict:
    features = np.load(features_path)
    probabilities = np.load(probabilities_path)
    cache = np.load(cache_path)
    patch_ids = features["patch_ids"].astype(np.int32)
    instance_ids = features["instance_ids"].astype(np.int32)
    if not np.array_equal(patch_ids, probabilities["patch_ids"]) or not np.array_equal(instance_ids, probabilities["instance_ids"]):
        raise ValueError(f"Feature/probability IDs are not aligned for {probabilities_path}")
    if len(cache["central"]) != len(patch_ids):
        raise ValueError(f"Cache records are not aligned for {cache_path}")
    return {
        "assignments": probabilities["class_probs"].argmax(axis=1).astype(np.int8) + 1,
        "labels": features["labels"].astype(np.int8),
        "ious": features["ious"].astype(np.float32),
        "patch_ids": patch_ids,
        "central": cache["central"].astype(bool),
        "gt_full_counts": cache["gt_full_counts"].astype(np.int32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    for prefix in ("a", "b"):
        parser.add_argument(f"--{prefix}-features", type=Path, required=True)
        parser.add_argument(f"--{prefix}-probabilities", type=Path, required=True)
        parser.add_argument(f"--{prefix}-cache", type=Path, required=True)
        parser.add_argument(f"--name-{prefix}", default=prefix.upper())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    sources = sorted(metadata.source.dropna().unique().tolist())
    methods = {
        prefix: load_method(
            getattr(args, f"{prefix}_features"),
            getattr(args, f"{prefix}_probabilities"),
            getattr(args, f"{prefix}_cache"),
        )
        for prefix in ("a", "b")
    }
    split_reports = {}
    for split in ("val", "test"):
        split_rows = metadata.loc[metadata.split == split].sort_values("patch_id")
        true_counts = split_rows[COUNT_COLUMNS].to_numpy(dtype=np.float64)
        method_metrics = {}
        source_metrics = {}
        for prefix, method in methods.items():
            method_metrics[prefix] = evaluate_fixed_masks(
                method["assignments"], method["labels"], method["ious"], method["patch_ids"], method["central"],
                metadata, method["gt_full_counts"], split,
            )
            source_metrics[prefix] = {}
            for source in sources:
                source_metadata = metadata.loc[metadata.source == source].copy()
                source_metrics[prefix][source] = evaluate_fixed_masks(
                    method["assignments"], method["labels"], method["ious"], method["patch_ids"], method["central"],
                    source_metadata, method["gt_full_counts"], split,
                )
        combinations = []
        for route_bits in itertools.product(("a", "b"), repeat=len(sources)):
            route = dict(zip(sources, route_bits))
            pq = pooled_pq([source_metrics[route[source]][source] for source in sources])
            counts = method_metrics["a"]["predicted_counts"].copy()
            use_b = split_rows.source.map(route).to_numpy() == "b"
            counts[use_b] = method_metrics["b"]["predicted_counts"][use_b]
            r2 = multiclass_r2(
                pd.DataFrame(true_counts, columns=CLASS_NAMES),
                pd.DataFrame(counts, columns=CLASS_NAMES),
            )
            combinations.append({
                "route": route,
                "mPQ+": pq["mPQ+"],
                "per_class_pq": pq["per_class_pq"],
                "R2": r2["R2"],
                "per_class_R2": r2["per_class"],
            })
        split_reports[split] = {
            "combinations": combinations,
            "endpoints": {
                prefix: {key: value for key, value in method_metrics[prefix].items() if key != "predicted_counts"}
                for prefix in ("a", "b")
            },
            "by_source": {
                source: {
                    prefix: {"R2": source_metrics[prefix][source]["R2"], "mPQ+": source_metrics[prefix][source]["mPQ+"]}
                    for prefix in ("a", "b")
                }
                for source in sources
            },
        }
    selected = max(
        split_reports["val"]["combinations"],
        key=lambda item: (item["mPQ+"], -sum(value == "b" for value in item["route"].values())),
    )
    selected_test = next(item for item in split_reports["test"]["combinations"] if item["route"] == selected["route"])
    report = {
        "selection_split": "val",
        "selection_metric": "exact pooled mPQ+ from additive per-source TP/FP/FN/sum-IoU statistics",
        "method_names": {"a": args.name_a, "b": args.name_b},
        "selected_route": selected["route"],
        "selected_validation": {key: selected[key] for key in ("R2", "mPQ+", "per_class_R2", "per_class_pq")},
        "selected_test": {key: selected_test[key] for key in ("R2", "mPQ+", "per_class_R2", "per_class_pq")},
        "endpoints": {split: split_reports[split]["endpoints"] for split in ("val", "test")},
        "by_source": {split: split_reports[split]["by_source"] for split in ("val", "test")},
        "validation_grid": [
            {"route": item["route"], "R2": item["R2"], "mPQ+": item["mPQ+"]}
            for item in split_reports["val"]["combinations"]
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "route": report["selected_route"],
        "validation": {key: report["selected_validation"][key] for key in ("R2", "mPQ+")},
        "test": {key: report["selected_test"][key] for key in ("R2", "mPQ+")},
        "endpoints": report["endpoints"],
    }, indent=2))


if __name__ == "__main__":
    main()
