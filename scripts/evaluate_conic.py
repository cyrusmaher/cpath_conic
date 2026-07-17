#!/usr/bin/env python
"""Evaluate prediction arrays with the official CoNIC pooled metrics."""
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
from cpath_conic.data import central_crop_counts, load_metadata, official_hovernet_fold
from cpath_conic.metrics import binary_instance_segmentation_metrics, multiclass_pq_plus, multiclass_r2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="N x 256 x 256 x 2 uint/int .npy: instance map, class map")
    parser.add_argument("--counts", type=Path, default=None, help="Optional N x 6 counts .npy or CSV; otherwise counts are derived from predictions")
    parser.add_argument("--split", default="test")
    parser.add_argument("--benchmarks", type=Path, default=None, help="Optional published benchmark JSON to include in the report")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metadata = load_metadata(args.prepared)
    if args.split == "official_hovernet_val":
        _, validation_ids = official_hovernet_fold(metadata)
        subset = metadata.loc[metadata.patch_id.isin(validation_ids)].sort_values("patch_id")
    else:
        subset = metadata.loc[metadata.split == args.split].sort_values("patch_id")
    pred_all = np.load(args.predictions)
    if pred_all.ndim != 4 or pred_all.shape[-1] != 2:
        raise ValueError("predictions must have shape N,H,W,2")
    pred = pred_all[subset.patch_id.to_numpy()]
    true = np.zeros_like(pred, dtype=np.int32)
    true_counts = subset[["patch_id", *COUNT_COLUMNS]].copy()
    for i, patch_id in enumerate(subset.patch_id):
        label = np.load(args.prepared / "labels" / f"{int(patch_id):05d}.npy", allow_pickle=True).item()
        true[i, ..., 0] = label["inst_map"]
        true[i, ..., 1] = label["class_map"]
    pq = multiclass_pq_plus(true, pred)
    segmentation = binary_instance_segmentation_metrics(true, pred)
    if args.counts:
        if args.counts.suffix == ".csv":
            pred_counts = pd.read_csv(args.counts)
        else:
            values = np.load(args.counts)
            pred_counts = pd.DataFrame(values, columns=CLASS_NAMES)
        if len(pred_counts) == len(metadata):
            pred_counts = pred_counts.iloc[subset.index].reset_index(drop=True)
        elif len(pred_counts) == len(subset):
            pred_counts = pred_counts.reset_index(drop=True)
        else:
            raise ValueError("counts must contain either all prepared patches or exactly the selected split")
    else:
        values = []
        for patch in pred:
            values.append(central_crop_counts(patch[..., 0], patch[..., 1]))
        pred_counts = pd.DataFrame(values, columns=CLASS_NAMES)
    true_count_frame = true_counts[COUNT_COLUMNS].copy()
    true_count_frame.columns = CLASS_NAMES
    r2 = multiclass_r2(true_count_frame, pred_counts[CLASS_NAMES])
    result = {
        "split": args.split,
        "n_patches": int(len(subset)),
        "mPQ+": pq["mPQ+"],
        "mDQ+": pq["mDQ+"],
        "mSQ+": pq["mSQ+"],
        "per_class_pq": pq["per_class"],
        "R2": r2["R2"],
        "per_class_R2": r2["per_class"],
        "segmentation_diagnostics": segmentation,
    }
    if args.benchmarks:
        benchmark_data = json.loads(args.benchmarks.read_text())
        result["published_benchmarks"] = benchmark_data
        result["benchmark_comparison"] = []
        for reference in benchmark_data.get("references", []):
            comparison = {"name": reference.get("name"), "split": reference.get("split")}
            if reference.get("mpq_plus") is not None:
                comparison["mPQ+_delta"] = float(result["mPQ+"] - reference["mpq_plus"])
            if reference.get("r2") is not None:
                comparison["R2_delta"] = float(result["R2"] - reference["r2"])
            if len(comparison) > 2:
                result["benchmark_comparison"].append(comparison)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
