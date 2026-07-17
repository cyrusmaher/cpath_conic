#!/usr/bin/env python
"""Paired patch bootstrap for source-level fixed-mask routing decisions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES


def patch_pq_stats(features_path: Path, probabilities_path: Path, cache_path: Path) -> dict[str, np.ndarray]:
    features = np.load(features_path)
    probabilities = np.load(probabilities_path)
    cache = np.load(cache_path)
    patch_ids = features["patch_ids"].astype(np.int32)
    instance_ids = features["instance_ids"].astype(np.int32)
    if not np.array_equal(patch_ids, probabilities["patch_ids"]) or not np.array_equal(instance_ids, probabilities["instance_ids"]):
        raise ValueError("Feature and probability IDs are not aligned")
    assignments = probabilities["class_probs"].argmax(axis=1).astype(np.int8) + 1
    labels = features["labels"].astype(np.int8)
    ious = features["ious"].astype(np.float64)
    n_patches = cache["gt_full_counts"].shape[0]
    tp = np.zeros((n_patches, len(CLASS_NAMES)), dtype=np.int64)
    pred = np.zeros_like(tp)
    sum_iou = np.zeros((n_patches, len(CLASS_NAMES)), dtype=np.float64)
    np.add.at(pred, (patch_ids, assignments - 1), 1)
    matched = (assignments == labels) & (labels > 0) & (ious > 0.5)
    np.add.at(tp, (patch_ids[matched], assignments[matched] - 1), 1)
    np.add.at(sum_iou, (patch_ids[matched], assignments[matched] - 1), ious[matched])
    gt = cache["gt_full_counts"].astype(np.int64)
    return {"tp": tp, "fp": pred - tp, "fn": gt - tp, "sum_iou": sum_iou}


def pooled_mpq(stats: dict[str, np.ndarray], rows: np.ndarray) -> float:
    tp = stats["tp"][rows].sum(axis=0)
    fp = stats["fp"][rows].sum(axis=0)
    fn = stats["fn"][rows].sum(axis=0)
    sum_iou = stats["sum_iou"][rows].sum(axis=0)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return float(np.divide(sum_iou, denominator, out=np.zeros_like(sum_iou), where=denominator > 0).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    for prefix in ("a", "b"):
        parser.add_argument(f"--{prefix}-features", type=Path, required=True)
        parser.add_argument(f"--{prefix}-probabilities", type=Path, required=True)
        parser.add_argument(f"--{prefix}-cache", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    methods = {
        prefix: patch_pq_stats(
            getattr(args, f"{prefix}_features"),
            getattr(args, f"{prefix}_probabilities"),
            getattr(args, f"{prefix}_cache"),
        )
        for prefix in ("a", "b")
    }
    rng = np.random.default_rng(args.seed)
    results = {}
    for source in sorted(metadata.source.dropna().unique()):
        patch_ids = metadata.loc[(metadata.split == "val") & (metadata.source == source), "patch_id"].to_numpy(dtype=np.int32)
        observed_a = pooled_mpq(methods["a"], patch_ids)
        observed_b = pooled_mpq(methods["b"], patch_ids)
        differences = np.empty(args.replicates, dtype=np.float64)
        for index in range(args.replicates):
            sample = rng.choice(patch_ids, size=len(patch_ids), replace=True)
            differences[index] = pooled_mpq(methods["b"], sample) - pooled_mpq(methods["a"], sample)
        results[str(source)] = {
            "patches": int(len(patch_ids)),
            "observed_a_mPQ+": observed_a,
            "observed_b_mPQ+": observed_b,
            "observed_delta_b_minus_a": observed_b - observed_a,
            "paired_bootstrap_95_ci": [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))],
            "bootstrap_probability_b_better": float((differences > 0).mean()),
        }
    report = {"split": "val", "replicates": args.replicates, "seed": args.seed, "sources": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
