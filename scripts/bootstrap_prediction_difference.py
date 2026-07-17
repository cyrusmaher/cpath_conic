#!/usr/bin/env python3
"""Paired patch bootstrap for two arbitrary CoNIC prediction/count outputs."""
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
from cpath_conic.metrics import multiclass_pq_plus


def patch_pq_stats(truth: np.ndarray, predictions: np.ndarray) -> dict[str, np.ndarray]:
    n_patches = len(truth)
    values = {name: np.zeros((n_patches, len(CLASS_NAMES)), dtype=dtype) for name, dtype in (
        ("tp", np.int32), ("fp", np.int32), ("fn", np.int32), ("sum_iou", np.float64)
    )}
    for patch_index in range(n_patches):
        metrics = multiclass_pq_plus(truth[patch_index : patch_index + 1], predictions[patch_index : patch_index + 1])
        for class_index, class_name in enumerate(CLASS_NAMES):
            item = metrics["per_class"][class_name]
            for name in values:
                values[name][patch_index, class_index] = item[name]
    return values


def pooled_mpq(stats: dict[str, np.ndarray], rows: np.ndarray) -> float:
    totals = {name: values[rows].sum(axis=0) for name, values in stats.items()}
    denominator = totals["tp"] + 0.5 * totals["fp"] + 0.5 * totals["fn"]
    pq = np.divide(totals["sum_iou"], denominator, out=np.zeros_like(denominator, dtype=np.float64), where=denominator > 0)
    return float(pq.mean())


def macro_r2(truth: np.ndarray, predicted: np.ndarray, rows: np.ndarray) -> float:
    true = truth[rows].astype(np.float64)
    pred = predicted[rows].astype(np.float64)
    denominator = np.square(true - true.mean(axis=0, keepdims=True)).sum(axis=0)
    numerator = np.square(pred - true).sum(axis=0)
    scores = np.divide(numerator, denominator, out=np.full_like(denominator, np.nan), where=denominator > 0)
    return float(np.nanmean(1.0 - scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--predictions-a", type=Path, required=True)
    parser.add_argument("--counts-a", type=Path, required=True)
    parser.add_argument("--predictions-b", type=Path, required=True)
    parser.add_argument("--counts-b", type=Path, required=True)
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--split", default="val")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    selected = metadata.loc[metadata.split == args.split].sort_values("patch_id")
    patch_ids = selected.patch_id.to_numpy(dtype=np.int32)
    predictions = {
        "a": np.load(args.predictions_a, mmap_mode="r")[patch_ids],
        "b": np.load(args.predictions_b, mmap_mode="r")[patch_ids],
    }
    counts = {
        "a": np.load(args.counts_a, mmap_mode="r")[patch_ids],
        "b": np.load(args.counts_b, mmap_mode="r")[patch_ids],
    }
    truth = np.zeros_like(predictions["a"], dtype=np.int32)
    for index, patch_id in enumerate(patch_ids):
        label = np.load(args.prepared / "labels" / f"{int(patch_id):05d}.npy", allow_pickle=True).item()
        truth[index, ..., 0] = label["inst_map"]
        truth[index, ..., 1] = label["class_map"]
    true_counts = selected[COUNT_COLUMNS].to_numpy(dtype=np.int32)
    pq_stats = {method: patch_pq_stats(truth, prediction) for method, prediction in predictions.items()}
    rng = np.random.default_rng(args.seed)

    def compare(rows: np.ndarray) -> dict:
        observed = {
            method: {
                "R2": macro_r2(true_counts, counts[method], rows),
                "mPQ+": pooled_mpq(pq_stats[method], rows),
            }
            for method in ("a", "b")
        }
        r2_delta = np.empty(args.replicates, dtype=np.float64)
        mpq_delta = np.empty(args.replicates, dtype=np.float64)
        for replicate in range(args.replicates):
            sample = rng.choice(rows, size=len(rows), replace=True)
            r2_delta[replicate] = macro_r2(true_counts, counts["b"], sample) - macro_r2(true_counts, counts["a"], sample)
            mpq_delta[replicate] = pooled_mpq(pq_stats["b"], sample) - pooled_mpq(pq_stats["a"], sample)
        return {
            "n_patches": int(len(rows)),
            "observed": observed,
            "delta_b_minus_a": {"R2": observed["b"]["R2"] - observed["a"]["R2"], "mPQ+": observed["b"]["mPQ+"] - observed["a"]["mPQ+"]},
            "paired_bootstrap_95_ci": {
                "R2": [float(np.quantile(r2_delta, 0.025)), float(np.quantile(r2_delta, 0.975))],
                "mPQ+": [float(np.quantile(mpq_delta, 0.025)), float(np.quantile(mpq_delta, 0.975))],
            },
            "probability_b_better": {"R2": float((r2_delta > 0).mean()), "mPQ+": float((mpq_delta > 0).mean())},
        }

    all_rows = np.arange(len(selected), dtype=np.int32)
    report = {
        "split": args.split,
        "method_names": {"a": args.name_a, "b": args.name_b},
        "replicates": args.replicates,
        "seed": args.seed,
        "overall": compare(all_rows),
        "by_source": {
            str(source): compare(np.flatnonzero(selected.source.to_numpy() == source))
            for source in sorted(selected.source.dropna().unique())
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
