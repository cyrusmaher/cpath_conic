#!/usr/bin/env python
"""Source-group bootstrap uncertainty for post-hoc CoNIC experiments."""
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
from scripts.run_posthoc_experiments import apply_vector_scaling, assignments_at_margin


def predicted_counts(assignments, patch_ids, central, n_patches):
    counts = np.zeros((n_patches, 6), dtype=np.int32)
    keep = central & (assignments > 0)
    np.add.at(counts, (patch_ids[keep], assignments[keep] - 1), 1)
    return counts


def r2_from_arrays(target, prediction):
    residual = np.square(target - prediction).sum(axis=0)
    total = np.square(target - target.mean(axis=0)).sum(axis=0)
    valid = total > 0
    return float(np.mean(1.0 - residual[valid] / total[valid]))


def group_pq_stats(assignments, labels, ious, patch_ids, group_by_patch, gt_full_counts, selected_patch_ids):
    selected = np.isin(patch_ids, selected_patch_ids)
    group_names = np.unique(group_by_patch[selected_patch_ids])
    group_lookup = {name: index for index, name in enumerate(group_names)}
    stats = np.zeros((len(group_names), 6, 4), dtype=np.float64)
    for group_name, group_index in group_lookup.items():
        group_patches = selected_patch_ids[group_by_patch[selected_patch_ids] == group_name]
        rows = selected & np.isin(patch_ids, group_patches)
        assigned = assignments[rows]
        target = labels[rows]
        overlap = ious[rows]
        correct = (assigned > 0) & (assigned == target) & (overlap > 0.5)
        gt_total = gt_full_counts[group_patches].sum(axis=0)
        for class_id in range(1, 7):
            matched = correct & (assigned == class_id)
            tp = int(matched.sum())
            pred_total = int((assigned == class_id).sum())
            stats[group_index, class_id - 1] = [tp, pred_total - tp, int(gt_total[class_id - 1]) - tp, float(overlap[matched].sum())]
    return group_names, stats


def pq_from_stats(stats):
    tp, fp, fn, sum_iou = stats.T
    denominator = tp + 0.5 * fp + 0.5 * fn
    return float(np.mean(np.divide(sum_iou, denominator, out=np.zeros_like(sum_iou), where=denominator > 0)))


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--posthoc-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    feature_data = np.load(args.features)
    probability_data = np.load(args.probabilities)
    patch_ids = feature_data["patch_ids"].astype(np.int32)
    labels = feature_data["labels"].astype(np.int8)
    ious = feature_data["ious"].astype(np.float32)
    raw_probabilities = probability_data["class_probs"].astype(np.float32)
    cache = np.load(args.posthoc_dir / "fixed_mask_cache.npz")
    central = cache["central"].astype(bool)
    gt_full_counts = cache["gt_full_counts"].astype(np.int32)
    report = json.loads((args.posthoc_dir / "posthoc_report.json").read_text())
    calibration = report["calibration"]
    biases = np.asarray([calibration["biases"][name] for name in CLASS_NAMES])
    calibrated = apply_vector_scaling(raw_probabilities, np.log(calibration["temperature"]), biases)
    shifted = np.load(args.posthoc_dir / "recalibrated_probabilities.npz")["class_probs"].astype(np.float32)
    variants = {
        "raw": raw_probabilities.argmax(axis=1).astype(np.int8) + 1,
        "calibrated": calibrated.argmax(axis=1).astype(np.int8) + 1,
        "label_shifted": shifted.argmax(axis=1).astype(np.int8) + 1,
        "margin_0.07": assignments_at_margin(shifted, 0.07),
    }

    n_patches = len(metadata)
    group_by_patch = metadata.source_group.to_numpy()
    test_patch_ids = metadata.loc[metadata.split == "test", "patch_id"].to_numpy(dtype=np.int32)
    test_groups = np.unique(group_by_patch[test_patch_ids])
    true_counts = metadata[COUNT_COLUMNS].to_numpy(dtype=np.float64)
    variant_counts = {name: predicted_counts(values, patch_ids, central, n_patches) for name, values in variants.items()}
    variant_group_pq = {}
    for name, values in variants.items():
        group_names, stats = group_pq_stats(values, labels, ious, patch_ids, group_by_patch, gt_full_counts, test_patch_ids)
        if not np.array_equal(group_names, test_groups):
            raise RuntimeError("Group alignment failure")
        variant_group_pq[name] = stats

    rng = np.random.default_rng(args.seed)
    bootstrap = {name: {"R2": [], "mPQ+": []} for name in variants}
    deltas = {name: {"R2": [], "mPQ+": []} for name in variants if name != "raw"}
    group_patch_indices = [test_patch_ids[group_by_patch[test_patch_ids] == group] for group in test_groups]
    for _ in range(args.replicates):
        sampled_group_indices = rng.integers(0, len(test_groups), len(test_groups))
        sampled_patches = np.concatenate([group_patch_indices[index] for index in sampled_group_indices])
        current = {}
        for name in variants:
            current[name] = {
                "R2": r2_from_arrays(true_counts[sampled_patches], variant_counts[name][sampled_patches]),
                "mPQ+": pq_from_stats(variant_group_pq[name][sampled_group_indices].sum(axis=0)),
            }
            for metric in ["R2", "mPQ+"]:
                bootstrap[name][metric].append(current[name][metric])
        for name in deltas:
            for metric in ["R2", "mPQ+"]:
                deltas[name][metric].append(current[name][metric] - current["raw"][metric])

    # Bootstrap the validation-selected global margin. Sufficient statistics
    # avoid repeatedly materializing all patch predictions.
    val_patch_ids = metadata.loc[metadata.split == "val", "patch_id"].to_numpy(dtype=np.int32)
    val_groups = np.unique(group_by_patch[val_patch_ids])
    thresholds = np.arange(0.0, 0.200001, 0.01)
    group_n = np.zeros(len(val_groups), dtype=np.int32)
    group_sum = np.zeros((len(val_groups), 6), dtype=np.float64)
    group_sum2 = np.zeros((len(val_groups), 6), dtype=np.float64)
    group_sse = np.zeros((len(thresholds), len(val_groups), 6), dtype=np.float64)
    for group_index, group in enumerate(val_groups):
        group_patches = val_patch_ids[group_by_patch[val_patch_ids] == group]
        targets = true_counts[group_patches]
        group_n[group_index] = len(group_patches)
        group_sum[group_index] = targets.sum(axis=0)
        group_sum2[group_index] = np.square(targets).sum(axis=0)
        for threshold_index, threshold in enumerate(thresholds):
            assignments = assignments_at_margin(calibrated, float(threshold))
            counts = predicted_counts(assignments, patch_ids, central, n_patches)[group_patches]
            group_sse[threshold_index, group_index] = np.square(targets - counts).sum(axis=0)
    selected_thresholds = []
    for _ in range(args.replicates):
        sampled = rng.integers(0, len(val_groups), len(val_groups))
        weights = np.bincount(sampled, minlength=len(val_groups))
        n = int(weights @ group_n)
        target_sum = weights @ group_sum
        target_sum2 = weights @ group_sum2
        sst = target_sum2 - np.square(target_sum) / n
        sse = np.einsum("g,tgc->tc", weights, group_sse)
        scores = np.nanmean(1.0 - np.divide(sse, sst[None, :], out=np.full_like(sse, np.nan), where=sst[None, :] > 0), axis=1)
        selected_thresholds.append(float(thresholds[int(np.nanargmax(scores))]))

    result = {
        "method": "source_group bootstrap with replacement",
        "replicates": args.replicates,
        "test_groups": int(len(test_groups)),
        "metrics": {name: {metric: interval(values) for metric, values in data.items()} for name, data in bootstrap.items()},
        "delta_vs_raw": {
            name: {
                metric: {
                    **interval(values),
                    "probability_positive": float(np.mean(np.asarray(values) > 0)),
                }
                for metric, values in data.items()
            }
            for name, data in deltas.items()
        },
        "margin_threshold_stability": {
            "validation_groups": int(len(val_groups)),
            "median": float(np.median(selected_thresholds)),
            "ci95": [float(np.quantile(selected_thresholds, 0.025)), float(np.quantile(selected_thresholds, 0.975))],
            "selection_frequency": {f"{threshold:.2f}": int(np.sum(np.isclose(selected_thresholds, threshold))) for threshold in thresholds if np.any(np.isclose(selected_thresholds, threshold))},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
