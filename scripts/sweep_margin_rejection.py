#!/usr/bin/env python
"""Select a top-1/top-2 rejection margin on validation only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CELLVIT_ROOT = ROOT / "third_party" / "CellViT-plus-plus"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CELLVIT_ROOT))

from cpath_conic.experiment_metrics import evaluate_fixed_masks, load_or_build_instance_cache
from scripts.train_metric_aligned_classifier import infer_assignments


def json_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "predicted_counts"}


def main() -> None:
    import torch
    from cellvit.models.classifier.linear_classifier import LinearClassifier

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--instance-maps", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--margin-step", type=float, default=0.005)
    parser.add_argument("--max-margin", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    data = np.load(args.features)
    features = data["features"].astype(np.float32)
    patch_ids = data["patch_ids"].astype(np.int32)
    instance_ids = data["instance_ids"].astype(np.int32)
    labels = data["labels"].astype(np.int8)
    ious = data["ious"].astype(np.float32)
    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    val_patch_ids = metadata.loc[metadata.split == "val", "patch_id"].to_numpy(dtype=np.int32)
    val_mask = np.isin(patch_ids, val_patch_ids)
    val_indices = np.flatnonzero(val_mask)

    checkpoint = torch.load(args.classifier, map_location="cpu", weights_only=False)
    if int(checkpoint["num_classes"]) != 6:
        raise ValueError("Margin rejection expects a six-class classifier")
    model = LinearClassifier(
        embed_dim=int(checkpoint["embed_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_classes=6,
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    raw_assignments, probabilities = infer_assignments(model, features[val_indices], args.device, 6, args.batch_size)
    pair = np.partition(probabilities, -2, axis=1)[:, -2:]
    margins = pair[:, 1] - pair[:, 0]

    cache = load_or_build_instance_cache(args.cache, args.prepared, patch_ids, instance_ids, args.instance_maps)
    central = cache["central"].astype(bool)
    gt_full_counts = cache["gt_full_counts"].astype(np.int32)

    def metrics_at(threshold: float) -> dict:
        assignments = raw_assignments.copy()
        assignments[margins < threshold] = 0
        all_assignments = np.zeros(len(features), dtype=np.int8)
        all_assignments[val_indices] = assignments
        return evaluate_fixed_masks(
            all_assignments,
            labels,
            ious,
            patch_ids,
            central,
            metadata,
            gt_full_counts,
            "val",
        )

    thresholds = np.arange(0.0, args.max_margin + args.margin_step / 2, args.margin_step)
    rows = []
    metrics_by_threshold = {}
    for threshold in thresholds:
        metrics = metrics_at(float(threshold))
        metrics_by_threshold[float(threshold)] = metrics
        rows.append(
            {
                "threshold": float(threshold),
                "val_R2": metrics["R2"],
                "val_mPQ+": metrics["mPQ+"],
                "val_rejected_fraction": metrics["rejected_fraction"],
            }
        )
    selected = {}
    for metric, column in (("R2", "val_R2"), ("mPQ+", "val_mPQ+")):
        best = max(rows, key=lambda row: row[column])
        selected[metric] = {
            "threshold": best["threshold"],
            "validation": json_metrics(metrics_by_threshold[best["threshold"]]),
        }

    matched = (labels[val_indices] > 0) & (ious[val_indices] > 0.5)
    central_val = central[val_indices]
    diagnostics = {}
    for name, mask in {
        "matched": matched,
        "unmatched": ~matched,
        "central": central_val,
        "central_unmatched": central_val & ~matched,
    }.items():
        values = margins[mask]
        diagnostics[name] = {
            "n": int(mask.sum()),
            "median_margin": float(np.median(values)) if len(values) else None,
            "p10_margin": float(np.quantile(values, 0.1)) if len(values) else None,
            "p90_margin": float(np.quantile(values, 0.9)) if len(values) else None,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "selection_split": "validation",
        "classifier": str(args.classifier),
        "rule": "Reject when raw top-1 minus top-2 probability is below the selected global threshold.",
        "raw_validation": json_metrics(metrics_by_threshold[0.0]),
        "selected": selected,
        "margin_diagnostics": diagnostics,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "raw": {key: report["raw_validation"][key] for key in ("R2", "mPQ+", "rejected_fraction")},
        "selected": {
            metric: {
                "threshold": value["threshold"],
                "R2": value["validation"]["R2"],
                "mPQ+": value["validation"]["mPQ+"],
                "rejected_fraction": value["validation"]["rejected_fraction"],
            }
            for metric, value in selected.items()
        },
        "margin_diagnostics": diagnostics,
    }, indent=2))


if __name__ == "__main__":
    main()
