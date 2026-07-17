#!/usr/bin/env python
"""Validation-select a per-class convex blend of two CoNIC count outputs."""
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

from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS
from cpath_conic.metrics import multiclass_r2


def blended_counts(counts_a: np.ndarray, counts_b: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Blend integer counts, with ``alphas`` weighting A, then round once."""
    return np.rint(counts_a * alphas[None, :] + counts_b * (1.0 - alphas[None, :])).astype(np.int32)


def class_r2(true: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.square(true - true.mean()).sum())
    if denominator <= 0:
        return float("nan")
    return float(1.0 - np.square(predicted - true).sum() / denominator)


def macro_r2(true: np.ndarray, predicted: np.ndarray) -> dict:
    true_frame = pd.DataFrame(true, columns=CLASS_NAMES)
    predicted_frame = pd.DataFrame(predicted, columns=CLASS_NAMES)
    return multiclass_r2(true_frame, predicted_frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--counts-a", type=Path, required=True)
    parser.add_argument("--counts-b", type=Path, required=True)
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--out-counts", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="After the validation recipe is locked, also report test metrics. Off by default to preserve the test gate.",
    )
    args = parser.parse_args()

    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    true = metadata[COUNT_COLUMNS].to_numpy(dtype=np.int32)
    counts_a = np.load(args.counts_a).astype(np.int32)
    counts_b = np.load(args.counts_b).astype(np.int32)
    if counts_a.shape != true.shape or counts_b.shape != true.shape:
        raise ValueError(f"Expected count arrays shaped {true.shape}; got {counts_a.shape} and {counts_b.shape}")
    if not 0 < args.alpha_step <= 1:
        raise ValueError("--alpha-step must be in (0, 1]")

    val_mask = metadata.split.to_numpy() == "val"
    grid = np.unique(np.append(np.arange(0.0, 1.0 + args.alpha_step / 2, args.alpha_step), 1.0))
    grid = grid[(grid >= 0.0) & (grid <= 1.0)]
    rows: list[dict] = []
    selected = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    selected_scores = np.zeros(len(CLASS_NAMES), dtype=np.float64)

    for class_index, class_name in enumerate(CLASS_NAMES):
        candidates = []
        for alpha in grid:
            predicted = np.rint(
                alpha * counts_a[val_mask, class_index] + (1.0 - alpha) * counts_b[val_mask, class_index]
            ).astype(np.int32)
            score = class_r2(true[val_mask, class_index], predicted)
            row = {"class": class_name, "alpha_a": float(alpha), "val_R2": score}
            rows.append(row)
            candidates.append(row)
        best = max(candidates, key=lambda row: (row["val_R2"], -abs(row["alpha_a"] - 0.5)))
        selected[class_index] = best["alpha_a"]
        selected_scores[class_index] = best["val_R2"]

    output = blended_counts(counts_a, counts_b, selected)
    val_metrics = macro_r2(true[val_mask], output[val_mask])
    endpoint_a = {"val": macro_r2(true[val_mask], counts_a[val_mask])}
    endpoint_b = {"val": macro_r2(true[val_mask], counts_b[val_mask])}
    test_metrics = None
    if args.evaluate_test:
        test_mask = metadata.split.to_numpy() == "test"
        test_metrics = macro_r2(true[test_mask], output[test_mask])
        endpoint_a["test"] = macro_r2(true[test_mask], counts_a[test_mask])
        endpoint_b["test"] = macro_r2(true[test_mask], counts_b[test_mask])

    args.out_counts.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_counts, output)
    curve_path = args.out_report.with_suffix(".csv")
    with curve_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "alpha_a", "val_R2"])
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "selection_split": "val",
        "alpha_a_definition": f"weight on {args.name_a}; 1-alpha weights {args.name_b}",
        "alpha_step": args.alpha_step,
        "selected_alpha_a": {name: float(value) for name, value in zip(CLASS_NAMES, selected)},
        "selected_per_class_val_R2": {name: float(value) for name, value in zip(CLASS_NAMES, selected_scores)},
        "validation": val_metrics,
        "test": test_metrics,
        "test_evaluated": args.evaluate_test,
        "endpoints": {args.name_a: endpoint_a, args.name_b: endpoint_b},
        "counts": str(args.out_counts),
        "curve": str(curve_path),
    }
    args.out_report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
