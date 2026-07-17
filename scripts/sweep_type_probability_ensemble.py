#!/usr/bin/env python
"""Validation-select a log-probability ensemble on fixed CellViT instances."""
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


def calibrated_logits(probabilities: np.ndarray, report: dict | None) -> np.ndarray:
    logits = np.log(np.clip(probabilities.astype(np.float64), 1e-9, 1.0))
    if report:
        calibration = report["calibration"]
        logits = logits / float(calibration["temperature"])
        logits += np.asarray([calibration["biases"][name] for name in CLASS_NAMES], dtype=np.float64)
    return logits


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--probabilities-a", type=Path, required=True)
    parser.add_argument("--probabilities-b", type=Path, required=True)
    parser.add_argument("--calibration-a", type=Path, default=None)
    parser.add_argument("--calibration-b", type=Path, default=None)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    features = np.load(args.features)
    a = np.load(args.probabilities_a)
    b = np.load(args.probabilities_b)
    for probability_data, path in [(a, args.probabilities_a), (b, args.probabilities_b)]:
        if not np.array_equal(features["patch_ids"], probability_data["patch_ids"]):
            raise ValueError(f"Patch IDs are not aligned for {path}")
        if not np.array_equal(features["instance_ids"], probability_data["instance_ids"]):
            raise ValueError(f"Instance IDs are not aligned for {path}")
    report_a = json.loads(args.calibration_a.read_text()) if args.calibration_a else None
    report_b = json.loads(args.calibration_b.read_text()) if args.calibration_b else None
    logits_a = calibrated_logits(a["class_probs"], report_a)
    logits_b = calibrated_logits(b["class_probs"], report_b)
    cache = np.load(args.cache)
    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    labels = features["labels"].astype(np.int8)
    ious = features["ious"].astype(np.float32)
    patch_ids = features["patch_ids"].astype(np.int32)
    central = cache["central"].astype(bool)
    gt_full_counts = cache["gt_full_counts"].astype(np.int32)

    rows = []
    best = None
    for alpha in [float(value) for value in args.alphas.split(",") if value.strip()]:
        probabilities = softmax(alpha * logits_a + (1.0 - alpha) * logits_b)
        assignments = probabilities.argmax(axis=1).astype(np.int8) + 1
        metrics = evaluate_fixed_masks(assignments, labels, ious, patch_ids, central, metadata, gt_full_counts, "val")
        score = 0.5 * (metrics["R2"] + metrics["mPQ+"])
        row = {"alpha_a": alpha, "val_R2": metrics["R2"], "val_mPQ+": metrics["mPQ+"], "selection_score": score}
        rows.append(row)
        if best is None or score > best["score"]:
            best = {"alpha": alpha, "score": score, "probabilities": probabilities, "val": metrics}

    assignments = best["probabilities"].argmax(axis=1).astype(np.int8) + 1
    test = evaluate_fixed_masks(assignments, labels, ious, patch_ids, central, metadata, gt_full_counts, "test")
    args.outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.outdir / "cell_probabilities.npz",
        patch_ids=features["patch_ids"],
        instance_ids=features["instance_ids"],
        class_probs=best["probabilities"].astype(np.float32),
    )
    with (args.outdir / "alpha_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "alpha_definition": "alpha * logits_a + (1-alpha) * logits_b",
        "selected_alpha_a": best["alpha"],
        "selection_metric": "mean(validation R2, validation mPQ+)",
        "validation": {key: value for key, value in best["val"].items() if key != "predicted_counts"},
        "test": {key: value for key, value in test.items() if key != "predicted_counts"},
    }
    (args.outdir / "ensemble_report.json").write_text(json.dumps(report, indent=2))
    (args.outdir / "metrics_test.json").write_text(json.dumps(report["test"], indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
