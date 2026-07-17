#!/usr/bin/env python
"""Validation-selected probability calibration, label shift, and rejection."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES
from cpath_conic.experiment_metrics import evaluate_fixed_masks, load_or_build_instance_cache


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def apply_vector_scaling(probabilities: np.ndarray, log_temperature: float, biases: np.ndarray) -> np.ndarray:
    temperature = float(np.exp(log_temperature))
    logits = np.log(np.clip(probabilities, 1e-9, 1.0)) / temperature
    logits += biases - biases.mean()
    return softmax(logits).astype(np.float32)


def fit_vector_scaling(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray, dict]:
    labels = labels.astype(np.int64)

    def objective(parameters: np.ndarray) -> float:
        log_temperature = float(parameters[0])
        biases = parameters[1:] - parameters[1:].mean()
        temperature = np.exp(log_temperature)
        logits = np.log(np.clip(probabilities, 1e-9, 1.0)) / temperature + biases
        nll = np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels])
        return float(nll + 1e-4 * np.square(biases).mean())

    result = minimize(
        objective,
        np.zeros(1 + probabilities.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(np.log(0.05), np.log(10.0))] + [(-5.0, 5.0)] * probabilities.shape[1],
        options={"maxiter": 200},
    )
    biases = result.x[1:] - result.x[1:].mean()
    return float(result.x[0]), biases, {
        "success": bool(result.success),
        "message": str(result.message),
        "nll": float(result.fun),
        "temperature": float(np.exp(result.x[0])),
        "biases": dict(zip(CLASS_NAMES, biases.tolist())),
    }


def label_shift_em(
    target_probabilities: np.ndarray,
    source_prior: np.ndarray,
    max_iterations: int = 500,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Saerens-style EM using only unlabeled target probabilities."""
    source_prior = np.clip(source_prior.astype(np.float64), 1e-6, None)
    source_prior /= source_prior.sum()
    target_prior = source_prior.copy()
    adjusted = target_probabilities.astype(np.float64)
    for iteration in range(1, max_iterations + 1):
        ratios = target_prior / source_prior
        adjusted = target_probabilities * ratios[None, :]
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        updated = adjusted.mean(axis=0)
        if np.max(np.abs(updated - target_prior)) < tolerance:
            target_prior = updated
            break
        target_prior = updated
    ratios = target_prior / source_prior
    return target_prior, ratios, iteration


def adjust_prior(probabilities: np.ndarray, ratios: np.ndarray) -> np.ndarray:
    adjusted = probabilities.astype(np.float64) * ratios[None, :]
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted.astype(np.float32)


def assignments_at_margin(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    order = np.argpartition(probabilities, -2, axis=1)[:, -2:]
    pair = np.take_along_axis(probabilities, order, axis=1)
    margin = pair.max(axis=1) - pair.min(axis=1)
    assignments = probabilities.argmax(axis=1).astype(np.int8) + 1
    assignments[margin < threshold] = 0
    return assignments


def json_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "predicted_counts"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--instance-maps", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--margin-step", type=float, default=0.01)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    feature_data = np.load(args.features)
    probability_data = np.load(args.probabilities)
    patch_ids = feature_data["patch_ids"].astype(np.int32)
    instance_ids = feature_data["instance_ids"].astype(np.int32)
    if not np.array_equal(patch_ids, probability_data["patch_ids"]) or not np.array_equal(instance_ids, probability_data["instance_ids"]):
        raise ValueError("Feature and probability records are not aligned")
    labels = feature_data["labels"].astype(np.int8)
    ious = feature_data["ious"].astype(np.float32)
    raw_probabilities = probability_data["class_probs"].astype(np.float32)
    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    split_by_patch = metadata.set_index("patch_id").split.to_dict()
    record_split = np.asarray([split_by_patch[int(patch)] for patch in patch_ids])

    cache = load_or_build_instance_cache(
        args.outdir / "fixed_mask_cache.npz",
        args.prepared,
        patch_ids,
        instance_ids,
        args.instance_maps,
    )
    central = cache["central"].astype(bool)
    gt_full_counts = cache["gt_full_counts"].astype(np.int32)

    matched_val = (record_split == "val") & (labels > 0) & (ious > 0.5)
    log_temperature, biases, calibration = fit_vector_scaling(raw_probabilities[matched_val], labels[matched_val] - 1)
    calibrated = apply_vector_scaling(raw_probabilities, log_temperature, biases)

    train_central_matched = (record_split == "train") & central & (labels > 0) & (ious > 0.5)
    test_central = (record_split == "test") & central
    # Saerens EM assumes the source-domain class prior associated with the
    # calibrated source posterior.  Use empirical matched training labels,
    # rather than the model's own mean predictions (which can preserve its
    # class bias and understate the shift we are trying to estimate).
    source_prior = np.bincount(labels[train_central_matched] - 1, minlength=len(CLASS_NAMES)).astype(np.float64)
    source_prior /= source_prior.sum()
    target_prior, prior_ratios, em_iterations = label_shift_em(calibrated[test_central], source_prior)
    shifted = calibrated.copy()
    shifted[record_split == "test"] = adjust_prior(calibrated[record_split == "test"], prior_ratios)

    raw_assignments = raw_probabilities.argmax(axis=1).astype(np.int8) + 1
    calibrated_assignments = calibrated.argmax(axis=1).astype(np.int8) + 1
    shifted_assignments = shifted.argmax(axis=1).astype(np.int8) + 1
    baselines = {
        "raw_val": json_metrics(evaluate_fixed_masks(raw_assignments, labels, ious, patch_ids, central, metadata, gt_full_counts, "val")),
        "calibrated_val": json_metrics(evaluate_fixed_masks(calibrated_assignments, labels, ious, patch_ids, central, metadata, gt_full_counts, "val")),
        "raw_test": json_metrics(evaluate_fixed_masks(raw_assignments, labels, ious, patch_ids, central, metadata, gt_full_counts, "test")),
        "calibrated_test": json_metrics(evaluate_fixed_masks(calibrated_assignments, labels, ious, patch_ids, central, metadata, gt_full_counts, "test")),
        "label_shifted_test": json_metrics(evaluate_fixed_masks(shifted_assignments, labels, ious, patch_ids, central, metadata, gt_full_counts, "test")),
    }

    thresholds = np.arange(0.0, 0.9500001, args.margin_step)
    sweep_rows = []
    val_metrics_by_threshold = {}
    for threshold in thresholds:
        assignments = assignments_at_margin(calibrated, float(threshold))
        metrics = evaluate_fixed_masks(assignments, labels, ious, patch_ids, central, metadata, gt_full_counts, "val")
        val_metrics_by_threshold[float(threshold)] = metrics
        sweep_rows.append({
            "threshold": float(threshold),
            "val_R2": metrics["R2"],
            "val_mPQ+": metrics["mPQ+"],
            "val_mean_R2_mPQ+": 0.5 * (metrics["R2"] + metrics["mPQ+"]),
            "val_rejected_fraction": metrics["rejected_fraction"],
        })

    selection = {}
    objectives = {"R2": "val_R2", "mPQ+": "val_mPQ+", "mean_R2_mPQ+": "val_mean_R2_mPQ+"}
    assignment_artifacts = {}
    for name, column in objectives.items():
        best = max(sweep_rows, key=lambda row: row[column])
        threshold = float(best["threshold"])
        val_assignments = assignments_at_margin(calibrated, threshold)
        test_assignments = assignments_at_margin(shifted, threshold)
        selection[name] = {
            "threshold": threshold,
            "validation": json_metrics(val_metrics_by_threshold[threshold]),
            "test_with_unlabeled_label_shift": json_metrics(evaluate_fixed_masks(test_assignments, labels, ious, patch_ids, central, metadata, gt_full_counts, "test")),
        }
        assignment_artifacts[f"assignments_{name.replace('+', 'plus')}"] = test_assignments

    with (args.outdir / "margin_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0]))
        writer.writeheader()
        writer.writerows(sweep_rows)
    np.savez_compressed(
        args.outdir / "recalibrated_probabilities.npz",
        patch_ids=patch_ids,
        instance_ids=instance_ids,
        class_probs=shifted,
        **assignment_artifacts,
    )
    report = {
        "methodology": {
            "calibration": "Vector scaling fit only on matched validation instances.",
            "label_shift": "Saerens EM estimated from unlabeled test probabilities; no test labels or test cell counts are used for fitting.",
            "dustbin": "Reject when calibrated top-1 minus top-2 probability is below a global threshold selected on validation.",
            "test_note": "Test labels are used only after selection to report metrics.",
        },
        "calibration": calibration,
        "label_shift": {
            "source": "empirical classes of matched train-central instances",
            "target": "unlabeled test central predicted instances",
            "iterations": em_iterations,
            "source_prior": dict(zip(CLASS_NAMES, source_prior.tolist())),
            "estimated_target_prior": dict(zip(CLASS_NAMES, target_prior.tolist())),
            "target_over_source_ratio": dict(zip(CLASS_NAMES, prior_ratios.tolist())),
        },
        "baselines": baselines,
        "margin_selection": selection,
    }
    (args.outdir / "posthoc_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "calibration": calibration,
        "label_shift": report["label_shift"],
        "baselines": {name: {"R2": value["R2"], "mPQ+": value["mPQ+"]} for name, value in baselines.items()},
        "selected": {name: {"threshold": value["threshold"], "val_R2": value["validation"]["R2"], "val_mPQ+": value["validation"]["mPQ+"], "test_R2": value["test_with_unlabeled_label_shift"]["R2"], "test_mPQ+": value["test_with_unlabeled_label_shift"]["mPQ+"]} for name, value in selection.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
