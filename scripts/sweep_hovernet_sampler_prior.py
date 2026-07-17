#!/usr/bin/env python3
"""Validation-only nucleus-exposure prior correction after replacement sampling.

This is deliberately narrower than generic train-to-test label-shift correction:
both central-224 nucleus priors are known from development-training labels, but
they are only a proxy for the pixel-wise type-loss prior learned by the network.
Geometry is held fixed and only decoded-instance type assignments are changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS
from cpath_conic.data import central_crop_counts
from cpath_conic.metrics import instance_type_confusion, multiclass_pq_plus, multiclass_r2
from scripts.train_hovernet_our_split import count_error_stats


def normalized_prior(values: np.ndarray, epsilon: float = 1.0e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(CLASS_NAMES),) or np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("a prior must contain six finite non-negative values")
    values = np.maximum(values, epsilon)
    return values / values.sum()


def apply_log_prior_correction(
    probabilities: np.ndarray,
    sampled_prior: np.ndarray,
    target_prior: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Apply ``p(class|x) * (target/sample)^strength`` and renormalize."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError("probabilities must have shape instances-by-six-classes")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and non-negative")
    sample = normalized_prior(sampled_prior)
    target = normalized_prior(target_prior)
    log_values = np.log(np.maximum(probabilities, 1.0e-12))
    log_values += float(strength) * (np.log(target) - np.log(sample))[None, :]
    log_values -= log_values.max(axis=1, keepdims=True)
    adjusted = np.exp(log_values)
    return (adjusted / adjusted.sum(axis=1, keepdims=True)).astype(np.float32)


def relabel_predictions(
    predictions: np.ndarray,
    patch_ids: np.ndarray,
    probability_patch_ids: np.ndarray,
    probability_instance_ids: np.ndarray,
    assignments: np.ndarray,
) -> np.ndarray:
    """Relabel every decoded instance exactly once while preserving geometry."""
    output = np.asarray(predictions).copy()
    patch_ids = np.asarray(patch_ids, dtype=np.int64)
    probability_patch_ids = np.asarray(probability_patch_ids, dtype=np.int64)
    probability_instance_ids = np.asarray(probability_instance_ids, dtype=np.int64)
    assignments = np.asarray(assignments, dtype=np.int64)
    if output.ndim != 4 or output.shape[-1] != 2 or len(output) != len(patch_ids):
        raise ValueError("predictions must be patch-aligned NxHxWx2 maps")
    if not (len(probability_patch_ids) == len(probability_instance_ids) == len(assignments)):
        raise ValueError("probability keys and assignments have different lengths")
    if len(np.unique(patch_ids)) != len(patch_ids):
        raise ValueError("prediction patch IDs are not unique")
    if np.any((assignments < 1) | (assignments > len(CLASS_NAMES))):
        raise ValueError("assignments must be CoNIC class IDs 1..6")

    patch_index = {int(patch_id): index for index, patch_id in enumerate(patch_ids)}
    assignments_by_patch: dict[int, dict[int, int]] = {}
    for patch_id, instance_id, class_id in zip(
        probability_patch_ids, probability_instance_ids, assignments, strict=True
    ):
        key = (int(patch_id), int(instance_id))
        patch_assignments = assignments_by_patch.setdefault(key[0], {})
        if key[1] in patch_assignments:
            raise ValueError(f"duplicate decoded-instance probability key: {key}")
        if key[0] not in patch_index:
            raise ValueError(f"probability patch {key[0]} is absent from predictions")
        patch_assignments[key[1]] = int(class_id)

    for patch_id, patch in zip(patch_ids, output, strict=True):
        instance_map = patch[..., 0]
        decoded_ids = np.unique(instance_map)
        decoded_ids = decoded_ids[decoded_ids > 0].astype(np.int64)
        patch_assignments = assignments_by_patch.get(int(patch_id), {})
        assigned_ids = np.asarray(sorted(patch_assignments), dtype=np.int64)
        if not np.array_equal(decoded_ids, assigned_ids):
            missing = np.setdiff1d(decoded_ids, assigned_ids)
            extra = np.setdiff1d(assigned_ids, decoded_ids)
            raise ValueError(
                f"decoded/probability instance mismatch for patch {int(patch_id)}: "
                f"{len(missing)} missing, {len(extra)} extra"
            )
        if not len(decoded_ids):
            patch[..., 1] = 0
            continue
        lookup = np.zeros(int(decoded_ids[-1]) + 1, dtype=np.int32)
        lookup[assigned_ids] = np.asarray([patch_assignments[int(value)] for value in assigned_ids])
        patch[..., 1] = lookup[instance_map]
    return output


def sampled_exposure_prior(curve_rows: list[dict], checkpoint_epoch: int) -> tuple[np.ndarray, int]:
    """Aggregate actual per-epoch sampled nucleus exposure through a checkpoint."""
    totals = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    used = 0
    for row in curve_rows:
        epoch = int(row.get("epoch", -1))
        if epoch < 1 or epoch > checkpoint_epoch:
            continue
        exposure = row.get("train_sampling_actual") or {}
        draws = int(exposure.get("draws", 0))
        means = exposure.get("mean_nuclei_per_draw") or {}
        if draws <= 0 or any(name not in means for name in CLASS_NAMES):
            raise ValueError(f"curve epoch {epoch} lacks actual sampled nucleus exposure")
        totals += draws * np.asarray([means[name] for name in CLASS_NAMES], dtype=np.float64)
        used += 1
    if used != checkpoint_epoch:
        raise ValueError(f"expected exposure for epochs 1..{checkpoint_epoch}, found {used}")
    return normalized_prior(totals), used


def metric_payload(
    truth: np.ndarray,
    prediction: np.ndarray,
    true_counts: np.ndarray,
    source_values: np.ndarray | None = None,
    *,
    include_confusion: bool = True,
) -> dict:
    pq = multiclass_pq_plus(truth, prediction)
    predicted_counts = np.asarray(
        [central_crop_counts(patch[..., 0], patch[..., 1]) for patch in prediction], dtype=np.int32
    )
    r2 = multiclass_r2(
        pd.DataFrame(true_counts, columns=CLASS_NAMES),
        pd.DataFrame(predicted_counts, columns=CLASS_NAMES),
    )
    payload = {
        "R2": float(r2["R2"]),
        "mPQ+": float(pq["mPQ+"]),
        "mDQ+": float(pq["mDQ+"]),
        "mSQ+": float(pq["mSQ+"]),
        "per_class_R2": r2["per_class"],
        "per_class_PQ": {name: values["pq"] for name, values in pq["per_class"].items()},
        "per_class_DQ": {name: values["dq"] for name, values in pq["per_class"].items()},
        "per_class_SQ": {name: values["sq"] for name, values in pq["per_class"].items()},
        "count_error": count_error_stats(true_counts, predicted_counts),
    }
    if include_confusion:
        payload["instance_type_confusion"] = instance_type_confusion(truth, prediction)
    if source_values is not None:
        source_values = np.asarray(source_values).astype(str)
        if source_values.shape != (len(truth),):
            raise ValueError("source values must align one-to-one with validation patches")
        per_source = {}
        for source in sorted(np.unique(source_values)):
            mask = source_values == source
            source_pq = multiclass_pq_plus(truth[mask], prediction[mask])
            source_r2 = multiclass_r2(
                pd.DataFrame(true_counts[mask], columns=CLASS_NAMES),
                pd.DataFrame(predicted_counts[mask], columns=CLASS_NAMES),
            )
            per_source[source] = {
                "patches": int(mask.sum()),
                "R2": float(source_r2["R2"]),
                "mPQ+": float(source_pq["mPQ+"]),
                "mDQ+": float(source_pq["mDQ+"]),
                "mSQ+": float(source_pq["mSQ+"]),
                "per_class_R2": source_r2["per_class"],
                "per_class_PQ": {
                    name: values["pq"] for name, values in source_pq["per_class"].items()
                },
                "per_class_DQ": {
                    name: values["dq"] for name, values in source_pq["per_class"].items()
                },
                "per_class_SQ": {
                    name: values["sq"] for name, values in source_pq["per_class"].items()
                },
                "count_error": count_error_stats(true_counts[mask], predicted_counts[mask]),
            }
        payload["per_source"] = per_source
    return payload


def delta_metrics(candidate: dict, baseline: dict) -> dict:
    return {
        key: float(candidate[key] - baseline[key])
        for key in ("R2", "mPQ+", "mDQ+", "mSQ+")
    }


def source_excluded_strength_audit(
    truth: np.ndarray,
    relabeled_by_strength: dict[float, np.ndarray],
    true_counts: np.ndarray,
    source_values: np.ndarray,
) -> dict:
    """Select correction strength without the institution being evaluated."""
    sources = sorted(np.unique(np.asarray(source_values).astype(str)))
    strengths = sorted(relabeled_by_strength)
    if len(strengths) < 2 or not sources:
        raise ValueError("source-excluded prior audit needs multiple strengths and at least one source")
    fit_metrics: dict[str, dict[str, dict]] = {}
    for held_source in sources:
        fit_mask = np.asarray(source_values).astype(str) != held_source
        if not fit_mask.any():
            raise ValueError("cannot select a correction strength with every patch held out")
        fit_metrics[held_source] = {
            f"{strength:g}": metric_payload(
                truth[fit_mask], prediction[fit_mask], true_counts[fit_mask], include_confusion=False
            )
            for strength, prediction in relabeled_by_strength.items()
        }

    result = {}
    step = float(np.min(np.diff(np.asarray(strengths, dtype=np.float64))))
    for metric in ("R2", "mPQ+"):
        selected_strengths = {}
        oof_prediction = np.zeros_like(truth)
        for held_source in sources:
            selected = max(
                strengths,
                key=lambda strength: (
                    fit_metrics[held_source][f"{strength:g}"][metric],
                    -strength,
                ),
            )
            selected_strengths[held_source] = float(selected)
            held_mask = np.asarray(source_values).astype(str) == held_source
            oof_prediction[held_mask] = relabeled_by_strength[selected][held_mask]
        selected_values = np.asarray(list(selected_strengths.values()), dtype=np.float64)
        result[metric] = {
            "selected_strength_excluding_each_source": selected_strengths,
            "strength_span": float(selected_values.max() - selected_values.min()),
            "stable_within_one_grid_step": bool(selected_values.max() - selected_values.min() <= step + 1.0e-12),
            "pooled_out_of_source": metric_payload(
                truth, oof_prediction, true_counts, np.asarray(source_values).astype(str)
            ),
        }
    result["fit_metrics_by_held_source_and_strength"] = fit_metrics
    result["strength_grid_step"] = step
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--diagnostic-json", type=Path, required=True)
    parser.add_argument("--candidate-curve", type=Path, default=None)
    parser.add_argument("--train-ids", type=Path, default=None)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--out-assignments", type=Path, default=None)
    parser.add_argument("--out-corrected-artifact", type=Path, default=None)
    args = parser.parse_args()

    diagnostic = json.loads(args.diagnostic_json.read_text())
    artifact_path = Path(diagnostic["prediction_artifact"])
    if not artifact_path.is_absolute():
        artifact_path = ROOT / artifact_path
    artifact = np.load(artifact_path)
    required = {
        "patch_ids", "predictions", "probability_patch_ids",
        "probability_instance_ids", "class_probs",
    }
    missing_keys = required - set(artifact.files)
    if missing_keys:
        raise RuntimeError(f"prediction artifact lacks probability fields: {sorted(missing_keys)}")

    checkpoint_epoch = int(diagnostic["checkpoint_epoch"])
    checkpoint = Path(diagnostic["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    run_dir = checkpoint.parent
    curve_path = args.candidate_curve or (run_dir / "training_curve.json")
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    train_ids_path = args.train_ids or Path(summary["args"]["train_ids"])
    if not train_ids_path.is_absolute():
        train_ids_path = ROOT / train_ids_path

    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    patch_ids = artifact["patch_ids"].astype(np.int32)
    forbidden = set(metadata.loc[metadata.split.eq("test"), "patch_id"].astype(int))
    if set(map(int, patch_ids)) & forbidden:
        raise RuntimeError("sampler-prior sweep refuses to read locked-test patches")
    val_rows = metadata.set_index("patch_id").loc[patch_ids]
    train_ids = np.load(train_ids_path).astype(np.int32)
    if set(map(int, train_ids)) & forbidden:
        raise RuntimeError("development-training IDs overlap the locked test set")
    train_rows = metadata.set_index("patch_id").loc[train_ids]

    target_prior = normalized_prior(train_rows[COUNT_COLUMNS].sum(axis=0).to_numpy(dtype=np.float64))
    sampled_prior, exposure_epochs = sampled_exposure_prior(
        json.loads(curve_path.read_text()), checkpoint_epoch
    )
    predictions = artifact["predictions"].astype(np.int32)
    truth = np.zeros_like(predictions)
    for index, patch_id in enumerate(patch_ids):
        label = np.load(args.prepared / "labels" / f"{int(patch_id):05d}.npy", allow_pickle=True).item()
        truth[index, ..., 0] = label["inst_map"]
        truth[index, ..., 1] = label["class_map"]
    true_counts = val_rows[COUNT_COLUMNS].to_numpy(dtype=np.int32)
    source_values = val_rows["source"].astype(str).to_numpy()

    raw = metric_payload(truth, predictions, true_counts, source_values)
    recorded = diagnostic.get("metrics", {})
    for key, recorded_key in (("R2", "val_R2"), ("mPQ+", "val_mPQ+")):
        if recorded_key in recorded and not np.isclose(raw[key], recorded[recorded_key], rtol=1e-7, atol=1e-9):
            raise RuntimeError(f"raw artifact {key} does not reproduce diagnostic: {raw[key]} vs {recorded[recorded_key]}")

    probability_patch_ids = artifact["probability_patch_ids"].astype(np.int32)
    probability_instance_ids = artifact["probability_instance_ids"].astype(np.int32)
    probabilities = artifact["class_probs"].astype(np.float32)
    sweeps = []
    assignments_by_strength: dict[float, np.ndarray] = {}
    adjusted_by_strength: dict[float, np.ndarray] = {}
    relabeled_by_strength: dict[float, np.ndarray] = {}
    for strength in sorted(set(args.strengths)):
        adjusted = apply_log_prior_correction(probabilities, sampled_prior, target_prior, strength)
        adjusted_by_strength[float(strength)] = adjusted
        assignments = adjusted.argmax(axis=1).astype(np.int8) + 1
        assignments_by_strength[float(strength)] = assignments
        relabeled = relabel_predictions(
            predictions, patch_ids, probability_patch_ids, probability_instance_ids, assignments
        )
        relabeled_by_strength[float(strength)] = relabeled
        metrics = metric_payload(truth, relabeled, true_counts, source_values)
        sweeps.append({"strength": float(strength), "metrics": metrics})

    pooled_zero = next((row for row in sweeps if row["strength"] == 0.0), None)
    if pooled_zero is None:
        raise RuntimeError("strength 0 is required to isolate pooled-instance typing")
    selected = {
        metric: max(sweeps, key=lambda row: (row["metrics"][metric], -row["strength"]))
        for metric in ("R2", "mPQ+")
    }
    source_excluded = source_excluded_strength_audit(
        truth, relabeled_by_strength, true_counts, source_values
    )
    for metric in ("R2", "mPQ+"):
        source_excluded[metric]["delta_vs_raw"] = delta_metrics(
            source_excluded[metric]["pooled_out_of_source"], raw
        )
        source_excluded[metric]["delta_vs_pooled_strength_0"] = delta_metrics(
            source_excluded[metric]["pooled_out_of_source"], pooled_zero["metrics"]
        )
    selected_mpq = selected["mPQ+"]
    corrected_artifact_path = args.out_corrected_artifact or args.out_report.with_name(
        f"{args.out_report.stem}_mpq_corrected_predictions.npz"
    )
    report = {
        "protocol": (
            "validation-only correction of the development-training central-224 nucleus-exposure prior; "
            "this is a proxy, not an exact pixel-loss label-shift correction; geometry fixed; locked test refused"
        ),
        "evaluation_set": f"{len(patch_ids)}-patch source-group-disjoint development validation",
        "diagnostic_json": str(args.diagnostic_json),
        "prediction_artifact": str(artifact_path),
        "candidate_curve": str(curve_path),
        "checkpoint_epoch": checkpoint_epoch,
        "sampled_exposure_epochs": exposure_epochs,
        "target_natural_train_prior": dict(zip(CLASS_NAMES, map(float, target_prior))),
        "actual_sampled_exposure_prior": dict(zip(CLASS_NAMES, map(float, sampled_prior))),
        "raw_decoder": raw,
        "pooled_instance_strength_0": pooled_zero["metrics"],
        "pooled_instance_delta_vs_raw": delta_metrics(pooled_zero["metrics"], raw),
        "sweep": sweeps,
        "selected": {
            metric: {
                **row,
                "delta_vs_raw": delta_metrics(row["metrics"], raw),
                "delta_vs_pooled_strength_0": delta_metrics(row["metrics"], pooled_zero["metrics"]),
                "at_strength_boundary": row["strength"] in (min(args.strengths), max(args.strengths)),
            }
            for metric, row in selected.items()
        },
        "leave_one_source_out": source_excluded,
        "selected_mPQ_corrected_prediction_artifact": str(corrected_artifact_path),
        "promotion_rule": (
            "A correction is evidence only if its selected metric improves over pooled strength 0; "
            "promotion also requires the resulting endpoint to improve over the raw decoder, the source-excluded "
            "pooled metric to improve, and held-source strength selections to stay within one grid step."
        ),
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2))

    assignments_path = args.out_assignments or args.out_report.with_name(f"{args.out_report.stem}_assignments.npz")
    selected_r2 = selected["R2"]
    np.savez_compressed(
        assignments_path,
        patch_ids=probability_patch_ids,
        instance_ids=probability_instance_ids,
        r2_strength=np.asarray(selected_r2["strength"], dtype=np.float32),
        r2_assignments=assignments_by_strength[selected_r2["strength"]],
        mpq_strength=np.asarray(selected_mpq["strength"], dtype=np.float32),
        mpq_assignments=assignments_by_strength[selected_mpq["strength"]],
    )
    corrected_prediction = relabeled_by_strength[selected_mpq["strength"]]
    corrected_counts = np.asarray(
        [central_crop_counts(patch[..., 0], patch[..., 1]) for patch in corrected_prediction],
        dtype=np.int32,
    )
    corrected_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        corrected_artifact_path,
        patch_ids=patch_ids,
        predictions=corrected_prediction,
        predicted_counts=corrected_counts,
        probability_patch_ids=probability_patch_ids,
        probability_instance_ids=probability_instance_ids,
        class_probs=adjusted_by_strength[selected_mpq["strength"]],
    )
    print(json.dumps({
        "evaluation_set": report["evaluation_set"],
        "raw": {key: raw[key] for key in ("R2", "mPQ+")},
        "pooled_strength_0": {key: pooled_zero["metrics"][key] for key in ("R2", "mPQ+")},
        "selected": {
            metric: {"strength": row["strength"], metric: row["metrics"][metric]}
            for metric, row in selected.items()
        },
        "report": str(args.out_report),
    }, indent=2))


if __name__ == "__main__":
    main()
