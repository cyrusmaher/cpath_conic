#!/usr/bin/env python
"""Render stratified CoNIC panels, animations, gallery, and agent triage."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cpath_conic.data import load_metadata, patch_count_from_maps
from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS
from cpath_conic.experiment_metrics import evaluate_fixed_masks
from cpath_conic.metrics import instance_type_confusion, multiclass_pq_plus
from cpath_conic.dashboard_model import build_trajectory, normalize_rows, outcome_tally
from cpath_conic.visuals import render_case, triage_cases, write_gallery


def choose_ids(
    metadata,
    split: str,
    max_cases: int,
    seed: int,
    predicted_counts: np.ndarray | None = None,
) -> tuple[list[int], dict[int, str]]:
    """Mix difficult, rare, source-extreme, and random cases for visual review.

    Returns the ordered patch ids and a reason string per patch (why it was
    selected), so the gallery can tell a reader these cases are curated, not random.
    Ground-truth-aware ranking is retrospective test-set auditing only. It must
    never be used to select a model or tune a threshold.
    """
    subset = metadata.loc[metadata.split == split].sort_values("patch_id")
    if len(subset) <= max_cases:
        ids = subset.patch_id.astype(int).tolist()
        return ids, {patch_id: "Every test patch is shown" for patch_id in ids}
    rng = np.random.default_rng(seed)
    # (patch_id, reason) in priority order; first reason wins on dedup.
    picks: list[tuple[int, str]] = []
    if predicted_counts is not None:
        patch_ids = subset.patch_id.to_numpy(dtype=np.int32)
        truth = subset[COUNT_COLUMNS].to_numpy(dtype=np.float64)
        predicted = predicted_counts[patch_ids].astype(np.float64)
        residual = predicted - truth
        audit = subset[["patch_id", "source"]].copy()
        audit["total_abs_error"] = np.abs(residual).sum(axis=1)
        audit["largest_class_error"] = np.abs(residual).max(axis=1)
        audit["signed_error"] = residual.sum(axis=1)
        dpath_zero_gt = []
        dpath_mask = subset.source.astype(str).str.lower().eq("dpath").to_numpy()
        for class_index in range(len(CLASS_NAMES)):
            candidates = np.flatnonzero(dpath_mask & (truth[:, class_index] == 0) & (predicted[:, class_index] > 0))
            for offset in candidates:
                dpath_zero_gt.append((int(predicted[offset, class_index]), int(patch_ids[offset])))
        dpath_zero_gt.sort(reverse=True)
        for _, patch_id in dpath_zero_gt[:12]:
            picks.append((patch_id, "DPath patch with a large predicted count for a cell type absent in ground truth"))
        n_tail = max(2, max_cases // 4)
        for patch_id in audit.nlargest(n_tail, "total_abs_error").patch_id.astype(int):
            picks.append((int(patch_id), "Among the largest total count errors on the test set"))
        for patch_id in audit.nlargest(n_tail, "largest_class_error").patch_id.astype(int):
            picks.append((int(patch_id), "Has one of the largest single-class count errors"))
        for source, rows in audit.groupby("source"):
            for patch_id in rows.nsmallest(1, "signed_error").patch_id.astype(int):
                picks.append((int(patch_id), f"Most under-counted patch from source “{source}”"))
            for patch_id in rows.nlargest(1, "signed_error").patch_id.astype(int):
                picks.append((int(patch_id), f"Most over-counted patch from source “{source}”"))

    for name in ["eosinophil", "plasma", "neutrophil"]:
        column = f"count_{name}"
        for patch_id in subset.sort_values(column, ascending=False).head(max(1, max_cases // 12)).patch_id.astype(int):
            picks.append((int(patch_id), f"Contains an unusually high number of {name}s (a rare cell type)"))
    for patch_id in rng.choice(subset.patch_id.to_numpy(), size=min(max_cases, len(subset)), replace=False):
        picks.append((int(patch_id), "Random representative sample"))

    reasons: dict[int, str] = {}
    for patch_id, reason in picks:
        reasons.setdefault(patch_id, reason)
    ordered = list(reasons)[:max_cases]
    return ordered, {patch_id: reasons[patch_id] for patch_id in ordered}


def e43_interim_note(experiment: dict) -> str:
    """Format E43's provisional evidence without coupling it to other schemas."""
    if experiment.get("id") != "E43":
        return ""
    interim = experiment.get("interim_validation_observation", {})
    candidate = interim.get("candidate", {})
    delta = interim.get("exact_matched_control_delta", {})
    if not (candidate and delta):
        return ""
    note = (
        f" Provisional only on the {interim['evaluation_set']}: auxiliary weight "
        f"{candidate['instance_type_loss_weight']:g}, LR {candidate['learning_rate']:g}, epoch "
        f"{candidate['epoch']} reached R²/mPQ+ {candidate['R2']:.4f}/{candidate['mPQ+']:.4f}; "
        f"exact control deltas are {delta['R2']:+.4f}/{delta['mPQ+']:+.4f}, with mDQ+/mSQ+ "
        f"deltas {delta['mDQ+']:+.4f}/{delta['mSQ+']:+.4f}. Type NLL changes by "
        f"{delta['instance_type_NLL']:+.4f}, pixel type accuracy by "
        f"{delta['pixel_type_accuracy']:+.4f}, and correctly typed matched nuclei by "
        f"{delta['correctly_typed_nuclei']:+d}. The complete staged LR/weight grid remains "
        "required before selection."
    )
    if interim.get("interpretation"):
        note += f" Mechanism audit: {interim['interpretation']}"
    endpoint = interim.get("completed_low_lr_endpoint", {})
    if endpoint:
        note += (
            f" The completed low-LR endpoint reaches R²/mPQ+ "
            f"{endpoint['R2']:.4f}/{endpoint['mPQ+']:.4f}; exact low-LR control deltas are "
            f"{endpoint['exact_control_delta_R2']:+.4f}/{endpoint['exact_control_delta_mPQ+']:+.4f}, "
            "but both absolute scores remain below independently selected ordinary loss. "
            f"Eosinophil R² is {endpoint['eosinophil_R2']:.4f} at a predicted/true count ratio of "
            f"{endpoint['eosinophil_count_ratio']:.2f}, explaining the negative macro R². The "
            f"mSQ+ delta of {endpoint['exact_control_delta_mSQ+']:+.4f} is chiefly a zero-to-first-match "
            "discontinuity for rare classes, not broad mask-shape improvement."
        )
    interior = interim.get("interior_lr_epoch5", {})
    if interior:
        note += (
            f" At interior LR {interior['learning_rate']:g}, epoch {interior['epoch']}, the candidate reaches "
            f"R²/mPQ+ {interior['R2']:.4f}/{interior['mPQ+']:.4f}; exact-control R²/mPQ+/mDQ+/mSQ+ "
            f"deltas are {interior['exact_control_delta_R2']:+.4f}/"
            f"{interior['exact_control_delta_mPQ+']:+.4f}/"
            f"{interior['exact_control_delta_mDQ+']:+.4f}/"
            f"{interior['exact_control_delta_mSQ+']:+.4f}. All six class PQ values and CRAG, DPath, and "
            "GLaS mPQ+ improve while binary geometry and count tails remain safe. This is promising but "
            "unselected pending epoch 10, LR 3e-4, the auxiliary-dose stage, and exact-control spatial-JS backfill."
        )
    interior_endpoint = interim.get("interior_lr_epoch10", {})
    if interior_endpoint:
        note += (
            f" The midpoint reverses at epoch {interior_endpoint['epoch']}: candidate R²/mPQ+ is "
            f"{interior_endpoint['R2']:.4f}/{interior_endpoint['mPQ+']:.4f}, with exact-control "
            f"R²/mPQ+/mDQ+/mSQ+ deltas {interior_endpoint['exact_control_delta_R2']:+.4f}/"
            f"{interior_endpoint['exact_control_delta_mPQ+']:+.4f}/"
            f"{interior_endpoint['exact_control_delta_mDQ+']:+.4f}/"
            f"{interior_endpoint['exact_control_delta_mSQ+']:+.4f}. Neutrophil PQ falls "
            f"{interior_endpoint['neutrophil_PQ_delta']:+.4f} as the model removes 349 false positives but "
            "also loses 81 typed true positives; DPath and GLaS regress. This endpoint is rejected pending "
            "the LR 3e-4 bracket and lower-dose stage."
        )
    upper = interim.get("upper_lr_epoch5", {})
    if upper:
        note += (
            f" At upper LR {upper['learning_rate']:g}, epoch {upper['epoch']}, candidate R²/mPQ+ is "
            f"{upper['R2']:.4f}/{upper['mPQ+']:.4f}; exact-control R²/mPQ+/mDQ+/mSQ+ deltas are "
            f"{upper['exact_control_delta_R2']:+.4f}/{upper['exact_control_delta_mPQ+']:+.4f}/"
            f"{upper['exact_control_delta_mDQ+']:+.4f}/{upper['exact_control_delta_mSQ+']:+.4f}. "
            "The mPQ+ movement is plasma-led, while CRAG mDQ+ and GLaS mSQ+ cross the source-safety "
            "boundary and correctly typed nuclei fall by 131. This checkpoint is diagnostic only; epoch 10 "
            "must complete the bracket before lower-dose selection."
        )
    upper_endpoint = interim.get("completed_upper_lr_endpoint", {})
    if upper_endpoint:
        note += (
            f" By upper-LR epoch {upper_endpoint['epoch']}, candidate R²/mPQ+ is "
            f"{upper_endpoint['R2']:.4f}/{upper_endpoint['mPQ+']:.4f} and the exact-control deltas reverse to "
            f"{upper_endpoint['exact_control_delta_R2']:+.4f}/{upper_endpoint['exact_control_delta_mPQ+']:+.4f}. "
            "Although type NLL and pixel accuracy improve, 1,167 extra spurious nuclei coincide with 198 "
            "fewer correctly typed nuclei, worse count tails, and DQ losses in CRAG, DPath, and GLaS."
        )
    stage_a = interim.get("stage_a_selection", {})
    if stage_a:
        note += (
            f" The completed stage-A bracket selects interior LR {stage_a['learning_rate_for_mPQ+']:g} for "
            "both independently scored objectives; only auxiliary weights 0.05 and 0.25 at that LR proceed "
            "to the validation-only dose stage."
        )
    lower_dose = interim.get("lower_dose_epoch5", {})
    if lower_dose:
        note += (
            f" At auxiliary weight {lower_dose['instance_type_loss_weight']:g}, epoch {lower_dose['epoch']}, "
            f"candidate R²/mPQ+ is {lower_dose['R2']:.4f}/{lower_dose['mPQ+']:.4f}; exact-control "
            f"R²/mPQ+/mDQ+/mSQ+ deltas are {lower_dose['exact_control_delta_R2']:+.4f}/"
            f"{lower_dose['exact_control_delta_mPQ+']:+.4f}/{lower_dose['exact_control_delta_mDQ+']:+.4f}/"
            f"{lower_dose['exact_control_delta_mSQ+']:+.4f}. All six class PQs and every source aggregate "
            "improve, with 446 more correctly typed and 123 fewer spurious nuclei. This broad mPQ-specific "
            "signal remains provisional until epoch 10, weight 0.25, and paired spatial-JS backfill."
        )
    return note


def _load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def _fixed_mask_method_metrics(
    prepared: Path,
    features_path: Path,
    probabilities_path: Path,
    cache_path: Path,
    calibration_report: dict | None = None,
) -> tuple[dict, dict[str, dict], np.ndarray, dict]:
    """Evaluate one fixed-mask method globally and by source/institution proxy."""
    features = np.load(features_path)
    probabilities = np.load(probabilities_path)
    cache = np.load(cache_path)
    patch_ids = features["patch_ids"].astype(np.int32)
    instance_ids = features["instance_ids"].astype(np.int32)
    if not np.array_equal(patch_ids, probabilities["patch_ids"]) or not np.array_equal(instance_ids, probabilities["instance_ids"]):
        raise ValueError(f"Feature/probability records are not aligned for {probabilities_path}")
    class_probabilities = probabilities["class_probs"].astype(np.float32)
    if calibration_report:
        calibration = calibration_report["calibration"]
        temperature = float(calibration["temperature"])
        biases = np.asarray([calibration["biases"][name] for name in CLASS_NAMES], dtype=np.float64)
        logits = np.log(np.clip(class_probabilities, 1e-9, 1.0)) / temperature + biases[None, :]
        assignments = logits.argmax(axis=1).astype(np.int8) + 1
    else:
        assignments = class_probabilities.argmax(axis=1).astype(np.int8) + 1

    metadata = load_metadata(prepared).sort_values("patch_id")
    labels = features["labels"].astype(np.int8)
    ious = features["ious"].astype(np.float32)
    central = cache["central"].astype(bool)
    gt_full_counts = cache["gt_full_counts"].astype(np.int32)
    overall = evaluate_fixed_masks(assignments, labels, ious, patch_ids, central, metadata, gt_full_counts, "test")
    by_source = {}
    for source in sorted(metadata.loc[metadata.split == "test", "source"].dropna().unique()):
        source_metadata = metadata.loc[metadata.source == source].copy()
        by_source[str(source)] = evaluate_fixed_masks(
            assignments, labels, ious, patch_ids, central, source_metadata, gt_full_counts, "test"
        )
    return overall, by_source, gt_full_counts, {
        "assignments": assignments,
        "labels": labels,
        "ious": ious,
        "patch_ids": patch_ids,
        "metadata": metadata,
    }


def _detection_aware_confusion(diagnostic: dict, gt_full_counts: np.ndarray) -> dict:
    """Classification confusion augmented with missed GT and spurious predictions."""
    metadata = diagnostic["metadata"]
    test_ids = metadata.loc[metadata.split == "test", "patch_id"].to_numpy(dtype=np.int32)
    patch_ids = diagnostic["patch_ids"]
    labels = diagnostic["labels"]
    ious = diagnostic["ious"]
    assignments = diagnostic["assignments"]
    on_test = np.isin(patch_ids, test_ids)
    matched = on_test & (labels > 0) & (ious > 0.5)
    matrix = np.zeros((len(CLASS_NAMES) + 1, len(CLASS_NAMES) + 1), dtype=np.int64)
    np.add.at(matrix, (labels[matched] - 1, assignments[matched] - 1), 1)
    matched_gt = np.bincount(labels[matched] - 1, minlength=len(CLASS_NAMES))
    gt_totals = gt_full_counts[test_ids].sum(axis=0).astype(np.int64)
    matrix[: len(CLASS_NAMES), -1] = np.maximum(gt_totals - matched_gt, 0)
    unmatched_predictions = on_test & ~matched
    np.add.at(
        matrix,
        (np.full(int(unmatched_predictions.sum()), len(CLASS_NAMES)), assignments[unmatched_predictions] - 1),
        1,
    )
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=np.float64), where=row_totals > 0)
    return {
        "rows": [*CLASS_NAMES, "spurious prediction"],
        "columns": [*CLASS_NAMES, "missed GT"],
        "counts": matrix.tolist(),
        "row_normalized": normalized.tolist(),
    }


def _map_detection_aware_confusion(true_maps: np.ndarray, pred_maps: np.ndarray) -> dict:
    """Match instances geometrically, then expose classification and detection errors."""
    diagnostic = instance_type_confusion(true_maps, pred_maps)
    matrix = np.asarray(diagnostic["matrix"], dtype=np.int64)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=np.float64), where=row_totals > 0)
    return {
        "rows": [*CLASS_NAMES, "spurious prediction"],
        "columns": [*CLASS_NAMES, "missed GT"],
        "counts": matrix.tolist(),
        "row_normalized": normalized.tolist(),
    }


def _count_r2(true_counts: np.ndarray, pred_counts: np.ndarray) -> tuple[float, dict[str, float]]:
    per_class: dict[str, float] = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        truth = true_counts[:, class_index].astype(np.float64)
        prediction = pred_counts[:, class_index].astype(np.float64)
        denominator = float(np.square(truth - truth.mean()).sum())
        per_class[class_name] = (
            float(1.0 - np.square(prediction - truth).sum() / denominator)
            if denominator > 0
            else float("nan")
        )
    finite = [value for value in per_class.values() if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan"), per_class


def _direct_prediction_breakdown(
    prepared: Path,
    baseline: dict,
    baseline_sources: dict[str, dict],
    baseline_diagnostic: dict,
    gt_full_counts: np.ndarray,
    predictions_path: Path,
    counts_path: Path,
    intermediate_counts_path: Path | None,
    best_name: str,
) -> dict:
    """Build subgroup diagnostics directly from deployed maps and count outputs."""
    metadata = load_metadata(prepared).sort_values("patch_id")
    test = metadata.loc[metadata.split == "test"].sort_values("patch_id").reset_index(drop=True)
    test_ids = test.patch_id.to_numpy(dtype=np.int32)
    predictions = np.load(predictions_path, mmap_mode="r")
    all_counts = np.load(counts_path)
    if predictions.shape[0] <= int(test_ids.max()) or all_counts.shape[0] <= int(test_ids.max()):
        raise ValueError("Prediction/count arrays do not cover all test patch IDs")
    if all_counts.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"Expected six count columns, got {all_counts.shape}")

    true_maps = []
    for patch_id in test_ids:
        label = np.load(prepared / "labels" / f"{int(patch_id):05d}.npy", allow_pickle=True).item()
        true_maps.append(np.stack([label["inst_map"], label["class_map"]], axis=-1))
    true_maps = np.stack(true_maps)
    pred_maps = np.asarray(predictions[test_ids])
    best_counts = np.asarray(all_counts[test_ids])
    positions = {int(patch_id): index for index, patch_id in enumerate(test_ids)}

    observed_full_counts = np.stack(
        [patch_count_from_maps(item[..., 0], item[..., 1]) for item in true_maps]
    )
    if not np.array_equal(observed_full_counts, gt_full_counts[test_ids]):
        raise ValueError("Direct GT maps disagree with the baseline full-mask support cache")

    def evaluate(rows) -> dict:
        indexes = np.asarray([positions[int(patch_id)] for patch_id in rows.patch_id], dtype=np.int32)
        pq = multiclass_pq_plus(true_maps[indexes], pred_maps[indexes])
        truth = rows[COUNT_COLUMNS].to_numpy(dtype=np.float64)
        predicted = best_counts[indexes]
        r2, per_class_r2 = _count_r2(truth, predicted)
        return {
            "R2": r2,
            "per_class_R2": per_class_r2,
            "mPQ+": pq["mPQ+"],
            "per_class_pq": pq["per_class"],
            "predicted_counts": predicted,
        }

    best = evaluate(test)
    best_sources = {str(source): evaluate(test.loc[test.source == source]) for source in sorted(test.source.unique())}

    def support(rows, class_index: int | None = None) -> dict:
        patch_ids = rows.patch_id.to_numpy(dtype=np.int32)
        central_counts = rows[COUNT_COLUMNS].to_numpy(dtype=np.int64)
        full_counts = gt_full_counts[patch_ids]
        return {
            "patches": int(len(rows)),
            "count_gt": int(central_counts.sum() if class_index is None else central_counts[:, class_index].sum()),
            "mask_gt": int(full_counts.sum() if class_index is None else full_counts[:, class_index].sum()),
        }

    def row(label: str, samples: dict, base: dict, candidate: dict, class_name: str | None = None) -> dict:
        if class_name is None:
            base_r2, candidate_r2 = base["R2"], candidate["R2"]
            base_pq, candidate_pq = base["mPQ+"], candidate["mPQ+"]
            base_dq = float(np.mean([base["per_class_pq"][name]["dq"] for name in CLASS_NAMES]))
            candidate_dq = float(np.mean([candidate["per_class_pq"][name]["dq"] for name in CLASS_NAMES]))
            base_sq = float(np.mean([base["per_class_pq"][name]["sq"] for name in CLASS_NAMES]))
            candidate_sq = float(np.mean([candidate["per_class_pq"][name]["sq"] for name in CLASS_NAMES]))
        else:
            base_r2, candidate_r2 = base["per_class_R2"][class_name], candidate["per_class_R2"][class_name]
            base_pq, candidate_pq = base["per_class_pq"][class_name]["pq"], candidate["per_class_pq"][class_name]["pq"]
            base_dq, candidate_dq = base["per_class_pq"][class_name]["dq"], candidate["per_class_pq"][class_name]["dq"]
            base_sq, candidate_sq = base["per_class_pq"][class_name]["sq"], candidate["per_class_pq"][class_name]["sq"]
        return {
            "label": label,
            "samples": samples,
            "baseline_r2": base_r2,
            "best_r2": candidate_r2,
            "baseline_pq": base_pq,
            "best_pq": candidate_pq,
            "baseline_dq": base_dq,
            "best_dq": candidate_dq,
            "baseline_sq": base_sq,
            "best_sq": candidate_sq,
        }

    by_class = [
        row(class_name, support(test, class_index), baseline, best, class_name)
        for class_index, class_name in enumerate(CLASS_NAMES)
    ]
    by_institution = []
    by_both = []
    for source in sorted(baseline_sources):
        source_rows = test.loc[test.source == source]
        by_institution.append(row(source, support(source_rows), baseline_sources[source], best_sources[source]))
        for class_index, class_name in enumerate(CLASS_NAMES):
            by_both.append(
                row(
                    f"{source} · {class_name}",
                    support(source_rows, class_index),
                    baseline_sources[source],
                    best_sources[source],
                    class_name,
                )
            )

    baseline_counts = np.asarray(baseline["predicted_counts"])
    if baseline_counts.shape != best_counts.shape:
        raise ValueError(f"Baseline and best test-count shapes differ: {baseline_counts.shape} vs {best_counts.shape}")
    intermediate_counts = None
    if intermediate_counts_path is not None and intermediate_counts_path.exists():
        intermediate_counts = np.load(intermediate_counts_path)[test_ids]
    scatter_points = []
    for patch_index, metadata_row in test.iterrows():
        for class_index, class_name in enumerate(CLASS_NAMES):
            scatter_points.append(
                {
                    "patch_id": int(metadata_row.patch_id),
                    "patch_info": str(metadata_row.patch_info),
                    "source": str(metadata_row.source),
                    "source_group": str(metadata_row.source_group),
                    "class_name": class_name,
                    "gt": int(metadata_row[COUNT_COLUMNS[class_index]]),
                    "baseline_pred": int(baseline_counts[patch_index, class_index]),
                    "best_pred": int(best_counts[patch_index, class_index]),
                    **(
                        {"rotation_pred": int(intermediate_counts[patch_index, class_index])}
                        if intermediate_counts is not None
                        else {}
                    ),
                }
            )
    return {
        "baseline_name": "Initial CellViT baseline",
        "best_name": best_name,
        "support_note": "Support is (test patches; central-crop GT cells used by count R²; full-mask GT cells used by PQ). CoNIC source repository is used as the institution/domain proxy. Test labels are used only for retrospective visualization and reporting.",
        "by_class": by_class,
        "by_institution": by_institution,
        "by_both": by_both,
        "previous_name": None,
        "intermediate_name": "E32 HoVer-Net TTA mask-derived counts" if intermediate_counts is not None else None,
        "confusions": [
            {"name": "Initial CellViT baseline", **_detection_aware_confusion(baseline_diagnostic, gt_full_counts)},
            {
                "name": "Final rare-class-trained HoVer-Net",
                **_map_detection_aware_confusion(true_maps, pred_maps),
            },
        ],
        "scatter_points": scatter_points,
    }


def build_subgroup_breakdown(
    prepared: Path | None,
    runs_root: Path | None,
    baseline_posthoc: Path | None,
    best_predictions_path: Path | None = None,
    best_counts_path: Path | None = None,
    intermediate_counts_path: Path | None = None,
    best_name: str = "Final rare-class-trained HoVer-Net with six-view TTA",
) -> dict:
    """Build baseline-vs-best class, institution, and joint metric tables."""
    if prepared is None or runs_root is None or baseline_posthoc is None:
        return {}
    direct_best_name = best_name
    baseline_paths = {
        "features": runs_root / "full" / "features.npz",
        "probabilities": runs_root / "full" / "cell_probabilities.npz",
        "cache": baseline_posthoc.parent / "fixed_mask_cache.npz",
    }
    focal_paths = {
        "features": runs_root / "hv_tta" / "features.npz",
        "probabilities": runs_root / "e17_focal" / "cell_probabilities.npz",
        "cache": runs_root / "e17_focal" / "posthoc" / "fixed_mask_cache.npz",
        "report": None,
    }
    rotation_paths = {
        "features": runs_root / "hv_tta_rot" / "features.npz",
        "probabilities": runs_root / "e21_rot_tta_focal" / "cell_probabilities.npz",
        "cache": runs_root / "e21_rot_tta_focal" / "fixed_mask_cache.npz",
        "report": None,
    }
    if all(path.exists() for path in [rotation_paths["features"], rotation_paths["probabilities"], rotation_paths["cache"]]):
        best_paths = rotation_paths
        best_name = "Tuned HV + flip/rotation-TTA + complement-balanced focal type head"
    elif all(path.exists() for path in [focal_paths["probabilities"], focal_paths["cache"]]):
        best_paths = focal_paths
        best_name = "Tuned HV + flip-TTA + complement-balanced focal type head"
    else:
        best_paths = {
            "features": runs_root / "hv_tta" / "features.npz",
            "probabilities": runs_root / "hv_tta" / "cell_probabilities.npz",
            "cache": runs_root / "hv_tta" / "posthoc" / "fixed_mask_cache.npz",
            "report": runs_root / "hv_tta" / "posthoc" / "posthoc_report.json",
        }
        best_name = "Tuned HV + flip-TTA + validation vector calibration"
    if not all(path.exists() for path in baseline_paths.values()):
        return {}
    baseline, baseline_sources, gt_full_counts, baseline_diagnostic = _fixed_mask_method_metrics(
        prepared, baseline_paths["features"], baseline_paths["probabilities"], baseline_paths["cache"]
    )
    if (
        best_predictions_path is not None
        and best_counts_path is not None
        and best_predictions_path.exists()
        and best_counts_path.exists()
    ):
        return _direct_prediction_breakdown(
            prepared,
            baseline,
            baseline_sources,
            baseline_diagnostic,
            gt_full_counts,
            best_predictions_path,
            best_counts_path,
            intermediate_counts_path,
            direct_best_name,
        )

    required = [path for path in best_paths.values() if path is not None]
    if not all(path.exists() for path in required):
        return {}
    best_report = _load_json(best_paths["report"])
    best, best_sources, best_gt_full_counts, best_diagnostic = _fixed_mask_method_metrics(
        prepared,
        best_paths["features"],
        best_paths["probabilities"],
        best_paths["cache"],
        calibration_report=best_report,
    )
    previous = None
    previous_diagnostic = None
    previous_name = None
    if "flip/rotation-TTA" in best_name:
        previous, _, previous_gt_full_counts, previous_diagnostic = _fixed_mask_method_metrics(
            prepared,
            focal_paths["features"],
            focal_paths["probabilities"],
            focal_paths["cache"],
        )
        previous_name = "Flip-TTA + complement-balanced focal (R² high-water)"
        if not np.array_equal(gt_full_counts, previous_gt_full_counts):
            raise ValueError("Ground-truth support differs for the previous-best subgroup cache")
    elif best_name.startswith("Tuned HV + flip-TTA + complement"):
        previous_report = _load_json(runs_root / "hv_tta" / "posthoc" / "posthoc_report.json")
        previous, _, previous_gt_full_counts, previous_diagnostic = _fixed_mask_method_metrics(
            prepared,
            runs_root / "hv_tta" / "features.npz",
            runs_root / "hv_tta" / "cell_probabilities.npz",
            runs_root / "hv_tta" / "posthoc" / "fixed_mask_cache.npz",
            calibration_report=previous_report,
        )
        previous_name = "Tuned HV + flip-TTA + validation vector calibration"
        if not np.array_equal(gt_full_counts, previous_gt_full_counts):
            raise ValueError("Ground-truth support differs for the previous-best subgroup cache")
    if not np.array_equal(gt_full_counts, best_gt_full_counts):
        raise ValueError("Ground-truth support differs between baseline and best subgroup caches")

    metadata = load_metadata(prepared).sort_values("patch_id")
    pq_route = None
    source_gate_report = _load_json(ROOT / "outputs" / "conic_experiments" / "e30_e21_e27_source_gated_mpq.json")
    e27_paths = {
        "features": runs_root / "e27_e26_masks_base_rot_tokens" / "features.npz",
        "probabilities": runs_root / "e27_mpq_final" / "cell_probabilities.npz",
        "cache": runs_root / "e27_e26_masks_base_rot_tokens" / "fixed_mask_cache.npz",
    }
    if source_gate_report and all(path.exists() for path in e27_paths.values()) and "flip/rotation-TTA" in best_name:
        e27, e27_sources, e27_gt_counts, e27_diagnostic = _fixed_mask_method_metrics(
            prepared, e27_paths["features"], e27_paths["probabilities"], e27_paths["cache"]
        )
        if not np.array_equal(gt_full_counts, e27_gt_counts):
            raise ValueError("Ground-truth support differs for the E30 source-routed candidate")
        pq_route = source_gate_report["selected_route"]
        selected_test = source_gate_report["selected_test"]
        best["mPQ+"] = selected_test["mPQ+"]
        best["per_class_pq"] = selected_test["per_class_pq"]
        for source, method_name in pq_route.items():
            if method_name == "b":
                best_sources[source]["mPQ+"] = e27_sources[source]["mPQ+"]
                best_sources[source]["per_class_pq"] = e27_sources[source]["per_class_pq"]

        source_by_patch = metadata.set_index("patch_id").source.to_dict()
        combined_diagnostic = {"metadata": metadata}
        for key in ("assignments", "labels", "ious", "patch_ids"):
            left = best_diagnostic[key]
            right = e27_diagnostic[key]
            left_keep = np.asarray([pq_route.get(source_by_patch.get(int(patch)), "a") == "a" for patch in best_diagnostic["patch_ids"]])
            right_keep = np.asarray([pq_route.get(source_by_patch.get(int(patch)), "a") == "b" for patch in e27_diagnostic["patch_ids"]])
            combined_diagnostic[key] = np.concatenate([left[left_keep], right[right_keep]])
        best_diagnostic = combined_diagnostic

    test = metadata.loc[metadata.split == "test"]
    e28_count_path = ROOT / "outputs" / "conic_experiments" / "e28_e22_e27_count_ensemble_counts.npy"
    count_ensemble_path = e28_count_path if e28_count_path.exists() else runs_root / "e22_count_ensemble" / "counts.npy"
    rotation_counts = None
    if "flip/rotation-TTA" in best_name and count_ensemble_path.exists():
        rotation_counts = best["predicted_counts"].copy()
        ensemble_counts = np.load(count_ensemble_path)

        def replace_count_metrics(metrics: dict, rows) -> None:
            predicted = ensemble_counts[rows.patch_id.to_numpy(dtype=np.int32)]
            true = rows[COUNT_COLUMNS].to_numpy(dtype=np.float64)
            per_class = {}
            for class_index, class_name in enumerate(CLASS_NAMES):
                centered = true[:, class_index] - true[:, class_index].mean()
                denominator = float(np.square(centered).sum())
                per_class[class_name] = (
                    float(1.0 - np.square(predicted[:, class_index] - true[:, class_index]).sum() / denominator)
                    if denominator > 0
                    else float("nan")
                )
            metrics["R2"] = float(np.mean([value for value in per_class.values() if np.isfinite(value)]))
            metrics["per_class_R2"] = per_class
            metrics["predicted_counts"] = predicted

        replace_count_metrics(best, test)
        for source, source_metrics in best_sources.items():
            replace_count_metrics(source_metrics, test.loc[test.source == source])
        best_name = (
            "Rotation-TTA masks + validation-selected E22/E27 count blend"
            if count_ensemble_path == e28_count_path
            else "Rotation-TTA masks + validation-selected E17/E21 count blend"
        )
        if pq_route is not None:
            best_name = "Current best: metric-specific count blend + source-routed masks/types"

    def support(rows, class_index: int | None = None) -> dict:
        patch_ids = rows.patch_id.to_numpy(dtype=np.int32)
        central_counts = rows[COUNT_COLUMNS].to_numpy(dtype=np.int64)
        full_counts = gt_full_counts[patch_ids]
        if class_index is None:
            central_n = int(central_counts.sum())
            full_n = int(full_counts.sum())
        else:
            central_n = int(central_counts[:, class_index].sum())
            full_n = int(full_counts[:, class_index].sum())
        return {"patches": int(len(rows)), "count_gt": central_n, "mask_gt": full_n}

    def row(label: str, samples: dict, base_r2, best_r2, base_pq, best_pq, base_dq, best_dq, base_sq, best_sq) -> dict:
        return {
            "label": label,
            "samples": samples,
            "baseline_r2": base_r2,
            "best_r2": best_r2,
            "baseline_pq": base_pq,
            "best_pq": best_pq,
            "baseline_dq": base_dq,
            "best_dq": best_dq,
            "baseline_sq": base_sq,
            "best_sq": best_sq,
        }

    def mean_component(metrics: dict, component: str) -> float:
        return float(np.mean([metrics["per_class_pq"][name][component] for name in CLASS_NAMES]))

    by_class = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        by_class.append(
            row(
                class_name,
                support(test, class_index),
                baseline["per_class_R2"][class_name],
                best["per_class_R2"][class_name],
                baseline["per_class_pq"][class_name]["pq"],
                best["per_class_pq"][class_name]["pq"],
                baseline["per_class_pq"][class_name]["dq"],
                best["per_class_pq"][class_name]["dq"],
                baseline["per_class_pq"][class_name]["sq"],
                best["per_class_pq"][class_name]["sq"],
            )
        )

    by_institution = []
    by_both = []
    for source in sorted(baseline_sources):
        source_rows = test.loc[test.source == source]
        base_source = baseline_sources[source]
        best_source = best_sources[source]
        by_institution.append(
            row(
                source,
                support(source_rows),
                base_source["R2"],
                best_source["R2"],
                base_source["mPQ+"],
                best_source["mPQ+"],
                mean_component(base_source, "dq"),
                mean_component(best_source, "dq"),
                mean_component(base_source, "sq"),
                mean_component(best_source, "sq"),
            )
        )
        for class_index, class_name in enumerate(CLASS_NAMES):
            by_both.append(
                row(
                    f"{source} · {class_name}",
                    support(source_rows, class_index),
                    base_source["per_class_R2"][class_name],
                    best_source["per_class_R2"][class_name],
                    base_source["per_class_pq"][class_name]["pq"],
                    best_source["per_class_pq"][class_name]["pq"],
                    base_source["per_class_pq"][class_name]["dq"],
                    best_source["per_class_pq"][class_name]["dq"],
                    base_source["per_class_pq"][class_name]["sq"],
                    best_source["per_class_pq"][class_name]["sq"],
                )
            )
    test_sorted = test.sort_values("patch_id").reset_index(drop=True)
    baseline_counts = baseline["predicted_counts"]
    best_counts = best["predicted_counts"]
    previous_counts = previous["predicted_counts"] if previous is not None else None
    scatter_points = []
    for patch_index, metadata_row in test_sorted.iterrows():
        for class_index, class_name in enumerate(CLASS_NAMES):
            scatter_points.append(
                {
                    "patch_id": int(metadata_row.patch_id),
                    "patch_info": str(metadata_row.patch_info),
                    "source": str(metadata_row.source),
                    "source_group": str(metadata_row.source_group),
                    "class_name": class_name,
                    "gt": int(metadata_row[COUNT_COLUMNS[class_index]]),
                    "baseline_pred": int(baseline_counts[patch_index, class_index]),
                    "best_pred": int(best_counts[patch_index, class_index]),
                    **({"previous_pred": int(previous_counts[patch_index, class_index])} if previous_counts is not None else {}),
                    **({"rotation_pred": int(rotation_counts[patch_index, class_index])} if rotation_counts is not None else {}),
                }
            )
    return {
        "baseline_name": "Initial CellViT baseline",
        "best_name": best_name,
        "support_note": "Support is (test patches; central-crop GT cells used by count R²; full-mask GT cells used by PQ). CoNIC source repository is used as the institution/domain proxy.",
        "by_class": by_class,
        "by_institution": by_institution,
        "by_both": by_both,
        "previous_name": previous_name,
        "intermediate_name": "Rotation-TTA focal raw counts" if rotation_counts is not None else None,
        "confusions": [
            {"name": "Initial baseline", **_detection_aware_confusion(baseline_diagnostic, gt_full_counts)},
            *([{"name": "Previous best", **_detection_aware_confusion(previous_diagnostic, gt_full_counts)}] if previous_diagnostic is not None else []),
            {"name": "Current best", **_detection_aware_confusion(best_diagnostic, gt_full_counts)},
        ],
        "scatter_points": scatter_points,
    }


def build_performance_summary(
    metrics_path: Path | None,
    posthoc_path: Path | None,
    matrix_path: Path | None,
    runs_root: Path | None = None,
    prepared: Path | None = None,
    best_predictions_path: Path | None = None,
    best_counts_path: Path | None = None,
    intermediate_counts_path: Path | None = None,
) -> dict:
    """Build a dashboard-ready ablation sketch from completed experiment artifacts."""
    metrics = _load_json(metrics_path)
    posthoc = _load_json(posthoc_path)
    matrix = _load_json(matrix_path)
    baseline = posthoc.get("baselines", {}).get("raw_test", metrics)
    calibrated = posthoc.get("baselines", {}).get("calibrated_test", {})
    shifted = posthoc.get("baselines", {}).get("label_shifted_test", {})
    margin = posthoc.get("margin_selection", {}).get("R2", {}).get("test_with_unlabeled_label_shift", {})

    def pq_component(source: dict, component: str) -> float | None:
        direct_key = "mDQ+" if component == "dq" else "mSQ+"
        if source.get(direct_key) is not None:
            return float(source[direct_key])
        per_class = source.get("per_class_pq", {})
        values = [per_class.get(name, {}).get(component) for name in CLASS_NAMES]
        if not values or any(value is None for value in values):
            return None
        return float(np.mean(values))

    rows = [
        {
            "id": "E00",
            "stage": "Initial approach",
            "method": "Initial CellViT-SAM-H baseline",
            "kind": "baseline",
            "status": "complete",
            "r2": baseline.get("R2"),
            "mpq": baseline.get("mPQ+"),
            "selection": "Reference run",
            "notes": "Official central-224 count rule; full-patch pooled mPQ+.",
        },
        {
            "id": "E01",
            "stage": "Individual improvement",
            "method": "Validation-only vector calibration",
            "kind": "isolated",
            "status": "complete",
            "r2": calibrated.get("R2"),
            "mpq": calibrated.get("mPQ+"),
            "selection": "Fit on validation labels only",
            "notes": "Clean isolated ablation: same masks and instances; only class probabilities change.",
        },
        {
            "id": "E02a",
            "stage": "Individual improvement",
            "method": "Unlabeled train→test prior correction alone",
            "kind": "isolated",
            "status": "not run yet",
            "r2": None,
            "mpq": None,
            "selection": "EM uses unlabeled target probabilities",
            "notes": "Required isolation run; current measured prior-correction result also includes vector calibration.",
        },
        {
            "id": "E04a",
            "stage": "Individual improvement",
            "method": "Margin dustbin alone",
            "kind": "isolated",
            "status": "not run yet",
            "r2": None,
            "mpq": None,
            "selection": "Threshold selected on validation",
            "notes": "Required isolation run; current measured margin result is stacked on calibration and prior correction.",
        },
        {
            "id": "E02",
            "stage": "Measured combination",
            "method": "Vector calibration + unlabeled prior correction",
            "kind": "failed",
            "status": "failed guard",
            "r2": shifted.get("R2"),
            "mpq": shifted.get("mPQ+"),
            "selection": "Validation calibration; unlabeled EM target prior",
            "notes": "Excluded from combinations: EM inferred implausible class ratios and collapsed both metrics, indicating violated label-shift assumptions.",
        },
        {
            "id": "E04-stack",
            "stage": "Measured combination",
            "method": "Calibration + prior correction + margin dustbin",
            "kind": "failed",
            "status": "failed guard",
            "r2": margin.get("R2"),
            "mpq": margin.get("mPQ+"),
            "selection": "Margin 0.07 selected for validation R²",
            "notes": "Excluded: it inherits the failed prior correction; the margin threshold is also unstable across validation source groups.",
        },
    ]
    for row, source in zip(rows, [baseline, calibrated, {}, {}, shifted, margin]):
        row["dq"] = pq_component(source, "dq")
        row["sq"] = pq_component(source, "sq")

    planned_labels = {
        "E03": ("HV decoder tuning alone", "Tune postprocessing on validation binary PQ before retraining."),
        "E04": ("Learned dustbin class alone", "Seven-class head with unmatched predictions supervised as dustbin."),
        "E05": ("Differentiable patch-count loss alone", "Sweep loss weight and learning rate on validation R²."),
        "E06": ("Soft pooled-PQ surrogate alone", "Sweep loss weight and learning rate on validation mPQ+."),
        "E07": ("SAM-H LoRA segmentation alone", "Tune decoder first; select LoRA learning rate on validation binary PQ."),
        "E08": ("Official CoNIC-trained HoVer-Net control", "Run the challenge authors' public checkpoint through the official output contract; their group-stratified validation reports R² 0.8585 and mPQ+ 0.4998."),
        "E09": ("Flip-TTA map ensemble alone", "Average original/horizontal/vertical flip maps before the locked decoder, including HV sign correction."),
        "E10": ("Color-jitter + blur robustness alone", "Pathology AI ablations favored color jitter and small blur; geometric distortion hurt."),
        "E11": ("Raw-map model ensemble alone", "Average foreground/HV maps before watershed; do not merge already-decoded masks."),
        "E12": ("2× HV decoding alone", "Decode foreground/HV maps at 0.25-mpp-equivalent resolution; select all decoder settings on validation bPQ."),
        "E13": ("2× HV decoding + flip-TTA", "Combine only after the 2× decoder is locked on validation; average aligned raw maps/tokens before decoding/classification."),
        "E14": ("Minority-patch sampling for true LoRA", "Equalize aggregate sampling mass across classes, blended with uniform sampling; keep augmentation and loss unchanged for attribution."),
        "E15": ("Color jitter alone for true LoRA", "Isolate ±10% RGB brightness/contrast/saturation jitter without blur or sampling changes."),
        "E16": ("Light blur alone for true LoRA", "Isolate light Gaussian blur to test whether it drove the combined augmentation's rare-cell recall loss."),
        "E17": ("Complement-balanced focal type head", "Use fixed best-TTA masks/tokens; compare against a matched inverse-frequency CE control and select directly on validation mean(R², mPQ+)."),
        "E18": ("2× HV + flip-TTA + focal type head", "Combine the PQ-high-water 2× TTA masks with the focal head that corrected native-TTA class-count bias."),
        "E19": ("Native-TTA type-probability ensemble", "Blend focal and previous calibrated type-head logits on identical instances; select the blend on validation only."),
        "E20": ("Focal local hyperparameter ablation", "Ablate complement-weight exponent, focal gamma, and label smoothing around E17; select on validation before one test score."),
        "E21": ("Rotation-augmented raw-map TTA", "Add 90° rotations with exact spatial and HV-vector inversion; gate on locked-decoder validation bPQ before full extraction."),
        "E22": ("Rotation-TTA masks + validation-selected count blend", "Blend focal-head and rotation-TTA count outputs independently per class on validation; retain the rotation-TTA masks and therefore their mPQ+."),
        "E23": ("CellViT LoRA with HED stain-space augmentation", "Perturb H/E optical-density channels without blur or geometric distortion; gate standard validation and leave-GLaS-out performance separately."),
        "E24": ("Source-aware sampling", "Allocate replacement-sampling budget across source repositories before testing a source×class mixture."),
        "E25": ("Diverse-fit raw-map ensemble", "Select ensemble membership independently for validation mPQ+ and exact validation R²."),
        "E26": ("Four-direction distance maps", "Add +45° and −45° centroid maps, train only their header plus LoRA, and decode all directional edge evidence before watershed."),
        "E27": ("Four-direction masks + stable base-model token classifier", "Hold four-direction instance geometry fixed, pool untouched base-model flip+rotation-TTA tokens over those masks, and select R²/mPQ+ heads independently."),
        "E28": ("Rotation-TTA + four-direction metric-specific count blend", "Validation-select one count-blend weight per class between the rotation-TTA and four-direction stable-token recipes, while retaining the rotation-TTA masks and mPQ+."),
        "E29": ("Rotation-TTA per-class confidence rejection", "Select six top-1 probability thresholds independently on validation class PQ, then apply them jointly."),
        "E30": ("Source-gated rotation/four-direction mask ensemble", "Validation-select the complete rotation-TTA or four-direction mask/type endpoint per institution using exact additive PQ statistics; no patch-level confidence or test labels."),
        "E31": ("Extended Sobel/edge decoder search", "Continue both boundary sweeps to their legal limits, then repeat kernel and edge passes after the other decoder settings are tuned."),
        "E32": ("HoVer-Net ResNet-50 with six-view spatial TTA", "Train the joint foreground/HV/type architecture from generic ImageNet initialization on our group-disjoint development split, then inverse-align and average identity, flip, and rotation predictions before one decoder pass."),
        "E33": ("HoVer-Net masks/types plus CellViT count blend", "Keep E32 masks/types and validation-select one convex E32/E28 count weight per class; preserve E32 mPQ+."),
        "E34": ("Nested source/density count stacker", "Select global/source-aware linear or quadratic ridge count models inside nested source-group CV; do not evaluate test unless OOF beats E33."),
        "E35": ("Group-disjoint HoVer-Net multi-fit ensemble", "Train independent development-fold fits and average NP/TP probabilities plus exactly inverted HV maps before one decode."),
        "E36": ("HoVer-Net empirical H/E stain-transfer augmentation", "Sample source-balanced H/E concentration targets observed in development training data and transfer each training patch toward one target; compare the complete LR grid against an exact seed-, split-, and step-matched clean control."),
        "E37": ("HoVer-Net source/class sampling", "Isolate source-balanced and minority-class-aware replacement sampling against an exact seed-, split-, LR-, and step-matched uniform control; optimize mPQ+ and R² separately."),
        "E38": ("Deterministic empirical stain test-time augmentation", "Average native and one training-defined H/E target view before decoding; reject unless it improves the metric-specific validation endpoint after matching train-time stain augmentation."),
        "E39": ("Cancelled development-fold HoVer bagging study", "Retained only as a diagnostic record; cross-validation-fold averaging was not the requested heterogeneous-model ensemble."),
        "E40": ("Whole-source HoVer-Net versus CellViT routing", "Choose one complete mask/type endpoint per source on validation pooled mPQ+; reject patch-level routing without broad complementary support."),
        "E41": ("SE-ResNeXt-101 HoVer-Net backbone", "Change only the ImageNet-pretrained encoder to test whether a genuinely different architecture is competitive and complementary before considering a raw-map ensemble."),
        "E42": ("Instance-equalized HoVer-Net foreground loss", "Blend the NP/TP cross-entropy and HV regression/gradient terms from ordinary pixel weighting toward equal total loss mass per ground-truth nucleus; compare against a same-seed zero-blend control."),
        "E43": ("One-loss-per-nucleus pooled type supervision", "Mean-pool type probabilities over each ground-truth nucleus and add one auxiliary cross-entropy per nucleus as a differentiable relaxation of inference-time majority type voting."),
        "E44": ("HoVer-Net type-loss imbalance ablation", "Isolate complement-frequency weighting, focal emphasis, and their pure combination inside segmentation training; give each a full validation learning-rate bracket, then test label smoothing only as a gated marginal add-on so attribution stays identifiable."),
        "E45": ("Lower-intensity class-aware sampling", "Refine the mPQ-specific E37 signal at class-sampling fractions 0.25 and 0.10; give each fraction its own LR bracket and denser validation curve before testing an early-heavy schedule."),
        "E46": ("Metric-specific validation-gated composition", "Build separate recipes: validation-select complementary class-count vectors for R², and for mPQ+ first preserve the strongest geometry while blending only matched-instance type probabilities; test branch-specific raw-map weights only after that fixed-geometry screen passes."),
    }
    experiments = {item.get("id"): item for item in matrix.get("experiments", [])}
    artifact_paths = {
        "E03": runs_root / "hv_tuned" / "metrics_test.json" if runs_root else None,
        "E04": runs_root / "e04_dustbin_weighted" / "metrics_test.json" if runs_root else None,
        "E05": runs_root / "e05_weighted_count_w10" / "metrics_test.json" if runs_root else None,
        "E06": runs_root / "e06_pq_w10" / "metrics_test.json" if runs_root else None,
        "E07": runs_root / "e07_lora_frozen_full" / "metrics_test.json" if runs_root else None,
        "E09": runs_root / "hv_tta" / "metrics_test.json" if runs_root else None,
        "E12": runs_root / "hv_2x" / "metrics_test.json" if runs_root else None,
        "E13": runs_root / "hv_tta_2x" / "metrics_test.json" if runs_root else None,
        "E17": runs_root / "e17_focal" / "metrics_test.json" if runs_root else None,
        "E18": runs_root / "e18_2x_tta_focal" / "metrics_test.json" if runs_root else None,
        "E19": runs_root / "e19_type_ensemble" / "metrics_test.json" if runs_root else None,
        "E20": runs_root / "e20_rho2" / "metrics_test.json" if runs_root else None,
        "E21": runs_root / "e21_rot_tta_focal" / "metrics_test.json" if runs_root else None,
        "E22": runs_root / "e22_count_ensemble" / "metrics_test.json" if runs_root else None,
        "E08": ROOT / "outputs" / "conic_experiments" / "e08_hovernet_official_metrics.json",
        "E26": ROOT / "outputs" / "conic_experiments" / "e26_r2_metrics_test.json",
        "E32": ROOT / "outputs" / "conic_hovernet_our_split" / "locked_phase1_epoch50_flip_rotation_tta_test" / "metrics_test.json",
        "E33": ROOT / "outputs" / "conic_experiments" / "e33_metrics_test.json",
    }
    decoder_report = _load_json(runs_root / "hv_decoder" / "decoder_selection.json" if runs_root else None)
    tuned_posthoc = _load_json(runs_root / "hv_tuned" / "posthoc" / "posthoc_report.json" if runs_root else None)
    tta_posthoc = _load_json(runs_root / "hv_tta" / "posthoc" / "posthoc_report.json" if runs_root else None)
    lora_posthoc = _load_json(runs_root / "e07_lora_frozen_full" / "posthoc" / "posthoc_report.json" if runs_root else None)
    two_x_posthoc = _load_json(runs_root / "hv_2x" / "posthoc" / "posthoc_report.json" if runs_root else None)
    two_x_tta_posthoc = _load_json(runs_root / "hv_tta_2x" / "posthoc" / "posthoc_report.json" if runs_root else None)
    for experiment_id, (label, note) in planned_labels.items():
        # `note` accumulates result narration below; keep the pristine recipe so
        # the dashboard can show what we tried apart from what we found.
        recipe_text = note
        experiment = experiments.get(experiment_id, {})
        artifact = _load_json(artifact_paths.get(experiment_id))
        if experiment_id == "E27":
            test_metrics = experiment.get("result", {}).get("r2_selected", {}).get("test", {})
            artifact = {"R2": test_metrics.get("R2"), "mPQ+": test_metrics.get("mPQ+")}
        if experiment_id == "E28":
            test_metrics = experiment.get("result", {}).get("test", {})
            artifact = {"R2": test_metrics.get("R2"), "mPQ+": test_metrics.get("mPQ+")}
        if experiment_id == "E29":
            test_metrics = experiment.get("result", {}).get("test_selected", {})
            artifact = {"R2": test_metrics.get("R2"), "mPQ+": test_metrics.get("mPQ+")}
        if experiment_id == "E30":
            test_metrics = experiment.get("result", {}).get("test", {})
            artifact = {"R2": test_metrics.get("R2"), "mPQ+": test_metrics.get("mPQ+")}
        if experiment_id == "E37":
            selected = experiment.get("interim_validation_observation", {}).get(
                "provisional_class_0.5_metric_specific_selection", {}
            )
            r2_selected = selected.get("R2_raw_maximum", {})
            mpq_selected = selected.get("mPQ+", {})
            if r2_selected and mpq_selected:
                artifact = {
                    "R2": r2_selected.get("R2"),
                    "mPQ+": mpq_selected.get("mPQ+"),
                    "mDQ+": mpq_selected.get("mDQ+"),
                    "mSQ+": mpq_selected.get("mSQ+"),
                }
        selection = experiment.get("selection_metric", experiment.get("selection_metrics", "Validation-selected"))
        if isinstance(selection, dict):
            selection = " | ".join(f"{name}: {metric}" for name, metric in selection.items())
        if experiment_id == "E03" and decoder_report:
            default_bpq = decoder_report.get("default_20x", {}).get("bPQ")
            tuned_bpq = decoder_report.get("best", {}).get("metrics", {}).get("bPQ")
            note += f" Validation bPQ: intended defaults {default_bpq:.4f} → tuned {tuned_bpq:.4f}."
        if experiment_id == "E09" and decoder_report:
            tta_bpq = decoder_report.get("flip_tta_locked_decoder", {}).get("metrics", {}).get("bPQ")
            note += f" Locked-decoder flip-TTA validation bPQ: {tta_bpq:.4f}."
        if experiment_id == "E07":
            note += " True LoRA kept frozen normalization buffers fixed; deployment-matched validation bPQ 0.5338 → 0.5389 and persisted masks reproduced exactly."
        if experiment_id == "E12":
            note += " Validation bPQ 0.5338 → 0.5458; test shows an mPQ+/R² tradeoff rather than a balanced win."
        if experiment_id == "E13":
            note += " Extending the validation-selected 1e-4 classifier run moved the selected epoch from the sweep boundary (20) to an early-stopped epoch 30."
        if experiment_id == "E14":
            note += " Validation bPQ changed only 0.53889 → 0.53926; class-stratified detection changed neutrophil/plasma/eosinophil matches by -3/+7/+3, so full extraction was not justified."
        if experiment_id == "E10":
            note += " Validation bPQ rose 0.53889 → 0.54300, but matched nuclei fell by 1,650 and neutrophil/eosinophil recall fell 22.2/11.6 points; the aggregate gain came from fewer FPs and higher SQ."
        if experiment_id == "E17":
            note += " Raw test R²/mPQ+ 0.7206/0.3184 beats the prior balanced best on both metrics; validation-only vector calibration was rejected because it degraded validation and test simultaneously."
        if experiment_id == "E21":
            note += " The fresh focal head selected LR 3e-4 at epoch 18 on validation 0.7167/0.3327. Test 0.7123/0.3303 is the single-model PQ high-water and the highest predeclared mean(R²,mPQ+), while E17 retains the R² high-water."
        if experiment_id == "E22":
            note += " Per-class alpha selection raised validation R² to 0.7358 and transferred to test R² 0.7255; unchanged E21 masks retain mPQ+ 0.3303."
        if experiment_id == "E08":
            note += " Reproduced the authors' checkpoint on their exact 1,018-patch fold: R² 0.85843 and mPQ+ 0.49977, within 7.1e-5 and 2.7e-5 of the published values. This validates preprocessing and metrics, but the split is not paired with ours."
        if experiment_id == "E25":
            result = experiment.get("result", {})
            note += f" Correct channel-wise averaging preserved E26's diagonal channels but reached validation bPQ {result.get('channelwise_raw_map_ensemble_validation_bPQ', float('nan')):.5f}, below E26 alone at {result.get('directional_flip_tta_validation_bPQ', float('nan')):.5f}; rejected before test."
        if experiment_id == "E26":
            note += " Weighted four-direction flip-TTA reached validation bPQ 0.56476 and passed leave-GLaS-out (+0.01436, paired CI [0.01193, 0.01662]). The fresh R²-selected head scored test 0.68296/0.31767; improved binary masks did not beat the stable E17/E21 typing recipes."
        if experiment_id == "E27":
            note += " Exact fixed-mask alignment passed for 562,048 instances. Stable base rotation-TTA tokens raise the R²-selected E26-mask recipe to test 0.71804/0.32564, but it remains below E22 R² and E21 mPQ+. Mask-level GLaS undercount and CRAG overcount persist across heads."
        if experiment_id == "E28":
            note += " Validation R² 0.74033 transfers to test R² 0.73200 (+0.00650 over E22). E21 masks retain mPQ+ 0.33027; E27 contributes primarily epithelial, eosinophil, and connective count corrections."
        if experiment_id == "E29":
            note += " Rejects 1.9% of instances. Validation mPQ+ rises 0.33268→0.33409, but test is 0.33020 versus 0.33027 raw and R² falls 0.00731; rejected."
        if experiment_id == "E30":
            note += " Validation routes DPath/PanNuke to E27 and all other sources to E21. Validation mPQ+ 0.34064 transfers to test 0.33309 (+0.00282 over E21). Paired bootstrap supports DPath/PanNuke routing in 100%/99.3% of 2,000 resamples; promoted."
        if experiment_id == "E32":
            note += " Phase 1 epoch 50 reached validation 0.79662/0.43989. Six-transform TTA raised deployment-pipeline validation to 0.83200/0.46950 with positive paired CIs for both metrics; locked test reached 0.81539/0.43727. Phase 2 was not promoted."
        if experiment_id == "E33":
            note += " Validation R² 0.86444; source-group-disjoint OOF R² 0.85120 (+0.01920 over E32 TTA, paired CI [0.00778, 0.03252]); locked test R² 0.84563. E32 masks/types preserve test mPQ+ 0.43727."
        if experiment_id == "E34":
            note += " Nested OOF R² 0.82027 versus E33 0.85120; delta -0.03093, paired CI [-0.04539, -0.01703]. Rejected without test evaluation because extra flexibility overfit fold-specific tails."
        if experiment_id == "E35":
            note += " Six-fold manifest validated, but fold training was stopped after fold 1 because cross-validation-fold averaging was not the requested heterogeneous-model ensemble. No multi-fold result was promoted or evaluated on test."
        if experiment_id == "E36":
            decision = experiment.get("metric_specific_decision", {})
            final = experiment.get("matched_control_final", {})
            r2_decision = decision.get("R2", {})
            mpq_decision = decision.get("mPQ+", {})
            if final and r2_decision and mpq_decision:
                pq_deltas = final.get("delta_per_class_PQ", {})
                note += (
                    f" On the {final['evaluation_set']}, fixed-strength H/E is rejected for both primary "
                    f"metrics. Its independently validation-selected R² is "
                    f"{r2_decision['candidate_grid_maximum']:.4f} versus "
                    f"{r2_decision['control_grid_maximum']:.4f} clean "
                    f"(Δ {r2_decision['independently_selected_delta']:+.4f}). Its mPQ+ is "
                    f"{mpq_decision['candidate_grid_maximum']:.4f} versus "
                    f"{mpq_decision['control_grid_maximum']:.4f} "
                    f"(Δ {mpq_decision['independently_selected_delta']:+.4f}; mDQ+/mSQ+ "
                    f"Δ {mpq_decision['delta_mDQ+']:+.4f}/{mpq_decision['delta_mSQ+']:+.4f}). "
                    f"That tiny boundary gain is entirely supplied by neutrophil PQ "
                    f"({pq_deltas.get('neutrophil', float('nan')):+.4f}); epithelial, lymphocyte, "
                    "plasma, and eosinophil PQ all regress, and eosinophil R² falls by 1.0171. "
                    "Early same-checkpoint gains turned over by epoch 10, so fixed-strength H/E is not "
                    "eligible for the best combination. A milder or epoch-annealed schedule remains a "
                    "low-priority future hypothesis requiring its own clean and fixed-dose controls."
                )
        if experiment_id == "E37" and artifact:
            note += " " + experiment.get("interpretation", "")
        if experiment_id == "E41":
            final = experiment.get("final_validation_selection", {})
            if final:
                note += (
                    f" On the {final['evaluation_set']}, the complete equal-horizon LR grid selected "
                    f"LR {final['selected_learning_rate']:g}, epoch {final['selected_epoch']} for both pilot objectives "
                    f"at R²/mPQ+ {final['R2']:.4f}/{final['mPQ+']:.4f}. Versus the independently selected "
                    f"matched ResNet pilot, deltas are {final['R2_delta_vs_independently_selected_resnet']:+.4f}/"
                    f"{final['mPQ+_delta_vs_independently_selected_resnet']:+.4f}, but mSQ+ changes by "
                    f"{final['mSQ+_delta_vs_independently_selected_resnet']:+.4f}. It advances to count-residual "
                    "and fixed-geometry type-complementarity audits, not standalone mPQ+ promotion."
                )
            else:
                interim = experiment.get("interim_validation_observation", {})
                checkpoint = interim.get("latest_scored_checkpoint", {})
                delta = interim.get("exact_matched_control_delta", {})
                if checkpoint and delta:
                    note += (
                        f" Interim only on the {interim['evaluation_set']}: LR {checkpoint['learning_rate']:g}, "
                        f"epoch {checkpoint['epoch']} reached R²/mPQ+ {checkpoint['R2']:.4f}/{checkpoint['mPQ+']:.4f}; "
                        f"exact matched-checkpoint deltas are {delta['R2']:+.4f}/{delta['mPQ+']:+.4f}. "
                        "The complete LR grid remains required before selection."
                    )
        if experiment_id == "E42":
            interim = experiment.get("interim_validation_observation", {})
            candidate = interim.get("candidate", {})
            delta = interim.get("exact_matched_control_delta", {})
            if candidate and delta:
                final = interim.get("final_selection", {})
                if final:
                    independent = final.get("delta_vs_independently_selected_ordinary_loss", {})
                    note += (
                        f" Complete result on the {interim['evaluation_set']}: the bounded grid selected blend "
                        f"{final['selected_blend']:g}, LR {final['selected_learning_rate']:g}, epoch "
                        f"{final['selected_epoch']} at R²/mPQ+ {final['R2']:.4f}/{final['mPQ+']:.4f}. "
                        f"Versus independently selected ordinary loss, R²/mPQ+/mDQ+/mSQ+ deltas are "
                        f"{independent.get('R2', float('nan')):+.4f}/"
                        f"{independent.get('mPQ+', float('nan')):+.4f}/"
                        f"{independent.get('mDQ+', float('nan')):+.4f}/"
                        f"{independent.get('mSQ+', float('nan')):+.4f}. The intervention is rejected for "
                        "both leaderboard objectives; no capped or mature extension is admitted."
                    )
                else:
                    note += (
                        f" Provisional only on the {interim['evaluation_set']}: blend "
                        f"{candidate['instance_loss_blend']:g}, LR {candidate['learning_rate']:g}, epoch "
                        f"{candidate['epoch']} reached R²/mPQ+ {candidate['R2']:.4f}/{candidate['mPQ+']:.4f}; "
                        f"exact control deltas are {delta['R2']:+.4f}/{delta['mPQ+']:+.4f}, with binary-DQ "
                        f"delta {delta.get('binary_DQ', float('nan')):+.4f}. The complete staged LR/blend grid "
                        "remains required before selection."
                    )
                if interim.get("interpretation"):
                    note += f" Mechanism audit: {interim['interpretation']}"
                latest = interim.get("latest_high_lr_checkpoint", {})
                latest_delta = latest.get("exact_control_delta", {})
                if latest and latest_delta:
                    note += (
                        f" Latest unselected trajectory point: LR {latest['learning_rate']:g}, epoch "
                        f"{latest['epoch']} has R²/mPQ+ {latest['R2']:.4f}/{latest['mPQ+']:.4f} and exact "
                        f"control deltas {latest_delta['R2']:+.4f}/{latest_delta['mPQ+']:+.4f}; this is "
                        "not eligible until the predeclared later checkpoints complete."
                    )
        if experiment_id == "E43":
            note += e43_interim_note(experiment)
        if experiment_id == "E44":
            interim = experiment.get("interim_validation_observation", {})
            combination_family = interim.get("weighting_focal_combination_interim", {})
            final_combination = combination_family.get("upper_lr_epoch10", {})
            candidate = interim.get("candidate", {})
            delta = interim.get("exact_matched_control_delta", {})
            mechanism = interim.get("mechanism", {})
            if candidate and delta and not final_combination:
                note += (
                    f" Provisional only on the {interim['evaluation_set']}: R²/mPQ+ "
                    f"{candidate['R2']:.4f}/{candidate['mPQ+']:.4f}; exact ordinary-control deltas are "
                    f"{delta['R2']:+.4f}/{delta['mPQ+']:+.4f}, with mDQ+/mSQ+ deltas "
                    f"{delta['mDQ+']:+.4f}/{delta['mSQ+']:+.4f}. Plasma supplies most of the class-level "
                    f"PQ gain ({mechanism.get('plasma_PQ_delta', float('nan')):+.4f}), while neutrophil and "
                    "eosinophil PQ remain zero. This is an early conservative-count signal, not yet proof of "
                    "successful rare-class reweighting; epoch 10 and the complete LR bracket remain required."
                )
                endpoint = interim.get("completed_low_lr_mPQ_endpoint", {})
                independent = endpoint.get("delta_vs_independently_selected_ordinary_loss", {})
                if endpoint and independent:
                    note += (
                        f" The completed LR {endpoint['learning_rate']:g} epoch-{endpoint['epoch']} mPQ+ endpoint "
                        f"reaches {endpoint['candidate']['mPQ+']:.4f}; although it rescues the exact low-LR control, "
                        f"it remains {independent['mPQ+']:+.4f} mPQ+, {independent['mDQ+']:+.4f} mDQ+, "
                        f"{independent['mSQ+']:+.4f} mSQ+, and {independent['R2']:+.4f} R² versus independently "
                        "selected ordinary loss. The low-LR rescue is not promotable."
                    )
                interior = interim.get("interior_lr_epoch5", {})
                interior_delta = interior.get("exact_matched_control_delta", {})
                if interior and interior_delta:
                    note += (
                        f" At interior LR {interior['learning_rate']:g}, epoch {interior['epoch']}, the exact-control "
                        f"R²/mPQ+/mDQ+/mSQ+ deltas are {interior_delta['R2']:+.4f}/"
                        f"{interior_delta['mPQ+']:+.4f}/{interior_delta['mDQ+']:+.4f}/"
                        f"{interior_delta['mSQ+']:+.4f}; binary PQ and count tails also improve. Five class PQ "
                        "values move positively, but plasma remains the largest driver and the 12-patch CoNSeP "
                        "subset has an SQ warning. This is promising, not selected, pending epoch 10 and LR 3e-4."
                    )
                completed = interim.get("interior_lr_epoch10", {})
                completed_delta = completed.get("exact_matched_control_delta", {})
                if completed and completed_delta:
                    note += (
                        f" That signal reverses at epoch {completed['epoch']}: exact-control R²/mPQ+/mDQ+/mSQ+ "
                        f"deltas become {completed_delta['R2']:+.4f}/{completed_delta['mPQ+']:+.4f}/"
                        f"{completed_delta['mDQ+']:+.4f}/{completed_delta['mSQ+']:+.4f}, with all six class PQ "
                        "deltas negative and worse spurious/count-tail behavior. Fixed weighting is therefore not "
                        "promotable at LR 1e-4; the LR 3e-4 endpoint only completes the bounded grid."
                    )
                upper = interim.get("upper_lr_epoch5", {})
                upper_delta = upper.get("exact_matched_control_delta", {})
                if upper and upper_delta:
                    note += (
                        f" At upper LR {upper['learning_rate']:g}, epoch {upper['epoch']}, exact-control "
                        f"R²/mPQ+/mDQ+/mSQ+ deltas are {upper_delta['R2']:+.4f}/"
                        f"{upper_delta['mPQ+']:+.4f}/{upper_delta['mDQ+']:+.4f}/"
                        f"{upper_delta['mSQ+']:+.4f}. The pooled typed gains are below the practical threshold, "
                        "mSQ+ fails its tolerance, and the class movement is plasma-driven while neutrophil, "
                        "lymphocyte, and eosinophil PQ decline. This does not confirm the interior-LR transient."
                    )
                final_weighting = interim.get("weighting_only_final_selection", {})
                if final_weighting:
                    selected_r2 = final_weighting["selected_R2"]
                    selected_mpq = final_weighting["selected_mPQ+"]
                    selected_mpq_delta = selected_mpq["delta_vs_independently_selected_ordinary_loss"]
                    note += (
                        f" Complete weighting-only decision on the {final_weighting['evaluation_set']}: LR "
                        f"{selected_mpq['learning_rate']:g}, epoch {selected_mpq['epoch']} is selected for both "
                        f"objectives, but R² is {selected_r2['delta_vs_independently_selected_ordinary_loss']:+.4f} "
                        f"and mPQ+/mDQ+/mSQ+ are {selected_mpq_delta['mPQ+']:+.4f}/"
                        f"{selected_mpq_delta['mDQ+']:+.4f}/{selected_mpq_delta['mSQ+']:+.4f} versus independently "
                        "selected ordinary loss. Weighting-only is rejected for standalone use and fixed-geometry "
                        "type composition; focal-only is the active family."
                    )
                focal = interim.get("focal_only_interim", {})
                focal_delta = focal.get("exact_matched_control_delta", {})
                if focal and focal_delta:
                    note += (
                        f" Focal-only is provisional on the {focal['evaluation_set']}: exact-control "
                        f"R²/mPQ+/mDQ+/mSQ+ deltas are {focal_delta['R2']:+.4f}/"
                        f"{focal_delta['mPQ+']:+.4f}/{focal_delta['mDQ+']:+.4f}/"
                        f"{focal_delta['mSQ+']:+.4f}. Its small gain comes with fewer spurious predictions and "
                        "better count tails, but also fewer geometry and correctly typed matches; it remains below "
                        "the typed-gain threshold pending epoch 10 and the competitive learning rates."
                    )
                    focal_final = focal.get("focal_only_final_selection", {})
                    if focal_final:
                        final_r2 = focal_final["selected_R2"]
                        final_mpq = focal_final["selected_mPQ+"]
                        final_delta = final_mpq["delta_vs_independently_selected_ordinary_loss"]
                        note += (
                            f" Complete focal-only decision on the {focal_final['evaluation_set']}: its selected "
                            f"R² recipe loses {abs(final_r2['delta_vs_independently_selected_ordinary_loss']):.4f} "
                            f"R², while its selected mPQ+ recipe changes mPQ+/mDQ+/mSQ+ by "
                            f"{final_delta['mPQ+']:+.4f}/{final_delta['mDQ+']:+.4f}/{final_delta['mSQ+']:+.4f} "
                            "versus independently selected ordinary loss. Falling focal validation loss alongside "
                            "a collapsing upper-LR count score identifies surrogate misalignment, not conventional "
                            "validation-loss overfit. Focal-only is rejected; the pure weighting+focal interaction "
                            + ("is evaluated separately below." if final_combination else "is still running for attribution.")
                        )
                combination = combination_family.get("low_lr_epoch10", {})
                if combination:
                    candidate = combination["candidate"]
                    independent = combination["delta_vs_independently_selected_ordinary_loss"]
                    note += (
                        f" The completed low-LR weighting+focal endpoint reaches R²/mPQ+ "
                        f"{candidate['R2']:.4f}/{candidate['mPQ+']:.4f}, still "
                        f"{independent['R2']:+.4f}/{independent['mPQ+']:+.4f} versus independently selected "
                        "ordinary loss. Its exact-control mPQ+ rescue is subadditive relative to the two isolated "
                        "low-LR effects, so this is not positive synergy; competitive LRs remain pending."
                    )
                combination_midpoint = combination_family.get("interior_lr_epoch5", {})
                combination_endpoint = combination_family.get("interior_lr_epoch10", {})
                if combination_midpoint and combination_endpoint:
                    midpoint_delta = combination_midpoint["exact_matched_control_delta"]
                    endpoint_delta = combination_endpoint["exact_matched_control_delta"]
                    additivity = combination_endpoint["additivity_check"]
                    note += (
                        f" At LR 1e-4, the combination's epoch-5 exact-control mPQ+ gain of "
                        f"{midpoint_delta['mPQ+']:+.4f} reverses by epoch 10 to R²/mPQ+/mDQ+/mSQ+ "
                        f"deltas {endpoint_delta['R2']:+.4f}/{endpoint_delta['mPQ+']:+.4f}/"
                        f"{endpoint_delta['mDQ+']:+.4f}/{endpoint_delta['mSQ+']:+.4f}. All six class PQs "
                        "decline, matched type accuracy falls, and both large under- and over-count tails rise. "
                        f"The combined mPQ+ loss ({additivity['combined_exact_delta_mPQ+']:+.4f}) is much larger "
                        f"than the sum of isolated endpoint losses ({additivity['sum_of_isolated_exact_delta_mPQ+']:+.4f}), "
                        "so the interaction is antagonistic at LR 1e-4 and cannot justify composition at that LR."
                    )
            if final_combination:
                final_candidate = final_combination["candidate"]
                final_delta = final_combination["exact_and_independently_selected_control_delta"]
                class_delta = final_combination["mechanism"]["class_PQ_deltas"]
                note += (
                    f" On the {final_combination['evaluation_set']}, the completed LR 3e-4 endpoint reaches "
                    f"R²/mPQ+/mDQ+/mSQ+ {final_candidate['R2']:.4f}/{final_candidate['mPQ+']:.4f}/"
                    f"{final_candidate['mDQ+']:.4f}/{final_candidate['mSQ+']:.4f}, with matched ordinary-loss "
                    f"deltas {final_delta['R2']:+.4f}/{final_delta['mPQ+']:+.4f}/"
                    f"{final_delta['mDQ+']:+.4f}/{final_delta['mSQ+']:+.4f}. Neutrophil, lymphocyte, and "
                    f"eosinophil PQ improve ({class_delta['neutrophil']:+.4f}/"
                    f"{class_delta['lymphocyte']:+.4f}/{class_delta['eosinophil']:+.4f}), while epithelial "
                    f"and connective soften ({class_delta['epithelial']:+.4f}/"
                    f"{class_delta['connective']:+.4f}). This is a stable typed-segmentation signal, not an "
                    "R² recipe. The 6e-4 candidate turns down to R²/mPQ+/mDQ+/mSQ+ "
                    "0.5431/0.3758/0.4618/0.7974, establishing an upper LR bracket. Ordinary loss at 6e-4 "
                    "reaches mPQ+/mDQ+/mSQ+ 0.3905/0.4798/0.8000, so independently selected combined-loss "
                    "deltas are -0.0038/-0.0070/+0.0054. The earlier exact-3e-4 gain was an LR-response shift, "
                    "not a superior recipe; E44 is rejected for both leaderboard objectives. The predeclared "
                    "1e-3 pair was adaptively skipped after both candidate metrics turned down."
                )
                smoothing = combination_family.get("conditional_label_smoothing_endpoint", {})
                if smoothing:
                    smooth_delta = smoothing["delta_vs_unsmoothed_combination"]
                    note += (
                        f" Conditional label smoothing is rejected: versus the unsmoothed endpoint it changes "
                        f"R²/mPQ+/mDQ+/mSQ+ by {smooth_delta['R2']:+.4f}/{smooth_delta['mPQ+']:+.4f}/"
                        f"{smooth_delta['mDQ+']:+.4f}/{smooth_delta['mSQ+']:+.4f}."
                    )
        if experiment_id == "E45":
            note += (
                " The refined design includes a full seed-206 uniform-with-replacement LR control, because every "
                "nonzero weighted sampler also changes ordinary one-pass training into replacement sampling. On "
                "the exact development-training fold, class fractions 0.10/0.25/0.50 retain expected unique-patch "
                "fractions 0.630/0.622/0.597 per epoch. An mPQ gain must beat both ordinary no-replacement and "
                "uniform-replacement controls before it can be attributed to class exposure, and CRAG, DPath, "
                "and GLaS mPQ+/mDQ+/mSQ+ must remain within 0.01 of both independently selected controls."
            )
        if experiment_id == "E46":
            pilot = experiment.get("r2_recipe", {}).get("robust_count_correction_pilot", {})
            if pilot:
                full = pilot["full_validation"]
                nested = pilot["nested_group_cv"]
                dpath = pilot["dpath_zero_epithelial"]
                note += (
                    f" Robust count-correction pilot on the {pilot['evaluation_set']}: cap auxiliary additions "
                    "and removals separately in sqrt(mask-count+1) units after selecting the ordinary classwise "
                    f"blend. Full-validation R² changes {full['E33_R2']:.5f}→{full['robust_R2']:.5f}; nested "
                    f"group-CV changes {nested['ordinary_E33_R2']:.5f}→{nested['robust_R2']:.5f}, with paired "
                    f"95% CI [{nested['paired_group_bootstrap_95_CI'][0]:.5f}, "
                    f"{nested['paired_group_bootstrap_95_CI'][1]:.5f}]. On {dpath['support']} DPath patches "
                    f"with zero epithelial truth, >10/>20 false-count prevalence falls from "
                    f"{dpath['E33_over_10_fraction']:.1%}/{dpath['E33_over_20_fraction']:.1%} to "
                    f"{dpath['robust_over_10_fraction']:.1%}/{dpath['robust_over_20_fraction']:.1%}. "
                    "This remains a held reliability Pareto candidate because the full gain is below +0.003, "
                    "the bootstrap interval crosses zero, and OOF general tails rise slightly."
                )
        status = "complete" if artifact else experiment.get("status", "planned").replace("waiting_for_cuda", "ready to run")
        if experiment_id in {"E08", "E25", "E26", "E27", "E28", "E29", "E30", "E32", "E33", "E34", "E35", "E37"}:
            status = experiment.get("status", status)
        if experiment_id == "E08":
            selection = "Authors' exact public group/source fold; external reproducibility control"
        rows.append(
            {
                "id": experiment_id,
                "stage": "Measured combination" if experiment_id in {"E13", "E22", "E28", "E30", "E33", "E35"} else ("Individual improvement" if experiment_id != "E08" else "External control"),
                "method": label,
                "kind": "combination" if experiment_id in {"E13", "E22", "E28", "E30", "E33", "E35"} else ("isolated" if experiment_id != "E08" else "benchmark"),
                "status": status,
                "r2": artifact.get("R2"),
                "mpq": artifact.get("mPQ+"),
                "dq": pq_component(artifact, "dq"),
                "sq": pq_component(artifact, "sq"),
                "selection": selection,
                "notes": note,
                "recipe": recipe_text,
                "findings": note[len(recipe_text):].strip(),
                "delta_comparable": experiment_id != "E37",
                "leaderboard_ineligible": experiment_id == "E37",
            }
        )
        if experiment_id == "E26":
            pq_artifact = _load_json(ROOT / "outputs" / "conic_experiments" / "e26_mpq_metrics_test.json")
            rows.append(
                {
                    "id": "E26-PQ",
                    "stage": "Metric-specific recipe",
                    "method": "Four-direction distance-map masks + PQ-surrogate-selected type head",
                    "kind": "failed",
                    "status": "complete — validation gain did not transfer",
                    "r2": pq_artifact.get("R2"),
                    "mpq": pq_artifact.get("mPQ+"),
                    "dq": pq_component(pq_artifact, "dq"),
                    "sq": pq_component(pq_artifact, "sq"),
                    "selection": "LR/epoch selected independently on validation mPQ+",
                    "notes": "Validation mPQ+ increased from 0.32259 to 0.32286, mainly through neutrophil and lymphocyte gains, but test fell to 0.31244 (below both the matched E26 control and E21).",
                }
            )
        if experiment_id == "E27":
            pq_test = experiment.get("result", {}).get("mpq_selected", {}).get("test", {})
            rows.append(
                {
                    "id": "E27-PQ",
                    "stage": "Metric-specific recipe",
                    "method": "Stable-token classifier selected for mPQ+",
                    "kind": "failed",
                    "status": "complete — validation gain did not transfer",
                    "r2": pq_test.get("R2"),
                    "mpq": pq_test.get("mPQ+"),
                    "selection": "Full 20-epoch horizon per LR; selected 1e-3 epoch 13 by validation mPQ+",
                    "notes": "Validation mPQ+ 0.33471 exceeded E21, but locked test mPQ+ was 0.32930 versus E21 0.33027. Early patience=5 was explicitly rejected because it truncated the winning LR before its late gains.",
                }
            )
    tuned_calibrated = tuned_posthoc.get("baselines", {}).get("calibrated_test", {})
    if tuned_calibrated:
        rows.append(
            {
                "id": "E03-cal",
                "stage": "Measured combination",
                "method": "Tuned HV decoder + validation vector calibration",
                "kind": "combination",
                "status": "complete",
                "r2": tuned_calibrated.get("R2"),
                "mpq": tuned_calibrated.get("mPQ+"),
                "dq": pq_component(tuned_calibrated, "dq"),
                "sq": pq_component(tuned_calibrated, "sq"),
                "selection": "Decoder and probability calibration selected on validation only",
                "notes": "Current strongest valid combination; no target labels or target counts used for fitting.",
            }
        )
    tta_calibrated = tta_posthoc.get("baselines", {}).get("calibrated_test", {})
    if tta_calibrated:
        rows.append(
            {
                "id": "E09-cal",
                "stage": "Measured combination",
                "method": "Tuned HV + flip-TTA + validation vector calibration",
                "kind": "combination",
                "status": "complete",
                "r2": tta_calibrated.get("R2"),
                "mpq": tta_calibrated.get("mPQ+"),
                "dq": pq_component(tta_calibrated, "dq"),
                "sq": pq_component(tta_calibrated, "sq"),
                "selection": "Locked decoder; TTA maps/tokens averaged before validation-selected head and calibration",
                "notes": "Current strongest valid combination; validation and test calibrated metrics transfer closely.",
            }
        )
    lora_calibrated = lora_posthoc.get("baselines", {}).get("calibrated_test", {})
    if lora_calibrated:
        rows.append(
            {
                "id": "E07-cal",
                "stage": "Measured combination",
                "method": "True LoRA segmentation + validation vector calibration",
                "kind": "combination",
                "status": "complete",
                "r2": lora_calibrated.get("R2"),
                "mpq": lora_calibrated.get("mPQ+"),
                "dq": pq_component(lora_calibrated, "dq"),
                "sq": pq_component(lora_calibrated, "sq"),
                "selection": "LoRA LR/epoch by validation bPQ; classifier LR by validation macro-F1; calibration on validation only",
                "notes": "Improves over tuned-HV + calibration, especially neutrophil count R², but rare-class detection quality remains the mPQ+ bottleneck and it trails flip-TTA.",
            }
        )
    two_x_calibrated = two_x_posthoc.get("baselines", {}).get("calibrated_test", {})
    if two_x_calibrated:
        rows.append(
            {
                "id": "E12-cal",
                "stage": "Measured combination",
                "method": "2× HV decoding + validation vector calibration",
                "kind": "combination",
                "status": "complete",
                "r2": two_x_calibrated.get("R2"),
                "mpq": two_x_calibrated.get("mPQ+"),
                "dq": pq_component(two_x_calibrated, "dq"),
                "sq": pq_component(two_x_calibrated, "sq"),
                "selection": "2× decoder by validation bPQ; classifier LR and vector calibration on validation",
                "notes": "Pareto tradeoff: improves mPQ+ over native tuned decoding but loses R², driven by plasma overprediction and weak neutrophil DQ.",
            }
        )
    two_x_tta_calibrated = two_x_tta_posthoc.get("baselines", {}).get("calibrated_test", {})
    if two_x_tta_calibrated:
        rows.append(
            {
                "id": "E13-cal",
                "stage": "Measured combination",
                "method": "2× HV + flip-TTA + validation vector calibration",
                "kind": "combination",
                "status": "complete",
                "r2": two_x_tta_calibrated.get("R2"),
                "mpq": two_x_tta_calibrated.get("mPQ+"),
                "dq": pq_component(two_x_tta_calibrated, "dq"),
                "sq": pq_component(two_x_tta_calibrated, "sq"),
                "selection": "Locked 2× decoder; classifier LR/epoch and vector calibration selected on validation",
                "notes": "PQ high-water mark, but not the strongest balanced method: rare-class DQ and class-count bias transfer keep R² below native-resolution flip-TTA.",
            }
        )
    rows.append(
        {
            "id": "E47",
            "stage": "Final-stack component",
            "method": "Mature class-weighted replacement HoVer-Net",
            "kind": "isolated",
            "status": "training",
            "r2": None,
            "mpq": None,
            "dq": None,
            "sq": None,
            "selection": "LR fixed from the matched pilot; checkpoint selected by development-validation mPQ+ only",
            "notes": (
                "Fifty-epoch HoVer-Net fit with 50% of replacement-sampling mass allocated to equal-class "
                "exposure. It uses generic ImageNet initialization, the same 3,613/711 development split as "
                "E32, LR 1e-4 with a tenfold drop after epoch 25, and no locked-test inference during training."
            ),
            "delta_comparable": False,
            "leaderboard_ineligible": True,
            "evaluation_set": "Source-group-disjoint development validation · 711 patches · training",
        }
    )
    final_curve_path = (
        ROOT
        / "outputs"
        / "conic_final_stack"
        / "e47_class_weighted_replacement_seed206_lr1e-4"
        / "training_curve.json"
    )
    final_curve = _load_json(final_curve_path)
    final_summary = _load_json(final_curve_path.with_name("summary.json"))
    scored_final_rows = (
        [
            item
            for item in final_curve
            if item.get("val_mPQ+") is not None and item.get("val_R2") is not None
        ]
        if isinstance(final_curve, list)
        else []
    )
    if scored_final_rows:
        selected_final = max(scored_final_rows, key=lambda item: float(item["val_mPQ+"]))
        latest_final_epoch = max(int(item["epoch"]) for item in final_curve)
        rows[-1].update(
            {
                "status": (
                    f"complete · selected epoch {int(selected_final['epoch'])}"
                    if final_summary
                    else (
                        f"training · epoch {latest_final_epoch}/50 · "
                        f"best scored epoch {int(selected_final['epoch'])}"
                    )
                ),
                "r2": selected_final.get("val_R2"),
                "mpq": selected_final.get("val_mPQ+"),
                "dq": selected_final.get("val_mDQ+"),
                "sq": selected_final.get("val_mSQ+"),
                "evaluation_set": (
                    "Source-group-disjoint development validation · 711 patches · checkpoint selected"
                    if final_summary
                    else "Source-group-disjoint development validation · 711 patches · training"
                ),
            }
        )
    final_test_metrics = _load_json(
        ROOT / "outputs" / "conic_final_stack" / "locked_weighted_tta_test" / "metrics_test.json"
    )
    complex_test_metrics = _load_json(
        ROOT / "outputs" / "conic_final_stack" / "locked_final_mpq_test" / "metrics_test.json"
    )
    final_is_tested = bool(final_test_metrics)
    rows.append(
        {
            "id": "E48",
            "stage": "Recommended final model",
            "method": "Rare-class-trained HoVer-Net with six-view TTA",
            "kind": "best-combination" if final_is_tested else "future-best",
            "status": "complete · promoted final model" if final_is_tested else "not run yet",
            "r2": final_test_metrics.get("R2"),
            "mpq": final_test_metrics.get("mPQ+"),
            "dq": final_test_metrics.get("mDQ+"),
            "sq": final_test_metrics.get("mSQ+"),
            "selection": "Learning rate and checkpoint selected on development validation; test scored once",
            "notes": "Replace CellViT++ with HoVer-Net, show it more training patches containing rare cell types, average six spatial views at inference, and use standard HoVer-Net cell separation at its intended resolution. This one-checkpoint recipe clears both targets without a second model or a separate count branch.",
            "evaluation_set": (
                "Retrospective internal test · 657 patches · single frozen-recipe run"
                if final_is_tested
                else "Not yet evaluated"
            ),
            "leaderboard_ineligible": not final_is_tested,
            "delta_comparable": final_is_tested,
        }
    )
    debias_test_metrics = _load_json(
        ROOT / "outputs" / "conic_final_stack" / "e50_debias_metrics_test.json"
    )
    if debias_test_metrics and final_is_tested:
        rows.append(
            {
                "id": "E50",
                "stage": "Complexity ablation",
                "method": "Validation-fitted count rescaling",
                "kind": "combination",
                "status": "complete · not promoted: final model already clears both targets",
                "r2": debias_test_metrics.get("R2"),
                # Counts do not touch masks, so the mPQ+ family is exactly E48's.
                "mpq": final_test_metrics.get("mPQ+"),
                "dq": final_test_metrics.get("mDQ+"),
                "sq": final_test_metrics.get("mSQ+"),
                "selection": "Six per-class count scales fit on development validation, frozen, then test scored once",
                "notes": "Keep E48's masks and types, then multiply each class count by a factor fit on validation. This raises locked-test R² without changing mPQ+, but it adds a calibrated post-processing branch after the one-checkpoint model already clears both targets. We retain it as an ablation rather than change the agreed final model narrative.",
                "evaluation_set": "Retrospective internal test · 657 patches · complexity audit",
                "leaderboard_ineligible": True,
                "delta_comparable": True,
            }
        )
    if complex_test_metrics:
        rows.append(
            {
                "id": "E49",
                "stage": "Complexity ablation",
                "method": "Two-HoVer-Net cell-type probability blend",
                "kind": "combination",
                "status": "complete · not promoted: marginal gain for double inference complexity",
                "r2": complex_test_metrics.get("R2"),
                "mpq": complex_test_metrics.get("mPQ+"),
                "dq": complex_test_metrics.get("mDQ+"),
                "sq": complex_test_metrics.get("mSQ+"),
                "selection": "75/25 type-probability blend selected on development validation",
                "notes": "Keep E48's cell masks, then use a second normally sampled HoVer-Net for 75% of each matched cell's type vote. Validation mPQ+ rose 0.4892→0.4914, but locked-test mPQ+ rose only 0.46140→0.46175 (+0.00035). We retain this as evidence, not as the recommended model: it doubles inference work for a negligible gain.",
                "evaluation_set": "Retrospective internal test · 657 patches · complexity audit",
                "leaderboard_ineligible": True,
                "delta_comparable": True,
            }
        )
    curated_visuals = {
        "E23": [
            {"path": "curves/e23_hed_augmentation_examples.png", "label": "HED augmentation examples"},
        ],
        "E32": [
            {"path": "curves/e32_six_way_tta_explainer.png", "label": "Six-view TTA explainer"},
            {"path": "curves/e32_hovernet_phase1_diagnostics.png", "label": "Training and class diagnostics"},
            {"path": "curves/e32_tta_val_detection_typing.png", "label": "Detection versus typing"},
        ],
        "E33": [
            {"path": "curves/e33_hovernet_tta_e28_count_blend.png", "label": "Validation-selected count blend"},
            {"path": "curves/e33_dpath_zero_epithelial_audit.png", "label": "DPath true-zero epithelial outlier-prevalence audit"},
            {"path": "panels/03529.png", "label": "Representative DPath connective-to-epithelial typing failure (retrospective test)"},
        ],
        "E36": [
            {"path": "curves/e36_hed_concentration_distribution.png", "label": "Observed H/E target distribution"},
            {"path": "curves/e36_hovernet_hed_examples.png", "label": "Empirical H/E target transfer"},
            {"path": "curves/e36_hed_vs_matched_control.png", "label": "Exact seed-matched causal audit"},
        ],
        "E37": [
            {"path": "curves/e37_sampling_budget.png", "label": "Sampling exposure and duplicate budget"},
            {"path": "curves/e37_class_sampling_lr_live.png", "label": "Class-sampling LR, DQ, and SQ trajectories"},
            {"path": "curves/e37_class_sampling_vs_control.png", "label": "Exact seed-matched class-sampling causal audit"},
        ],
        "E41": [
            {"path": "curves/e41_seresnext101_lr_live.png", "label": "Heterogeneous-backbone LR and class trajectories"},
            {"path": "curves/e41_seresnext101_vs_resnet50.png", "label": "Matched SE-ResNeXt-101 versus ResNet-50 causal audit"},
        ],
        "E42": [
            {"path": "curves/e42_instance_equalized_stage_a_live.png", "label": "Instance-equalized loss stage-A learning-rate trajectory"},
            {"path": "curves/e42_instance_loss_dose_progress.png", "label": "Selected-LR instance-loss dose response and directional tails"},
            {"path": "curves/e42_instance_loss_weight_exposure.png", "label": "Training-only instance-size and loss-weight exposure"},
        ],
        "E43": [
            {"path": "curves/e43_instance_type_lr_live.png", "label": "Per-nucleus type-loss LR trajectories and exact controls"},
            {"path": "curves/e43_instance_type_dose_live.png", "label": "Validation-selected LR auxiliary-weight trajectories"},
        ],
        "E44": [
            {"path": "curves/e44_lr_expansion_candidate_vs_control.png", "label": "Complete candidate/control LR bracket showing ordinary-loss mPQ+ selection wins"},
        ],
        "E45": [
            {"path": "curves/e37_sampling_budget.png", "label": "Class exposure, replacement control, and unique-patch budget"},
        ],
        "E47": [
            {"path": "curves/e47_final_stack_training_live.png", "label": "Live final-stack training and validation trajectory"},
        ],
        "E48": [
            {"path": "curves/e47_final_stack_training_live.png", "label": "Final model training and validation trajectory"},
            {"path": "curves/e32_six_way_tta_explainer.png", "label": "Six-view TTA explainer"},
        ],
    }
    test_scope = "Retrospective internal test · 657 patches"
    for row in rows:
        row["visuals"] = curated_visuals.get(row.get("id"), [])
        row["explanation"] = row.get("notes", "")
        experiment_record = experiments.get(row.get("id"), {})
        if row.get("evaluation_set"):
            continue
        if row.get("kind") == "benchmark":
            row["evaluation_set"] = "External authors' public fold · not paired with our split"
        elif row.get("leaderboard_ineligible") and experiment_record.get("interim_validation_observation"):
            row["evaluation_set"] = experiment_record["interim_validation_observation"].get(
                "evaluation_set", "Development validation only · interim"
            )
        elif row.get("r2") is not None or row.get("mpq") is not None:
            row["evaluation_set"] = test_scope
        elif experiment_record.get("interim_validation_observation"):
            row["evaluation_set"] = experiment_record["interim_validation_observation"].get(
                "evaluation_set", "Development validation only · interim"
            )
        elif experiment_record.get("pilot_result", {}).get("evaluation_set"):
            row["evaluation_set"] = experiment_record["pilot_result"]["evaluation_set"] + " only"
        elif any(key in experiment_record for key in ("validation", "result", "pilot_result", "base_model_gate")):
            row["evaluation_set"] = "Development validation / OOF only"
        else:
            row["evaluation_set"] = "Not yet evaluated"
    completed = [
        row for row in rows
        if row["r2"] is not None
        and row["mpq"] is not None
        and row.get("kind") != "benchmark"
        and not row.get("leaderboard_ineligible")
    ]
    best = max(completed, key=lambda row: 0.5 * (row["r2"] + row["mpq"]), default={})
    best_r2 = max(completed, key=lambda row: row["r2"], default={})
    best_mpq = max(completed, key=lambda row: row["mpq"], default={})
    published = metrics.get("published_benchmarks", {}).copy()
    references = list(published.get("references", []))
    references.append(
        {
            "name": "Pathology AI AugHoVer-Net (public Lizard Fold 0)",
            "role": "source-held-out public cross-validation context; mean over repeated runs",
            "split": "public Lizard source fold 0",
            "mpq_plus": 0.5599,
            "r2": 0.8437,
            "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11621583/",
        }
    )
    published["references"] = references
    frequency_shift = []
    if prepared is not None:
        metadata = load_metadata(prepared)
        count_columns = [column for column in metadata.columns if column.startswith("count_")]
        totals = metadata.groupby("split")[count_columns].sum()
        proportions = totals.div(totals.sum(axis=1), axis=0)
        shift_report = tta_posthoc or tuned_posthoc
        em_ratios = shift_report.get("label_shift", {}).get("target_over_source_ratio", {})
        for column in count_columns:
            class_name = column.removeprefix("count_")
            train_share = float(proportions.loc["train", column])
            test_share = float(proportions.loc["test", column])
            frequency_shift.append(
                {
                    "class": class_name,
                    "train_share": train_share,
                    "test_share": test_share,
                    "actual_ratio": test_share / train_share,
                    "em_ratio": em_ratios.get(class_name),
                }
            )
    native_extended = _load_json(
        ROOT / "outputs" / "conic_hovernet_our_split" / "phase1_epoch50_native_val" / "metrics_val_extended.json"
    )
    tta_extended = _load_json(
        ROOT / "outputs" / "conic_hovernet_our_split" / "phase1_epoch50_flip_rotation_tta_val" / "metrics_val_extended.json"
    )

    def segmentation_row(source: dict) -> dict:
        diagnostic = source.get("segmentation_diagnostics", {})
        return {
            "r2": source.get("R2"),
            "mpq": source.get("mPQ+"),
            "dq": pq_component(source, "dq"),
            "sq": pq_component(source, "sq"),
            "foreground_jaccard": diagnostic.get("foreground_jaccard"),
            "bpq": diagnostic.get("bPQ"),
            "binary_dq": diagnostic.get("binary_DQ"),
            "binary_sq": diagnostic.get("binary_SQ"),
            "aji_plus": diagnostic.get("AJI+"),
            "boundary_f1": diagnostic.get("boundary_F1"),
        }

    validation_segmentation = {}
    if native_extended and tta_extended:
        validation_segmentation = {
            "split": "group-disjoint validation only",
            "native": segmentation_row(native_extended),
            "tta": segmentation_row(tta_extended),
        }
    targets = {"r2": 0.76, "mpq": 0.457}
    rows = normalize_rows(rows)
    by_id = {row.get("id"): row for row in rows}
    recommended = by_id.get("E48")
    best = recommended or by_id.get(best.get("id"), best)
    best_r2 = recommended or by_id.get(best_r2.get("id"), best_r2)
    best_mpq = recommended or by_id.get(best_mpq.get("id"), best_mpq)
    summary = {
        "baseline": rows[0],
        "best": best,
        "best_r2": best_r2,
        "best_mpq": best_mpq,
        "targets": targets,
        "rows": rows,
        "outcome_tally": outcome_tally(rows),
        "trajectory": build_trajectory(rows, targets),
        "published": published,
        "selection_policy": matrix.get("selection_policy", {}),
        "best_rule": "Recommend the simplest single model that clears both targets. More complicated variants remain visible as ablations, but are not promoted for tiny validation-selected gains that do not transfer materially to the locked test.",
        "winner_lessons": [
            {"priority": "Run now", "idea": "Flip-TTA before decoding", "evidence": "Original + horizontal + vertical predictions are aligned, HV signs corrected, then raw maps averaged before watershed."},
            {"priority": "Adopt", "idea": "Source-held-out validation", "evidence": "Their Lizard folds follow source repositories; this directly targets institution/domain shift."},
            {"priority": "Ablate", "idea": "Color jitter and small blur", "evidence": "Both improved public Lizard R²; geometric distortion reduced both metrics."},
            {"priority": "Ablate", "idea": "Stronger HV gradient supervision", "evidence": "The winner emphasizes MSE plus gradient error on distance maps; our LoRA loss already exposes both weights."},
            {"priority": "Later", "idea": "Raw-map model ensemble", "evidence": "They average diverse backbone outputs before one postprocessor, reducing count variance and mask noise."},
            {"priority": "Do not assume", "idea": "Rare-class weighting", "evidence": "Their later ablation found focal and weighted CE below the unweighted result, despite older challenge code using custom loss logic."}
        ],
        "idea_references": [
            {
                "idea": "CoNIC task definitions and pooled mPQ+ / macro R² evaluation",
                "origin": "CoNIC challenge organizers",
                "source": "https://conic-challenge.grand-challenge.org/Evaluation/",
                "adaptation": "We reproduce the official central-224 count contract and pooled per-class PQ statistics; DQ and SQ are exposed separately for diagnosis.",
                "experiments": "All experiments",
            },
            {
                "idea": "Joint nuclear-pixel, horizontal/vertical distance, and type branches",
                "origin": "Graham et al., HoVer-Net",
                "source": "https://doi.org/10.1016/j.media.2019.101563",
                "adaptation": "E32 uses the architecture and six-term objective with generic ImageNet initialization on our leakage-free split.",
                "experiments": "E32, E36, E37, E41",
            },
            {
                "idea": "Foundation-model cell masks with lightweight taxonomy adaptation",
                "origin": "Hörst et al., CellViT++",
                "source": "https://arxiv.org/abs/2501.05269",
                "adaptation": "Initial baseline and the complementary count branch retained in E33; CoNIC-overlapping checkpoints are excluded from candidate initialization.",
                "experiments": "E00–E30, E33",
            },
            {
                "idea": "Spatial TTA and raw-map ensemble before one decoder",
                "origin": "Medical-segmentation TTA literature; Pathology AI CoNIC solution",
                "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6783308/",
                "secondary_source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11621583/",
                "adaptation": "E32 uses six exact symmetries, inverse-warped NP/TP probabilities, and HV axis/sign correction; selection is validation-only.",
                "experiments": "E09, E21, E32",
            },
            {
                "idea": "SE-ResNeXt backbone diversity, decoder dropout, and model averaging",
                "origin": "Pathology AI CoNIC winning solution",
                "source": "https://github.com/WinnieLaugh/CONIC_Pathology_AI",
                "secondary_source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11621583/",
                "adaptation": "E41 isolates generic-ImageNet SE-ResNeXt-101 first; no released CoNIC-trained weights are loaded, and ensembling is gated on standalone complementarity.",
                "experiments": "E41",
            },
            {
                "idea": "Sample realistic stain targets from an observed distribution",
                "origin": "RandStainNA; classical stain-normalization literature",
                "source": "https://arxiv.org/abs/2206.12694",
                "secondary_source": "https://cseweb.ucsd.edu/~mniethammer/publication/macenko-nmbwgst-09/",
                "adaptation": "E36 samples fold-local, source-balanced joint H/E concentration pairs with bounded jitter instead of independent or unconstrained colors.",
                "experiments": "E23, E36, E38",
            },
            {
                "idea": "Low-rank parameter-efficient fine-tuning",
                "origin": "Hu et al., LoRA",
                "source": "https://arxiv.org/abs/2106.09685",
                "adaptation": "CellViT segmentation adapters were tested with frozen normalization state and matched extraction; null or tradeoff results were retained rather than promoted.",
                "experiments": "E07, E10, E14–E16, E23–E26",
            },
            {
                "idea": "Source-balanced sampling and equal-nucleus loss alignment",
                "origin": "Project hypothesis motivated by CoNIC source shift and equal-instance leaderboard scoring",
                "source": "https://pages.up.pt/~up367235/publications/journals/2024GRAHAM2024103047.pdf",
                "adaptation": "E37 tests source and class exposure with matched optimizer steps; E42 equalizes foreground gradient mass across nuclei, and E43 adds one pooled type objective per nucleus to match inference-time aggregation.",
                "experiments": "E24, E37, E42, E43",
            },
            {
                "idea": "Minority-patch sampling and class-weighted focal supervision",
                "origin": "EPFL StarDist and MDC Berlin / IFP Bern CoNIC teams",
                "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11621583/",
                "adaptation": "E37 isolates class-aware replacement sampling in HoVer-Net. E44 separately brackets complement-frequency weighting, focal emphasis, and their combination; sampling plus loss is forbidden until both isolated effects pass.",
                "experiments": "E14, E17, E20, E37, E44",
            },
        ],
        "frequency_shift": frequency_shift,
        "validation_segmentation": validation_segmentation,
        "subgroups": build_subgroup_breakdown(
            prepared,
            runs_root,
            posthoc_path,
            best_predictions_path=best_predictions_path,
            best_counts_path=best_counts_path,
            intermediate_counts_path=intermediate_counts_path,
        ),
        "curves": [
            {
                "name": "HV decoder validation sweep",
                "path": "curves/hv_decoder_sweep.png",
                "caption": "Thirty-four decoder candidates selected on 711 validation patches; flip-TTA is evaluated afterward with the selected decoder locked.",
            },
            {
                "name": "HV-instance classifier learning-rate sweep",
                "path": "curves/hv_classifier_lr_sweep.png",
                "caption": "Selected learning rate 3e-4 at epoch 4 by validation macro-F1 (0.5488); all tested learning rates and early-stopping trajectories are shown.",
            },
            {
                "name": "Tuned-HV classifier learning-rate sweep",
                "path": "curves/hv_tuned_classifier_lr_sweep.png",
                "caption": "After validation-only HV decoder tuning, 1e-4 at epoch 14 was selected by validation macro-F1 (0.5449); 3e-4 and 1e-3 were both worse.",
            },
            {
                "name": "Flip-TTA classifier learning-rate sweep",
                "path": "curves/hv_tta_classifier_lr_sweep.png",
                "caption": "On TTA-averaged token features, 1e-4 at epoch 14 was selected by validation macro-F1 (0.5827), clearly above the non-TTA head.",
            },
            {
                "name": "Learned-dustbin learning-rate sweep",
                "path": "curves/e04_dustbin_lr_sweep.png",
                "caption": "Unweighted diagnostic variant: selected 3e-4 at epoch 8; retained for transparency but not treated as the clean E04 ablation.",
            },
            {
                "name": "Clean weighted learned-dustbin sweep",
                "path": "curves/e04_dustbin_weighted.png",
                "caption": "Inverse-frequency weighting matched to the initial head; selected 3e-4 at epoch 18, but failed the dual-metric guard on test.",
            },
            {
                "name": "Count-loss matched weighted-CE control",
                "path": "curves/e05_weighted_control.png",
                "caption": "Same inverse-frequency CE and validation-R² selection as E05, but without count loss.",
            },
            {
                "name": "Count loss weight 0.1",
                "path": "curves/e05_weighted_count_w01.png",
                "caption": "Clean weighted-CE count-loss candidate; validation R² 0.5492 with comparatively stronger mPQ+ 0.2654.",
            },
            {
                "name": "Count loss weight 0.3",
                "path": "curves/e05_weighted_count_w03.png",
                "caption": "Clean weighted-CE count-loss candidate; validation R² selected over the full bracketed LR sweep.",
            },
            {
                "name": "Count loss weight 1.0 (selected)",
                "path": "curves/e05_weighted_count_w10.png",
                "caption": "Selected for E05 by validation R²: 1e-4 at epoch 11, validation R² 0.5762.",
            },
            {
                "name": "PQ-surrogate matched weighted-CE control",
                "path": "curves/e06_weighted_control.png",
                "caption": "Same inverse-frequency CE and validation-mPQ+ selection as E06, but without the PQ surrogate.",
            },
            {
                "name": "PQ surrogate weight 0.1",
                "path": "curves/e06_pq_w01.png",
                "caption": "Validation mPQ+ 0.2606 after the bracketed LR sweep.",
            },
            {
                "name": "PQ surrogate weight 0.3",
                "path": "curves/e06_pq_w03.png",
                "caption": "Validation mPQ+ 0.2615 after the bracketed LR sweep.",
            },
            {
                "name": "PQ surrogate weight 1.0 (selected)",
                "path": "curves/e06_pq_w10.png",
                "caption": "Selected for E06 by validation mPQ+: 3e-4 at epoch 16, validation mPQ+ 0.2641; the small gain did not transfer to test.",
            },
            {
                "name": "SAM-H LoRA segmentation learning-rate sweep",
                "path": "curves/e07_lora_lr_sweep.png",
                "caption": "True LoRA keeps frozen normalization buffers fixed and matches deployment batch/precision. Selected 1e-5 at epoch 2: validation bPQ 0.5389 versus zero-adapter 0.5338; freshly extracted persisted masks reproduced every PQ statistic exactly.",
            },
            {
                "name": "True-LoRA token-classifier learning-rate sweep",
                "path": "curves/e07_lora_classifier_lr_sweep.png",
                "caption": "Selected 1e-4 at epoch 18 by validation macro-F1 0.5503; 3e-4 and 1e-3 generalized worse despite continued training-loss improvements.",
            },
            {
                "name": "2× HV decoder validation sweep",
                "path": "curves/hv_decoder_2x_sweep.png",
                "caption": "Validation-selected 0.25-mpp-equivalent decoder: native tuned bPQ 0.5338 → 2× 0.5458; locked flip-TTA reaches 0.5657.",
            },
            {
                "name": "2× HV token-classifier learning-rate sweep",
                "path": "curves/hv_2x_classifier_lr_sweep.png",
                "caption": "Selected 1e-4 at epoch 19 by validation macro-F1 0.5516; the isolated run improves mPQ+ but creates a count-R² tradeoff.",
            },
            {
                "name": "2× HV + flip-TTA classifier extension",
                "path": "curves/hv_tta_2x_classifier_extended.png",
                "caption": "The bracketed sweep selected 1e-4 at its epoch-20 boundary, so that rate alone was extended with early stopping. Validation macro-F1 selected epoch 30 (0.5848), preventing an undertrained endpoint.",
            },
            {
                "name": "Minority-patch LoRA learning-rate sweep",
                "path": "curves/e14_lora_minority_lr_sweep.png",
                "caption": "Selected 1e-5 at epoch 2 by validation bPQ 0.53926, only +0.00036 over uniform LoRA. A class-stratified detection gate found no material rare-cell gain, so full extraction was skipped.",
            },
            {
                "name": "Color-jitter + blur LoRA initial LR sweep",
                "path": "curves/e10_lora_color_blur_lr_sweep.png",
                "caption": "The initial bracket selected 1e-4 at its epoch-6 boundary, requiring an extension rather than accepting an undertrained endpoint.",
            },
            {
                "name": "Color-jitter + blur selected-LR extension",
                "path": "curves/e10_lora_color_blur_extended.png",
                "caption": "The 1e-4 extension reached validation bPQ 0.5430 at epoch 12, but a class/source audit rejected it because higher SQ and fewer FPs hid severe rare-cell recall loss.",
            },
            {
                "name": "Color-jitter-only LoRA LR sweep",
                "path": "curves/e15_lora_color_lr_sweep.png",
                "caption": "Selected 1e-5 at epoch 2 by validation bPQ 0.53885, an exact null versus clean LoRA 0.53889; class/source effects were negligible.",
            },
            {
                "name": "E17 matched inverse-frequency CE control",
                "path": "curves/e17_ce_control.png",
                "caption": "Direct validation metric selection reveals unstable count bias from aggressive inverse-frequency weighting; selected validation R²/mPQ+ was only 0.1487/0.3084.",
            },
            {
                "name": "E17 complement-balanced focal type head",
                "path": "curves/e17_focal.png",
                "caption": "Selected 3e-4 at epoch 11 directly on validation mean(R², mPQ+): 0.7024/0.3227. Raw test 0.7206/0.3184 remains the count-R² high-water.",
            },
            {
                "name": "E19 type-head log-probability ensemble",
                "path": "curves/e19_type_ensemble.png",
                "caption": "Validation selected 65% focal logits. Test 0.7085/0.3224 was a PQ-oriented Pareto point before E21 rotation-TTA reached 0.3303 mPQ+.",
            },
            {
                "name": "E18 focal head on 2×-TTA masks",
                "path": "curves/e18_2x_tta_focal.png",
                "caption": "Selected 1e-3 at epoch 11 on validation 0.6507/0.3220, but transferred to only 0.6251/0.3155; the 2× rare-class detection deficit remained.",
            },
            {
                "name": "E20 selected rho=2 focal sweep",
                "path": "curves/e20_rho2.png",
                "caption": "The local grid selected rho=2 at validation 0.7117/0.3179, but test fell to 0.7005/0.3127; E17 remains the less-overfit balanced best.",
            },
            {
                "name": "E21 rotation-TTA focal-head sweep",
                "path": "curves/e21_rotation_tta_focal.png",
                "caption": "Selected 3e-4 at epoch 18 on validation 0.7167/0.3327. Test 0.7123/0.3303 raises all six class PQs, while stronger count suppression worsens GLaS/DPath undercount tails.",
            },
            {
                "name": "E22 per-class count-ensemble sweep",
                "path": "curves/e22_count_ensemble.png",
                "caption": "Validation selects one E17/E21 blend weight per class. Validation R² 0.7358 transfers to test R² 0.7255, while E21 masks preserve mPQ+ 0.3303.",
            },
            {
                "name": "E23 source-stratified HED augmentation review",
                "path": "curves/e23_hed_augmentation_examples.png",
                "caption": "Original plus three deterministic HED variants for one training patch per source. A green reconstruction artifact was caught here, fixed by non-negative optical-density constraints, and guarded before restarting training.",
            },
            {
                "name": "E23 HED LoRA learning-rate sweep",
                "path": "curves/e23_lora_hed_lr_sweep.png",
                "caption": "All LoRA learning rates and epochs; the gold marker is the validation-bPQ-selected checkpoint.",
            },
            {
                "name": "E24 sampling-budget audit",
                "path": "curves/e24_sampling_budget.png",
                "caption": "Expected source and cell-type exposure under uniform, 50% source-balanced, and source×class-aware replacement sampling.",
            },
            {
                "name": "E24 source-balanced LoRA learning-rate sweep",
                "path": "curves/e24_lora_source_lr_sweep.png",
                "caption": "Learning-rate and early-stopping trajectories for the isolated source-balanced sampler.",
            },
            {
                "name": "E26 four-direction LoRA learning-rate sweep",
                "path": "curves/e26_directional_lr_sweep.png",
                "caption": "LoRA base LR and 10× map-header LR are recorded together; selection uses locked-decoder validation bPQ before downstream metric-specific heads.",
            },
            {
                "name": "E26 leave-GLaS-out clean-LoRA control",
                "path": "curves/e26_leave_glas_clean_lr_sweep.png",
                "caption": "Exact source-held-out control with no GLaS overlap: LR 3e-6 at epoch 1, followed by a matched decoder sweep; flip-TTA bPQ 0.48510.",
            },
            {
                "name": "E26 leave-GLaS-out four-direction sweep",
                "path": "curves/e26_leave_glas_directional_lr_sweep.png",
                "caption": "Selected LR 1e-5 at epoch 10. Matched decoder tuning and flip-TTA reach bPQ 0.49946, +0.01436 over clean LoRA with a positive paired bootstrap interval.",
            },
            {
                "name": "E26 R²-selected focal-head sweep",
                "path": "curves/e26_r2_focal_control.png",
                "caption": "Learning rate is selected directly by validation R²: 3e-4 at epoch 18 (R² 0.67982). Test R² is 0.68296, below E17/E22 despite stronger binary masks.",
            },
            {
                "name": "E26 mPQ+-selected focal control",
                "path": "curves/e26_mpq_focal_control.png",
                "caption": "Matched type-head control selected directly by validation mPQ+: 3e-4 at epoch 15 (mPQ+ 0.32259).",
            },
            {
                "name": "E26 mPQ+ PQ-surrogate candidate",
                "path": "curves/e26_mpq_pq_w003.png",
                "caption": "The 0.03 PQ surrogate selects LR 1e-3 at epoch 11 and gives a tiny validation gain to 0.32286, but test mPQ+ falls to 0.31244; rejected.",
            },
            {
                "name": "E27 stable-token R² sweep",
                "path": "curves/e27_r2_focal_control.png",
                "caption": "E26 masks with untouched base flip+rotation tokens. Validation selects LR 3e-4, epoch 11 at R² 0.69581; test reaches 0.71804 but does not beat E22's 0.7255.",
            },
            {
                "name": "E27 stable-token mPQ+ full-horizon sweep",
                "path": "curves/e27_mpq_focal_full_horizon.png",
                "caption": "Equal 20-epoch horizons prevent premature LR rejection. LR 1e-3, epoch 13 reaches validation mPQ+ 0.33471, but locked test mPQ+ 0.32930 does not beat E21.",
            },
            {
                "name": "E28 E22/E27 per-class count ensemble",
                "path": "curves/e28_e22_e27_count_ensemble.png",
                "caption": "Six weights are selected independently on validation R². Validation 0.74033 transfers to test 0.73200; E21 masks preserve mPQ+ 0.33027.",
            },
            {
                "name": "E29 E21 per-class confidence rejection",
                "path": "curves/e29_e21_class_confidence_rejection.png",
                "caption": "All six validation gains are microscopic. Joint rejection raises validation mPQ+ by 0.00141, but test mPQ+ is flat/slightly worse and R² falls 0.00731; retain confidence for review triage, not automatic deletion.",
            },
            {
                "name": "E31 extended Sobel-kernel and edge-threshold sweep",
                "path": "curves/e31_extended_sobel_edge_sweep.png",
                "caption": "The complete legal kernel grid turns over at k=3 after interacting settings are tuned; edge threshold turns over at 0.5. Flip-TTA gains only 0.00004 bPQ with a paired CI spanning zero, so the previous decoder remains deployed.",
            },
            {
                "name": "E32 leakage-free HoVer-Net phase-1 LR pilot",
                "path": "curves/e32_hovernet_phase1_lr_pilot.png",
                "caption": "Official architecture and six-term loss on our group-disjoint fold. LR 1e-4 is the smooth mPQ+ path; 3e-4 has the best pilot R² but severe rare-class count oscillation, so the metrics proceed as separate validation-selected branches.",
            },
            {
                "name": "E32 full HoVer-Net class diagnostics",
                "path": "curves/e32_hovernet_phase1_diagnostics.png",
                "caption": "Full phase-1 learning curves with per-class R², PQ, signed count bias, and count ratios. The scheduled 1e-5 refinement reaches epoch-50 validation 0.79662/0.43989.",
            },
            {
                "name": "E32 validation detection-versus-typing decomposition",
                "path": "curves/e32_tta_val_detection_typing.png",
                "caption": "Class-agnostic IoU>0.5 matching separates missed nuclei from type confusion. Neutrophil is weak in both detection and typing; eosinophil is primarily detection-limited; plasma is primarily confused with lymphocyte.",
            },
            {
                "name": "E33 HoVer-Net/CellViT count blend",
                "path": "curves/e33_hovernet_tta_e28_count_blend.png",
                "caption": "Validation-select six count weights while keeping HoVer-Net TTA masks/types fixed. Group-disjoint OOF R² is 0.85120 and locked test R² is 0.84563.",
            },
            {
                "name": "E36 empirical joint H/E concentration profile",
                "path": "curves/e36_hed_concentration_distribution.png",
                "caption": "Patch-p95 H/E concentration pairs from development data. Training samples observed joint pairs fold-locally, preserving multimodality and H/E correlation while excluding locked test pixels.",
            },
            {
                "name": "E36 empirical H/E transform safety audit",
                "path": "curves/e36_hovernet_hed_examples.png",
                "caption": "Source-stratified examples from the qualified empirical transfer: 99.8% moved toward their sampled target and 0.2% were rejected by visual safety guards.",
            },
            {
                "name": "E36 exact seed-matched H/E causal audit",
                "path": "curves/e36_hed_vs_matched_control.png",
                "caption": "Complete 3-LR × 10-epoch comparison on the 711-patch development validation set. Fixed-strength H/E loses independently selected R² by 0.06014; its boundary mPQ+ delta is only +0.00172 and is entirely driven by neutrophil while four classes regress, so the recipe is rejected.",
            },
            {
                "name": "E37/E45 source/class sampling and replacement-control budget",
                "path": "curves/e37_sampling_budget.png",
                "caption": "Expected source and rare-class exposure, effective sample size, and unique-patch cost for ordinary no-replacement sampling, uniform replacement, class fractions 0.10/0.25/0.50, and source-aware alternatives. Uniform replacement isolates duplicate-producing draw mode from class reweighting.",
            },
            {
                "name": "E37 class-balanced LR pilot (live)",
                "path": "curves/e37_class_sampling_lr_live.png",
                "caption": "Development-validation R², mPQ+, mDQ+, mSQ+, loss, and per-class R² for completed class-0.5 checkpoints. Interrupted attempts are archived and excluded; the displayed run uses the original matched persistent-worker protocol from epoch 0.",
            },
            {
                "name": "E37 exact seed-206 no-sampling control (live)",
                "path": "curves/e37_no_sampling_control_lr_live.png",
                "caption": "The causal denominator for both source- and class-aware replacement sampling on the 711-patch development validation set. Compare only identical LR/epoch checkpoints until the full grid permits independent metric-specific selection.",
            },
            {
                "name": "E38 native HoVer-Net performance by H/E bin",
                "path": "curves/e38_hed_bin_performance_native.png",
                "caption": "Training-defined joint H/E bins with validation mPQ+, R², signed count error, and support. H3/E2 selects the mPQ+ anchor and H3/E3 the R² anchor; both are DPath-dominated, so paired two-view inference remains the required gate.",
            },
            {
                "name": "E39 six-fold HoVer-Net training audit",
                "path": "curves/e39_hovernet_fold_training_curves.png",
                "caption": "Fold-overlaid training/validation losses and group-disjoint mPQ+/R² checkpoints. The dashed line marks the planned LR reduction from 1e-4 to 1e-5 after epoch 25; the plot is regenerated as folds complete.",
            },
            {
                "name": "E41 heterogeneous-backbone learning-rate pilot",
                "path": "curves/e41_seresnext101_lr_live.png",
                "caption": "SE-ResNeXt-101 validation R², mPQ+, mDQ+, mSQ+, loss, and class R². Dashed references are the independently selected seed-205 ResNet pilot endpoints; selection remains open until all three learning rates complete equal horizons.",
            },
            {
                "name": "E43 pooled per-nucleus type-loss learning-rate pilot (live)",
                "path": "curves/e43_instance_type_lr_live.png",
                "caption": "Weight-0.1 development-validation R², mPQ+, mDQ+, mSQ+, recipe loss, and per-class exact-control PQ deltas at each scored checkpoint. Solid lines are E43 and dashed lines are seed/LR/epoch-matched ordinary-loss controls; no locked-test inference.",
            },
            {
                "name": "E43 pooled per-nucleus type-loss dose response (live)",
                "path": "curves/e43_instance_type_dose_live.png",
                "caption": "At validation-selected LR 1e-4, compare auxiliary weights 0, 0.05, 0.1, and 0.25 on R², mPQ+, DQ, SQ, per-nucleus type NLL, and per-class exact-control PQ deltas. Curves populate only at scheduled metric checkpoints; no locked-test inference.",
            },
        ],
    }
    _attach_curve_evidence(summary["rows"], summary["curves"])
    return summary


def _scatter_macro_r2(scatter_points: list[dict], key: str) -> float:
    """Macro-R² over plotted patch×class points, matching the dashboard's JS."""
    by_class: dict[str, list[dict]] = {}
    for point in scatter_points:
        if key in point:
            by_class.setdefault(point["class_name"], []).append(point)
    scores = []
    for points in by_class.values():
        truth = np.asarray([p["gt"] for p in points], dtype=np.float64)
        predicted = np.asarray([p[key] for p in points], dtype=np.float64)
        denominator = float(np.square(truth - truth.mean()).sum())
        if denominator > 0:
            scores.append(1.0 - float(np.square(predicted - truth).sum()) / denominator)
    return float(np.mean(scores)) if scores else float("nan")


def _attach_curve_evidence(rows: list[dict], curves: list[dict], cap: int = 4) -> None:
    """Give rows without curated visuals their matching figures from the curve library.

    Each curve filename starts with its experiment id (``e05_...`` -> ``E05``);
    rows that already have hand-curated visuals keep them. This closes the gap
    where most experiments showed no visual evidence despite having figures on disk.
    """
    by_id: dict[str, list[dict]] = {}
    for curve in curves:
        path = curve.get("path", "")
        # Many library entries are aspirational; only attach figures that actually
        # exist on disk, so the strict Pages link-check never sees a dangling image.
        if not (ROOT / "outputs" / "conic_experiments" / path).exists():
            continue
        match = re.match(r"e(\d+)", Path(path).name, re.IGNORECASE)
        if not match:
            continue
        key = f"E{int(match.group(1)):02d}"
        by_id.setdefault(key, []).append(
            {"path": path, "label": curve.get("name") or curve.get("caption", "Experiment figure")}
        )
    for row in rows:
        if row.get("visuals"):
            continue
        figures = by_id.get(row.get("id"))
        if figures:
            row["visuals"] = figures[:cap]


def require_complete_dashboard_summary(performance: dict, runs_root: Path) -> None:
    """Fail loudly when requested interactive diagnostics were not materialized."""
    subgroup = performance.get("subgroups") or {}
    required = ("by_class", "by_institution", "by_both", "confusions", "scatter_points")
    missing = [name for name in required if not subgroup.get(name)]
    if missing:
        expected = runs_root / "full"
        raise RuntimeError(
            "Dashboard subgroup diagnostics are incomplete; refusing to render a page that "
            f"silently omits {', '.join(missing)}. Check --runs-root={runs_root}; the initial "
            f"CellViT artifacts are expected under {expected}."
        )
    # The scatter/count diagnostics are rebuilt from the count arrays passed on the
    # command line, while the headline rows come from each method's own metrics
    # artifact. If a recomputed series disagrees with its artifact, the wrong count
    # file was passed (e.g. an out-of-fold blend instead of the locked test counts)
    # and the dashboard would contradict itself. Every scatter series that maps to a
    # known method is checked, not just the E33 blend, so an intermediate/baseline
    # mismatch cannot slip through either.
    scatter_points = subgroup.get("scatter_points") or []
    baseline_row = performance.get("baseline", {})
    best_mpq_row = performance.get("best_mpq", {})
    intermediate_name = subgroup.get("intermediate_name") or ""
    checks = [
        ("best_pred", performance.get("best_r2", {}).get("r2"), subgroup.get("best_name", "current best")),
        ("baseline_pred", baseline_row.get("r2"), subgroup.get("baseline_name", "baseline")),
    ]
    # The intermediate series is the mPQ+ leader's mask-derived counts only when the
    # subgroup names it as such; guard against that artifact R² when so.
    if best_mpq_row.get("id") and best_mpq_row["id"] in intermediate_name:
        checks.append(("rotation_pred", best_mpq_row.get("r2"), intermediate_name))
    for key, artifact_r2, label in checks:
        if artifact_r2 is None or not any(key in point for point in scatter_points):
            continue
        scatter_r2 = _scatter_macro_r2(scatter_points, key)
        if np.isfinite(scatter_r2) and abs(scatter_r2 - float(artifact_r2)) > 0.02:
            raise RuntimeError(
                f"Count diagnostics disagree with the headline result: the scatter recomputes macro "
                f"R² {scatter_r2:.4f} for '{label}' ({key}), but its metrics artifact reports "
                f"{float(artifact_r2):.4f}. The count array feeding this series likely does not match the "
                "method that set its headline number (e.g. an out-of-fold cross-validation blend was passed "
                "instead of the locked-test counts). Refusing to render a self-contradicting page."
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, default=None, help="Optional .npz with patch_ids, instance_ids, and class_probs")
    parser.add_argument("--counts", type=Path, default=None, help="Optional final full-dataset count array, e.g. E33 blended counts")
    parser.add_argument("--intermediate-counts", type=Path, default=None, help="Optional count array shown as an intermediate method")
    parser.add_argument("--count-method-label", default="recommended-model counts", help="Label shown on review cards for the primary count array")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-cases", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--reuse-rendered-cases",
        action="store_true",
        help="Reuse outdir/rendered_cases.json; safe only when predictions, probabilities, counts, split, and selected cases are unchanged.",
    )
    parser.add_argument("--metrics", type=Path, default=ROOT / "outputs" / "metrics_test.json")
    parser.add_argument("--posthoc-report", type=Path, default=ROOT / "outputs" / "conic_experiments" / "posthoc" / "posthoc_report.json")
    parser.add_argument("--experiment-matrix", type=Path, default=ROOT / "experiments" / "conic_matrix.json")
    parser.add_argument("--runs-root", type=Path, default=Path.home() / "data" / "cpath_demos" / "conic" / "runs")
    args = parser.parse_args()
    predictions = np.load(args.predictions)
    final_counts = np.load(args.counts) if args.counts else None
    probability_by_patch = {}
    if args.probabilities:
        probability_data = np.load(args.probabilities)
        for patch_id, instance_id, probs in zip(probability_data["patch_ids"], probability_data["instance_ids"], probability_data["class_probs"]):
            probability_by_patch.setdefault(int(patch_id), {})[int(instance_id)] = probs
    case_cache = args.outdir / "rendered_cases.json"
    if args.reuse_rendered_cases:
        if not case_cache.exists():
            raise FileNotFoundError(f"--reuse-rendered-cases requested but cache is absent: {case_cache}")
        cases = json.loads(case_cache.read_text())
    else:
        metadata = load_metadata(args.prepared)
        ids, selection_reasons = choose_ids(metadata, args.split, args.max_cases, args.seed, predicted_counts=final_counts)
        cases = []
        for patch_id in ids:
            case = render_case(
                args.prepared,
                predictions,
                patch_id,
                args.outdir,
                args.split,
                probabilities=probability_by_patch.get(patch_id, {}),
                pred_counts_override=final_counts[patch_id] if final_counts is not None else None,
                count_method=args.count_method_label if final_counts is not None else "mask-derived counts",
            )
            case["selection_reason"] = selection_reasons.get(patch_id, "")
            cases.append(case)
        case_cache.parent.mkdir(parents=True, exist_ok=True)
        case_cache.write_text(json.dumps(cases, indent=2, allow_nan=True))
    triage = triage_cases(cases, max_cases=args.max_cases)
    (args.outdir / "agent_triage.json").parent.mkdir(parents=True, exist_ok=True)
    (args.outdir / "agent_triage.json").write_text(json.dumps(triage, indent=2))
    performance = build_performance_summary(
        args.metrics,
        args.posthoc_report,
        args.experiment_matrix,
        args.runs_root,
        args.prepared,
        best_predictions_path=args.predictions,
        best_counts_path=args.counts,
        intermediate_counts_path=args.intermediate_counts,
    )
    require_complete_dashboard_summary(performance, args.runs_root)
    (args.outdir / "performance_summary.json").write_text(json.dumps(performance, indent=2, allow_nan=True))
    visual_artifacts = [item for curve in performance.get("curves", []) for item in [curve]]
    visual_artifacts.extend(
        visual
        for row in performance.get("rows", [])
        for visual in row.get("visuals", [])
    )
    seen_visual_paths = set()
    for curve in visual_artifacts:
        relative = Path(curve["path"])
        if str(relative) in seen_visual_paths:
            continue
        seen_visual_paths.add(str(relative))
        source = ROOT / "outputs" / "conic_experiments" / relative
        target = args.outdir / relative
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    write_gallery(args.outdir, cases, triage, performance=performance)
    print(f"rendered {len(cases)} cases to {args.outdir}")


if __name__ == "__main__":
    main()
