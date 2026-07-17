#!/usr/bin/env python3
"""Compare selected HoVer checkpoints by GT nucleus size on development validation."""

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
from scripts.analyze_hovernet_detection_typing import match_patch


def quantile_bin_labels(areas: np.ndarray) -> tuple[np.ndarray, list[str], list[float]]:
    """Assign deterministic validation-diagnostic area quartiles."""
    values = np.asarray(areas, dtype=np.float64)
    if values.ndim != 1 or not len(values) or np.any(values <= 0):
        raise ValueError("areas must be a nonempty vector of positive values")
    edges = np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).astype(float)
    # Discrete pixel areas can tie at a quantile.  Unique internal cut points
    # avoid empty/ambiguous intervals while retaining fixed endpoint labels.
    cuts = np.unique(edges[1:-1])
    cuts = cuts[(cuts > values.min()) & (cuts < values.max())]
    bins = np.searchsorted(cuts, values, side="right")
    boundaries = [float(values.min()), *map(float, cuts), float(values.max())]
    labels = []
    for index in range(len(boundaries) - 1):
        left, right = boundaries[index], boundaries[index + 1]
        labels.append(f"Q{index + 1} · {left:g}–{right:g} px")
    return bins.astype(np.int16), labels, boundaries


def grouped_detection(records: pd.DataFrame, group_columns: list[str]) -> list[dict]:
    """Aggregate paired control/candidate GT outcomes with sample sizes."""
    rows = []
    grouper = group_columns[0] if len(group_columns) == 1 else group_columns
    for group, frame in records.groupby(grouper, sort=True, observed=True):
        values = (group,) if len(group_columns) == 1 else tuple(group)
        control_matched = int(frame.control_matched.sum())
        candidate_matched = int(frame.candidate_matched.sum())
        support = int(len(frame))
        row = dict(zip(group_columns, map(str, values)))
        row.update({
            "gt_instances": support,
            "control_matched": control_matched,
            "candidate_matched": candidate_matched,
            "control_detection_recall": control_matched / support,
            "candidate_detection_recall": candidate_matched / support,
            "delta_detection_recall": (candidate_matched - control_matched) / support,
            "control_matched_mean_iou": float(frame.loc[frame.control_matched, "control_iou"].mean())
            if control_matched else None,
            "candidate_matched_mean_iou": float(frame.loc[frame.candidate_matched, "candidate_iou"].mean())
            if candidate_matched else None,
        })
        if control_matched and candidate_matched:
            row["delta_matched_mean_iou"] = (
                row["candidate_matched_mean_iou"] - row["control_matched_mean_iou"]
            )
        else:
            row["delta_matched_mean_iou"] = None
        rows.append(row)
    return rows


def patch_outcomes(truth: np.ndarray, prediction: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, int]:
    matches, gt_classes, pred_classes, ious = match_patch(truth, prediction, threshold)
    matched = np.zeros(len(gt_classes), dtype=bool)
    matched_iou = np.full(len(gt_classes), np.nan, dtype=np.float64)
    if len(matches):
        matched[matches[:, 0]] = True
        matched_iou[matches[:, 0]] = ious
    spurious = int(len(pred_classes) - len(matches))
    return matched, matched_iou, spurious


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--control-name", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    control = np.load(args.control_artifact)
    candidate = np.load(args.candidate_artifact)
    patch_ids = control["patch_ids"].astype(np.int32)
    if not np.array_equal(patch_ids, candidate["patch_ids"]):
        raise ValueError("control and candidate patch IDs differ")
    metadata = pd.read_csv(args.prepared / "metadata.csv").set_index("patch_id").loc[patch_ids]
    if metadata.split.eq("test").any():
        raise RuntimeError("size audit refuses locked-test patches")

    records = []
    spurious = []
    for index, patch_id in enumerate(patch_ids):
        label = np.load(args.prepared / "labels" / f"{int(patch_id):05d}.npy", allow_pickle=True).item()
        truth = np.stack([label["inst_map"], label["class_map"]], axis=-1).astype(np.int32)
        gt_ids = np.unique(truth[..., 0])
        gt_ids = gt_ids[gt_ids > 0]
        areas = np.bincount(truth[..., 0].ravel())[gt_ids]
        classes = []
        for instance_id in gt_ids:
            values = truth[..., 1][truth[..., 0] == instance_id]
            values = values[(values >= 1) & (values <= len(CLASS_NAMES))]
            classes.append(int(np.bincount(values).argmax()) if len(values) else 0)
        control_matched, control_iou, control_spurious = patch_outcomes(
            truth, control["predictions"][index], args.iou_threshold
        )
        candidate_matched, candidate_iou, candidate_spurious = patch_outcomes(
            truth, candidate["predictions"][index], args.iou_threshold
        )
        source = str(metadata.loc[int(patch_id), "source"])
        for gt_index, (instance_id, area, class_id) in enumerate(zip(gt_ids, areas, classes)):
            if not 1 <= class_id <= len(CLASS_NAMES):
                continue
            records.append({
                "patch_id": int(patch_id),
                "source": source,
                "class": CLASS_NAMES[class_id - 1],
                "gt_instance_id": int(instance_id),
                "area_pixels": int(area),
                "control_matched": bool(control_matched[gt_index]),
                "candidate_matched": bool(candidate_matched[gt_index]),
                "control_iou": float(control_iou[gt_index]),
                "candidate_iou": float(candidate_iou[gt_index]),
            })
        spurious.append({
            "patch_id": int(patch_id), "source": source,
            "control_spurious": control_spurious, "candidate_spurious": candidate_spurious,
        })

    frame = pd.DataFrame(records)
    bins, bin_labels, bin_boundaries = quantile_bin_labels(frame.area_pixels.to_numpy())
    frame["size_bin"] = pd.Categorical.from_codes(bins, bin_labels, ordered=True)
    spurious_frame = pd.DataFrame(spurious)
    overall = grouped_detection(frame.assign(overall="all"), ["overall"])[0]
    report = {
        "protocol": "development-validation-only paired GT-nucleus detection audit; size quartiles are diagnostic and never model-selection inputs",
        "evaluation_set": f"{len(patch_ids)}-patch source-group-disjoint development validation",
        "control": args.control_name,
        "candidate": args.candidate_name,
        "iou_threshold_strictly_greater_than": args.iou_threshold,
        "size_bin_boundaries_pixels": bin_boundaries,
        "overall": overall,
        "by_size": grouped_detection(frame, ["size_bin"]),
        "by_class": grouped_detection(frame, ["class"]),
        "by_source": grouped_detection(frame, ["source"]),
        "by_size_and_source": grouped_detection(frame, ["size_bin", "source"]),
        "by_size_and_class": grouped_detection(frame, ["size_bin", "class"]),
        "spurious_predictions": {
            "control_total": int(spurious_frame.control_spurious.sum()),
            "candidate_total": int(spurious_frame.candidate_spurious.sum()),
            "delta_total": int((spurious_frame.candidate_spurious - spurious_frame.control_spurious).sum()),
            "by_source": [
                {
                    "source": str(source),
                    "patches": int(len(group)),
                    "control": int(group.control_spurious.sum()),
                    "candidate": int(group.candidate_spurious.sum()),
                    "delta": int((group.candidate_spurious - group.control_spurious).sum()),
                }
                for source, group in spurious_frame.groupby("source", sort=True)
            ],
        },
        "mechanism_gate": (
            "Support equalization only if the smallest bins gain detection recall without a disproportionate "
            "spurious-prediction increase, while pooled mDQ+ improves and mSQ+ remains within 0.005."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "evaluation_set": report["evaluation_set"],
        "overall": report["overall"],
        "by_size": report["by_size"],
        "spurious_predictions": report["spurious_predictions"],
    }, indent=2))


if __name__ == "__main__":
    main()
