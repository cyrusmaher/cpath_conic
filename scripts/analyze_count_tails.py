#!/usr/bin/env python3
"""Explain CoNIC macro-R² through signed bias, outlier rates, and normalized SSE."""
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
from scripts.sweep_count_ensemble import class_r2


def macro_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    values = [class_r2(truth[:, index], prediction[:, index]) for index in range(len(CLASS_NAMES))]
    return float(np.mean([value for value in values if np.isfinite(value)]))


def group_stats(frame: pd.DataFrame, thresholds: list[int]) -> dict:
    residual = frame.residual.to_numpy(dtype=np.float64)
    contribution = frame.normalized_sse.to_numpy(dtype=np.float64)
    return {
        "n": int(len(frame)),
        "mean_signed_error": float(residual.mean()),
        "mae": float(np.abs(residual).mean()),
        "under_fraction": float(np.mean(residual < 0)),
        "exact_fraction": float(np.mean(residual == 0)),
        "over_fraction": float(np.mean(residual > 0)),
        "normalized_sse": float(contribution.sum()),
        "normalized_sse_share": None,
        "outlier_fraction": {str(value): float(np.mean(np.abs(residual) > value)) for value in thresholds},
        "large_under_fraction": {str(value): float(np.mean(residual < -value)) for value in thresholds},
        "large_over_fraction": {str(value): float(np.mean(residual > value)) for value in thresholds},
    }


def tail_target(truth: np.ndarray, prediction: np.ndarray, target: float, scenario: str) -> dict | None:
    residual = prediction - truth
    thresholds = sorted(np.unique(np.abs(residual)).astype(int))
    result = None
    for threshold in thresholds:
        if scenario == "fix":
            candidate = prediction.copy()
            candidate[np.abs(residual) > threshold] = truth[np.abs(residual) > threshold]
            score = macro_r2(truth, candidate)
        else:
            scores = []
            for class_index in range(len(CLASS_NAMES)):
                keep = np.abs(residual[:, class_index]) <= threshold
                score_class = class_r2(truth[keep, class_index], prediction[keep, class_index])
                if np.isfinite(score_class):
                    scores.append(score_class)
            score = float(np.mean(scores)) if scores else float("nan")
        if np.isfinite(score) and score >= target:
            result = {
                "threshold": int(threshold),
                "resulting_R2": score,
                "affected_points": int(np.sum(np.abs(residual) > threshold)),
                "affected_fraction": float(np.mean(np.abs(residual) > threshold)),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--target-r2", type=float, default=0.8585)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[2, 5, 10, 20])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    selected = metadata.loc[metadata.split == args.split].sort_values("patch_id").reset_index(drop=True)
    patch_ids = selected.patch_id.to_numpy(dtype=np.int32)
    truth = selected[COUNT_COLUMNS].to_numpy(dtype=np.int32)
    prediction = np.load(args.counts)[patch_ids].astype(np.int32)
    residual = prediction - truth
    denominators = np.square(truth - truth.mean(axis=0, keepdims=True)).sum(axis=0)
    records = []
    for row_index, row in selected.iterrows():
        for class_index, class_name in enumerate(CLASS_NAMES):
            records.append(
                {
                    "patch_id": int(row.patch_id),
                    "source": str(row.source),
                    "source_group": str(row.source_group),
                    "class": class_name,
                    "gt": int(truth[row_index, class_index]),
                    "predicted": int(prediction[row_index, class_index]),
                    "residual": int(residual[row_index, class_index]),
                    "normalized_sse": float(residual[row_index, class_index] ** 2 / denominators[class_index])
                    if denominators[class_index] > 0
                    else 0.0,
                }
            )
    frame = pd.DataFrame(records)
    total_normalized_sse = float(frame.normalized_sse.sum())

    def summarize(columns: list[str]) -> list[dict]:
        output = []
        key = columns[0] if len(columns) == 1 else columns
        for values, rows in frame.groupby(key, sort=True):
            values = (values,) if len(columns) == 1 else values
            stats = group_stats(rows, args.thresholds)
            stats["normalized_sse_share"] = stats["normalized_sse"] / total_normalized_sse if total_normalized_sse else 0.0
            output.append({**dict(zip(columns, values)), **stats})
        return output

    targets = {
        scenario: tail_target(truth, prediction, args.target_r2, scenario)
        for scenario in ("filter", "fix")
    }
    tail_breakdowns = {}
    for scenario, result in targets.items():
        if result is None:
            tail_breakdowns[scenario] = []
            continue
        threshold = result["threshold"]
        tail = frame.loc[np.abs(frame.residual) > threshold]
        tail_breakdowns[scenario] = (
            tail.groupby(["source", "class"], sort=True)
            .agg(
                n=("residual", "size"),
                mean_signed_error=("residual", "mean"),
                mean_gt=("gt", "mean"),
                normalized_sse=("normalized_sse", "sum"),
            )
            .reset_index()
            .sort_values("normalized_sse", ascending=False)
            .to_dict("records")
        )
    report = {
        "split": args.split,
        "counts": str(args.counts),
        "n_patches": int(len(selected)),
        "R2": macro_r2(truth, prediction),
        "per_class_R2": {
            name: class_r2(truth[:, index], prediction[:, index]) for index, name in enumerate(CLASS_NAMES)
        },
        "target_R2": args.target_r2,
        "thresholds": args.thresholds,
        "overall": group_stats(frame, args.thresholds),
        "by_class": summarize(["class"]),
        "by_source": summarize(["source"]),
        "by_source_class": summarize(["source", "class"]),
        "tail_target": targets,
        "tail_breakdown": tail_breakdowns,
        "interpretation_note": "normalized_sse is squared error divided by that class's SST; summing it reflects each point's contribution to macro-R2 deficit. Test reports are retrospective only.",
    }
    report["overall"]["normalized_sse_share"] = 1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=True))
    print(json.dumps({key: report[key] for key in ("R2", "per_class_R2", "tail_target")}, indent=2))


if __name__ == "__main__":
    main()
