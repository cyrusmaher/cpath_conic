#!/usr/bin/env python
"""Compare two fixed-mask classifiers at the level that drives CoNIC metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS
from cpath_conic.data import load_metadata
from scripts.render_review import _detection_aware_confusion, _fixed_mask_method_metrics


def signed_error_summary(true: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted.astype(np.float64) - true.astype(np.float64)
    absolute = np.abs(error)
    return {
        "n": int(len(error)),
        "mean_signed_error": float(error.mean()),
        "mae": float(absolute.mean()),
        "p90_absolute_error": float(np.quantile(absolute, 0.9)),
        "under_fraction": float((error < 0).mean()),
        "exact_fraction": float((error == 0).mean()),
        "over_fraction": float((error > 0).mean()),
        "absolute_error_gt_10_fraction": float((absolute > 10).mean()),
    }


def method_driver_report(metrics: dict, sources: dict[str, dict], metadata) -> dict:
    test = metadata.loc[metadata.split == "test"].sort_values("patch_id")
    true = test[COUNT_COLUMNS].to_numpy(dtype=np.float64)
    predicted = metrics["predicted_counts"].astype(np.float64)
    classes = {}
    for index, name in enumerate(CLASS_NAMES):
        centered = true[:, index] - true[:, index].mean()
        sst = float(np.square(centered).sum())
        sse = float(np.square(predicted[:, index] - true[:, index]).sum())
        classes[name] = {
            "R2": float(metrics["per_class_R2"][name]),
            "count_SSE": sse,
            "count_SST": sst,
            "signed_error": signed_error_summary(true[:, index], predicted[:, index]),
            **metrics["per_class_pq"][name],
        }
    source_rows = {}
    for source, source_metrics in sources.items():
        rows = test.loc[test.source == source]
        source_true = rows[COUNT_COLUMNS].to_numpy(dtype=np.float64)
        source_predicted = predicted[test.source.to_numpy() == source]
        source_rows[source] = {
            "patches": int(len(rows)),
            "R2": float(source_metrics["R2"]),
            "mPQ+": float(source_metrics["mPQ+"]),
            "signed_error_all_class_patch_points": signed_error_summary(source_true.ravel(), source_predicted.ravel()),
        }
    return {
        "R2": float(metrics["R2"]),
        "mPQ+": float(metrics["mPQ+"]),
        "classes": classes,
        "sources": source_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    for prefix in ("reference", "candidate"):
        parser.add_argument(f"--{prefix}-features", type=Path, required=True)
        parser.add_argument(f"--{prefix}-probabilities", type=Path, required=True)
        parser.add_argument(f"--{prefix}-cache", type=Path, required=True)
        parser.add_argument(f"--{prefix}-name", default=prefix)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metadata = load_metadata(args.prepared).sort_values("patch_id")
    methods = {}
    diagnostics = {}
    gt_counts = None
    for prefix in ("reference", "candidate"):
        metrics, sources, method_gt, diagnostic = _fixed_mask_method_metrics(
            args.prepared,
            getattr(args, f"{prefix}_features"),
            getattr(args, f"{prefix}_probabilities"),
            getattr(args, f"{prefix}_cache"),
        )
        if gt_counts is not None and not np.array_equal(gt_counts, method_gt):
            raise ValueError("Ground-truth support differs between methods")
        gt_counts = method_gt
        methods[prefix] = method_driver_report(metrics, sources, metadata)
        diagnostics[prefix] = diagnostic

    reference = methods["reference"]
    candidate = methods["candidate"]
    deltas = {
        "R2": candidate["R2"] - reference["R2"],
        "mPQ+": candidate["mPQ+"] - reference["mPQ+"],
        "classes": {},
        "sources": {},
    }
    for name in CLASS_NAMES:
        left, right = reference["classes"][name], candidate["classes"][name]
        deltas["classes"][name] = {
            key: right[key] - left[key]
            for key in ("R2", "count_SSE", "pq", "dq", "sq", "tp", "fp", "fn")
        }
    for source in sorted(reference["sources"]):
        deltas["sources"][source] = {
            "R2": candidate["sources"][source]["R2"] - reference["sources"][source]["R2"],
            "mPQ+": candidate["sources"][source]["mPQ+"] - reference["sources"][source]["mPQ+"],
        }
    report = {
        "reference_name": args.reference_name,
        "candidate_name": args.candidate_name,
        "methods": methods,
        "deltas_candidate_minus_reference": deltas,
        "confusions": {
            "reference": _detection_aware_confusion(diagnostics["reference"], gt_counts),
            "candidate": _detection_aware_confusion(diagnostics["candidate"], gt_counts),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"reference": [reference["R2"], reference["mPQ+"]], "candidate": [candidate["R2"], candidate["mPQ+"]], "delta": [deltas["R2"], deltas["mPQ+"]]}, indent=2))


if __name__ == "__main__":
    main()
