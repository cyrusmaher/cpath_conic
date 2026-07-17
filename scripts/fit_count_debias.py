#!/usr/bin/env python
"""Fit a per-class multiplicative count debias on validation, apply it to test.

The recommended HoVer-Net model (E48) produces mask-derived cell counts that carry
a systematic per-class bias — most visibly, epithelial is over-counted. We correct
it with the simplest transform that generalizes: one multiplicative factor per cell
type, fit on the development-validation split so that each class's total predicted
count matches its validation ground-truth total, then frozen and applied to test.

Masks and types are untouched, so mPQ+ is unchanged; only the count R² moves. A
richer per-class linear fit was tried and rejected — it overfits the rare classes
(eosinophil), the same failure that sank the E34 count stacker.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from cpath_conic.data import load_metadata
from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS


def _macro_r2(truth: np.ndarray, predicted: np.ndarray) -> tuple[float, dict[str, float]]:
    per_class: dict[str, float] = {}
    for index, name in enumerate(CLASS_NAMES):
        gt = truth[:, index].astype(np.float64)
        pred = predicted[:, index].astype(np.float64)
        denominator = float(np.square(gt - gt.mean()).sum())
        per_class[name] = float(1.0 - np.square(pred - gt).sum() / denominator) if denominator > 0 else float("nan")
    finite = [value for value in per_class.values() if np.isfinite(value)]
    return (float(np.mean(finite)) if finite else float("nan")), per_class


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=Path.home() / "data" / "cpath_demos" / "conic" / "prepared")
    parser.add_argument(
        "--val-counts",
        type=Path,
        default=ROOT / "outputs" / "conic_final_stack" / "fusion" / "weighted_tta_val" / "counts.npy",
    )
    parser.add_argument(
        "--test-counts",
        type=Path,
        default=ROOT / "outputs" / "conic_final_stack" / "locked_weighted_tta_test" / "counts.npy",
    )
    parser.add_argument(
        "--model-metrics",
        type=Path,
        default=ROOT / "outputs" / "conic_final_stack" / "locked_weighted_tta_test" / "metrics_test.json",
        help="E48 test metrics; mPQ+/mDQ+/mSQ+ are carried over unchanged (masks are untouched).",
    )
    parser.add_argument(
        "--out-counts",
        type=Path,
        default=ROOT / "outputs" / "conic_final_stack" / "e50_debias_test_counts.npy",
    )
    parser.add_argument(
        "--out-metrics",
        type=Path,
        default=ROOT / "outputs" / "conic_final_stack" / "e50_debias_metrics_test.json",
    )
    args = parser.parse_args()

    metadata = load_metadata(args.prepared).sort_values("patch_id")
    val = metadata.loc[metadata.split == "val"].sort_values("patch_id")
    test = metadata.loc[metadata.split == "test"].sort_values("patch_id")
    val_ids = val.patch_id.to_numpy(np.int32)
    test_ids = test.patch_id.to_numpy(np.int32)
    val_truth = val[COUNT_COLUMNS].to_numpy(np.float64)
    test_truth = test[COUNT_COLUMNS].to_numpy(np.int64)

    val_counts = np.load(args.val_counts)
    test_counts = np.load(args.test_counts)
    val_pred = val_counts[val_ids]
    if val_pred.shape[1] != len(CLASS_NAMES) or test_counts.shape[1] != len(CLASS_NAMES):
        raise ValueError("Count arrays must have six class columns")

    # One factor per class: make each class's validation total match its GT total.
    val_totals = val_pred.sum(axis=0)
    scales = np.where(val_totals > 0, val_truth.sum(axis=0) / np.maximum(val_totals, 1e-9), 1.0)

    # Freeze the scales and apply to the full test-count array; round to whole cells.
    debiased_full = np.rint(np.clip(test_counts.astype(np.float64) * scales[None, :], 0, None)).astype(np.int32)
    args.out_counts.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_counts, debiased_full)

    raw_r2, raw_per_class = _macro_r2(test_truth, test_counts[test_ids])
    debiased_r2, debiased_per_class = _macro_r2(test_truth, debiased_full[test_ids])
    model_metrics = json.loads(args.model_metrics.read_text()) if args.model_metrics.exists() else {}

    report = {
        "method": "Per-class multiplicative count debias fit on development validation",
        "scales": {name: float(scales[index]) for index, name in enumerate(CLASS_NAMES)},
        "R2": debiased_r2,
        "per_class_R2": debiased_per_class,
        "raw_R2": raw_r2,
        "raw_per_class_R2": raw_per_class,
        "delta_R2_vs_raw": debiased_r2 - raw_r2,
        # Counts do not touch masks, so the mPQ+ family is carried over unchanged.
        "mPQ+": model_metrics.get("mPQ+"),
        "mDQ+": model_metrics.get("mDQ+"),
        "mSQ+": model_metrics.get("mSQ+"),
        "counts": str(args.out_counts.relative_to(ROOT)) if args.out_counts.is_relative_to(ROOT) else str(args.out_counts),
    }
    args.out_metrics.write_text(json.dumps(report, indent=2))
    print(f"scales: {report['scales']}")
    print(f"raw test macro R²      = {raw_r2:.4f}")
    print(f"debiased test macro R² = {debiased_r2:.4f}  ({debiased_r2 - raw_r2:+.4f})")
    print(f"mPQ+ carried over      = {report['mPQ+']}")
    print(f"wrote {args.out_counts} and {args.out_metrics}")


if __name__ == "__main__":
    main()
