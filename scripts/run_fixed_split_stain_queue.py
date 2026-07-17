#!/usr/bin/env python3
"""Run and diagnose the two fixed-split E38 stain-TTA gates."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_postfold_hovernet_queue import metric_brief, run, run_inference_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hed-report", type=Path, required=True)
    parser.add_argument("--native-val-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    anchors = json.loads(args.hed_report.read_text())["candidate_anchors"]
    jobs = [
        ("e38_two_view_h3e2_mpq_val", anchors["best_mPQ+"], "best_validation_mpq_H3E2"),
        ("e38_two_view_h3e3_r2_val", anchors["best_within_bin_R2"], "best_validation_r2_H3E3"),
    ]
    for name, anchor, label in jobs:
        concentration = anchor["concentration"]
        run_inference_and_evaluate(
            args.prepared,
            [args.checkpoint],
            args.out_root / name,
            "val",
            args.batch_size,
            [
                "--stain-target",
                str(concentration[0]),
                str(concentration[1]),
                "--stain-target-label",
                label,
                "--stain-anchor-patch-id",
                str(anchor["patch_id"]),
            ],
        )

    comparisons = []
    for name, _, _ in jobs:
        output = args.out_root / f"{name}_vs_native_bootstrap.json"
        comparisons.append(output)
        if output.exists():
            continue
        run(
            [
                sys.executable,
                "scripts/bootstrap_prediction_difference.py",
                "--prepared",
                str(args.prepared),
                "--predictions-a",
                str(args.native_val_dir / "predictions.npy"),
                "--counts-a",
                str(args.native_val_dir / "counts.npy"),
                "--predictions-b",
                str(args.out_root / name / "predictions.npy"),
                "--counts-b",
                str(args.out_root / name / "counts.npy"),
                "--name-a",
                "Native HoVer-Net",
                "--name-b",
                name,
                "--split",
                "val",
                "--replicates",
                "2000",
                "--out",
                str(output),
            ]
        )

    tail_reports = []
    for name, _, _ in jobs:
        output = args.out_root / f"{name}_count_drivers.json"
        tail_reports.append(output)
        if output.exists():
            continue
        run(
            [
                sys.executable,
                "scripts/analyze_count_tails.py",
                "--prepared",
                str(args.prepared),
                "--counts",
                str(args.out_root / name / "counts.npy"),
                "--split",
                "val",
                "--target-r2",
                "0.8585",
                "--out",
                str(output),
            ]
        )

    summary = {
        "selection_policy": "Two training-defined, validation-selected H/E anchors; native+styled raw-map average; no test access.",
        "results": {
            name: metric_brief(args.out_root / name / "metrics_val.json") for name, _, _ in jobs
        },
        "paired_comparisons": [str(path) for path in comparisons],
        "count_driver_reports": [str(path) for path in tail_reports],
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "e38_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
