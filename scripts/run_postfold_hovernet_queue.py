#!/usr/bin/env python3
"""Resumably run precommitted E38 and E39 jobs after the six folds finish."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(f"command failed with exit code {completed.returncode}: {' '.join(command)}")


def run_inference_and_evaluate(
    prepared: Path,
    checkpoints: list[Path],
    outdir: Path,
    split: str,
    batch_size: int,
    extra_inference_args: list[str],
) -> None:
    run_record = outdir / "run.json"
    metrics = outdir / f"metrics_{split}.json"
    if not run_record.exists():
        run(
            [
                sys.executable,
                "scripts/run_hovernet_control.py",
                "--prepared",
                str(prepared),
                "--checkpoint",
                *[str(path) for path in checkpoints],
                "--outdir",
                str(outdir),
                "--split",
                split,
                "--device",
                "cuda:0",
                "--batch-size",
                str(batch_size),
                *extra_inference_args,
            ]
        )
    else:
        print(f"inference complete, skipping: {outdir}", flush=True)
    if not metrics.exists():
        run(
            [
                sys.executable,
                "scripts/evaluate_conic.py",
                "--prepared",
                str(prepared),
                "--predictions",
                str(outdir / "predictions.npy"),
                "--counts",
                str(outdir / "counts.npy"),
                "--split",
                split,
                "--out",
                str(metrics),
            ]
        )
    else:
        print(f"metrics complete, skipping: {metrics}", flush=True)


def metric_brief(path: Path) -> dict:
    values = json.loads(path.read_text())
    return {
        "path": str(path),
        "mPQ+": values.get("mPQ+"),
        "R2": values.get("R2"),
        "per_class_pq": {
            name: metrics.get("pq") for name, metrics in values.get("per_class_pq", {}).items()
        },
        "per_class_R2": values.get("per_class_R2", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--fold0-root", type=Path, required=True)
    parser.add_argument("--hed-report", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--native-val-dir", type=Path, required=True)
    parser.add_argument("--reference-test-predictions", type=Path, required=True)
    parser.add_argument("--reference-mpq-counts", type=Path, required=True)
    parser.add_argument("--reference-r2-counts", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    summaries = [args.fold0_root / "summary.json"] + [
        args.fold_root / f"fold_{fold}_phase1_lr1e-4" / "summary.json" for fold in range(1, 6)
    ]
    while not all(path.exists() for path in summaries):
        missing = [str(path) for path in summaries if not path.exists()]
        print(f"waiting for {len(missing)} fold summaries; first missing: {missing[0]}", flush=True)
        time.sleep(max(1, args.poll_seconds))

    fold0_mpq = args.fold0_root / "best_mpq.pth"
    fold0_r2 = args.fold0_root / "best_r2.pth"
    mpq_checkpoints = [fold0_mpq] + [
        args.fold_root / f"fold_{fold}_phase1_lr1e-4" / "best_mpq.pth" for fold in range(1, 6)
    ]
    r2_checkpoints = [fold0_r2] + [
        args.fold_root / f"fold_{fold}_phase1_lr1e-4" / "best_r2.pth" for fold in range(1, 6)
    ]
    missing_checkpoints = [path for path in [*mpq_checkpoints, *r2_checkpoints] if not path.exists()]
    if missing_checkpoints:
        raise FileNotFoundError(f"fold summaries exist but checkpoints are missing: {missing_checkpoints}")

    stain_report = json.loads(args.hed_report.read_text())
    anchors = stain_report["candidate_anchors"]
    stain_jobs = [
        (
            "e38_two_view_h3e2_mpq_val",
            fold0_mpq,
            anchors["best_mPQ+"],
            "best_validation_mpq_H3E2",
        ),
        (
            "e38_two_view_h3e3_r2_val",
            fold0_r2,
            anchors["best_within_bin_R2"],
            "best_validation_r2_H3E3",
        ),
    ]
    for name, checkpoint, anchor, label in stain_jobs:
        concentration = anchor["concentration"]
        run_inference_and_evaluate(
            args.prepared,
            [checkpoint],
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

    ensemble_jobs = [
        ("e39_sixfold_best_mpq_tta_test", mpq_checkpoints),
        ("e39_sixfold_best_r2_tta_test", r2_checkpoints),
    ]
    for name, checkpoints in ensemble_jobs:
        run_inference_and_evaluate(
            args.prepared,
            checkpoints,
            args.out_root / name,
            "test",
            args.batch_size,
            ["--flip-tta", "--rotation-tta"],
        )

    comparisons = [
        (
            "e38_h3e2_vs_native_bootstrap.json",
            "Native HoVer-Net",
            args.native_val_dir / "predictions.npy",
            args.native_val_dir / "counts.npy",
            "E38 native + H3/E2",
            args.out_root / "e38_two_view_h3e2_mpq_val" / "predictions.npy",
            args.out_root / "e38_two_view_h3e2_mpq_val" / "counts.npy",
            "val",
        ),
        (
            "e38_h3e3_vs_native_bootstrap.json",
            "Native HoVer-Net",
            args.native_val_dir / "predictions.npy",
            args.native_val_dir / "counts.npy",
            "E38 native + H3/E3",
            args.out_root / "e38_two_view_h3e3_r2_val" / "predictions.npy",
            args.out_root / "e38_two_view_h3e3_r2_val" / "counts.npy",
            "val",
        ),
        (
            "e39_mpq_vs_e32_bootstrap.json",
            "E32 single-model spatial TTA",
            args.reference_test_predictions,
            args.reference_mpq_counts,
            "E39 sixfold best-mPQ spatial TTA",
            args.out_root / "e39_sixfold_best_mpq_tta_test" / "predictions.npy",
            args.out_root / "e39_sixfold_best_mpq_tta_test" / "counts.npy",
            "test",
        ),
        (
            "e39_r2_vs_e33_bootstrap.json",
            "E33 HoVer-Net/CellViT count blend",
            args.reference_test_predictions,
            args.reference_r2_counts,
            "E39 sixfold best-R2 spatial TTA",
            args.out_root / "e39_sixfold_best_r2_tta_test" / "predictions.npy",
            args.out_root / "e39_sixfold_best_r2_tta_test" / "counts.npy",
            "test",
        ),
    ]
    for filename, name_a, predictions_a, counts_a, name_b, predictions_b, counts_b, split in comparisons:
        output = args.out_root / filename
        if output.exists():
            print(f"bootstrap complete, skipping: {output}", flush=True)
            continue
        run(
            [
                sys.executable,
                "scripts/bootstrap_prediction_difference.py",
                "--prepared",
                str(args.prepared),
                "--predictions-a",
                str(predictions_a),
                "--counts-a",
                str(counts_a),
                "--predictions-b",
                str(predictions_b),
                "--counts-b",
                str(counts_b),
                "--name-a",
                name_a,
                "--name-b",
                name_b,
                "--split",
                split,
                "--replicates",
                "2000",
                "--out",
                str(output),
            ]
        )

    tail_jobs = [
        ("e38_h3e2_count_drivers.json", "val", args.out_root / "e38_two_view_h3e2_mpq_val" / "counts.npy"),
        ("e38_h3e3_count_drivers.json", "val", args.out_root / "e38_two_view_h3e3_r2_val" / "counts.npy"),
        ("e39_mpq_count_drivers.json", "test", args.out_root / "e39_sixfold_best_mpq_tta_test" / "counts.npy"),
        ("e39_r2_count_drivers.json", "test", args.out_root / "e39_sixfold_best_r2_tta_test" / "counts.npy"),
    ]
    for filename, split, counts in tail_jobs:
        output = args.out_root / filename
        if output.exists():
            print(f"count-driver report complete, skipping: {output}", flush=True)
            continue
        run(
            [
                sys.executable,
                "scripts/analyze_count_tails.py",
                "--prepared",
                str(args.prepared),
                "--counts",
                str(counts),
                "--split",
                split,
                "--target-r2",
                "0.8585",
                "--out",
                str(output),
            ]
        )

    paths = {
        name: args.out_root / name / f"metrics_{'val' if name.startswith('e38') else 'test'}.json"
        for name, *_ in [*stain_jobs, *ensemble_jobs]
    }
    summary = {
        "selection_policy": (
            "E38 anchors were selected from training-defined H/E bins on validation. E39 consists of exactly two "
            "precommitted uniform six-model recipes: foldwise best-mPQ checkpoints and foldwise best-R2 checkpoints. "
            "No ensemble subset or weight was selected on original validation or test."
        ),
        "results": {name: metric_brief(path) for name, path in paths.items()},
        "paired_comparisons": {filename.removesuffix(".json"): str(args.out_root / filename) for filename, *_ in comparisons},
        "count_driver_reports": {filename.removesuffix(".json"): str(args.out_root / filename) for filename, *_ in tail_jobs},
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "postfold_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
