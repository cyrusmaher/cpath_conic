#!/usr/bin/env python3
"""Run the validation-only final HoVer-Net composition after E47 finishes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def type_blend_eligible(report: dict) -> bool:
    delta = report["selected_delta_vs_control"]
    if delta["mPQ+"] <= 0 or delta["mDQ+"] <= 0 or delta["mSQ+"] < -0.005:
        return False
    if not report["weight_stable_across_source_exclusions"]:
        return False
    selected = report["selected"]
    baseline = next(row for row in report["candidates"] if row["candidate_type_weight"] == 0)
    for source, values in selected["by_source"].items():
        for metric in ("mPQ+", "mDQ+", "mSQ+"):
            if values[metric] - baseline["by_source"][source][metric] < -0.01:
                return False
    return True


def prior_eligible(report: dict) -> bool:
    selected = report["selected"]["mPQ+"]
    oos = report["leave_one_source_out"]["mPQ+"]
    return bool(
        selected["delta_vs_raw"]["mPQ+"] > 0
        and selected["delta_vs_pooled_strength_0"]["mPQ+"] > 0
        and oos["delta_vs_raw"]["mPQ+"] > 0
        and oos["stable_within_one_grid_step"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--weighted-run", type=Path, required=True)
    parser.add_argument("--uniform-checkpoint", type=Path, required=True)
    parser.add_argument("--uniform-artifact", type=Path, required=True)
    parser.add_argument("--uniform-diagnostic", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    summary_path = args.weighted_run / "summary.json"
    while not summary_path.exists():
        time.sleep(args.poll_seconds)
    weighted_checkpoint = args.weighted_run / "best_mpq.pth"
    if not weighted_checkpoint.exists():
        raise FileNotFoundError(weighted_checkpoint)
    args.out_root.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    queue_log = args.out_root / "queue.log"

    weighted_run = args.out_root / "weighted_tta_val"
    weighted_artifact = args.out_root / "weighted_tta_val_predictions.npz"
    weighted_diagnostic = args.out_root / "weighted_tta_val_diagnostic.json"
    if not weighted_diagnostic.exists():
        run(
            [
                python, "scripts/run_hovernet_control.py", "--prepared", str(args.prepared),
                "--checkpoint", str(weighted_checkpoint), "--outdir", str(weighted_run),
                "--split", "val", "--device", args.device, "--batch-size", str(args.batch_size),
                "--flip-tta", "--rotation-tta",
            ],
            queue_log,
        )
        run(
            [
                python, "scripts/materialize_hovernet_validation_artifact.py",
                "--prepared", str(args.prepared), "--run-dir", str(weighted_run),
                "--checkpoint", str(weighted_checkpoint), "--out-artifact", str(weighted_artifact),
                "--out-diagnostic", str(weighted_diagnostic),
            ],
            queue_log,
        )

    prior_report_path = args.out_root / "weighted_sampler_prior.json"
    corrected_artifact = args.out_root / "weighted_tta_val_prior_corrected.npz"
    if not prior_report_path.exists():
        run(
            [
                python, "scripts/sweep_hovernet_sampler_prior.py", "--prepared", str(args.prepared),
                "--diagnostic-json", str(weighted_diagnostic), "--candidate-curve",
                str(args.weighted_run / "training_curve.json"), "--train-ids", str(args.train_ids),
                "--out-report", str(prior_report_path), "--out-corrected-artifact", str(corrected_artifact),
            ],
            queue_log,
        )

    half_run = args.out_root / "half_geometry_tta_val"
    half_artifact = args.out_root / "half_geometry_tta_val_predictions.npz"
    half_diagnostic = args.out_root / "half_geometry_tta_val_diagnostic.json"
    if not half_diagnostic.exists():
        run(
            [
                python, "scripts/run_hovernet_control.py", "--prepared", str(args.prepared),
                "--checkpoint", str(args.uniform_checkpoint), str(weighted_checkpoint),
                "--np-model-weights", "0.5", "0.5", "--hv-model-weights", "0.5", "0.5",
                "--tp-model-weights", "0.5", "0.5", "--outdir", str(half_run),
                "--split", "val", "--device", args.device, "--batch-size", str(args.batch_size),
                "--flip-tta", "--rotation-tta",
            ],
            queue_log,
        )
        run(
            [
                python, "scripts/materialize_hovernet_validation_artifact.py",
                "--prepared", str(args.prepared), "--run-dir", str(half_run),
                "--checkpoint", str(weighted_checkpoint), "--out-artifact", str(half_artifact),
                "--out-diagnostic", str(half_diagnostic),
            ],
            queue_log,
        )

    blend_specs = [
        ("uniform_geometry_weighted_types", args.uniform_artifact, weighted_artifact, "uniform TTA", "weighted TTA"),
        ("uniform_geometry_corrected_weighted_types", args.uniform_artifact, corrected_artifact, "uniform TTA", "prior-corrected weighted TTA"),
        ("weighted_geometry_uniform_types", weighted_artifact, args.uniform_artifact, "weighted TTA", "uniform TTA"),
        ("corrected_weighted_geometry_uniform_types", corrected_artifact, args.uniform_artifact, "prior-corrected weighted TTA", "uniform TTA"),
        ("half_geometry_uniform_types", half_artifact, args.uniform_artifact, "50/50 NP-HV TTA", "uniform TTA"),
        ("half_geometry_weighted_types", half_artifact, weighted_artifact, "50/50 NP-HV TTA", "weighted TTA"),
        ("half_geometry_corrected_weighted_types", half_artifact, corrected_artifact, "50/50 NP-HV TTA", "prior-corrected weighted TTA"),
    ]
    blend_reports = []
    for name, control, candidate, control_name, candidate_name in blend_specs:
        report_path = args.out_root / f"{name}.json"
        if not report_path.exists():
            run(
                [
                    python, "scripts/analyze_hovernet_type_complementarity.py",
                    "--control-artifact", str(control), "--candidate-artifact", str(candidate),
                    "--control-name", control_name, "--candidate-name", candidate_name,
                    "--prepared", str(args.prepared), "--weights", "0", "0.25", "0.5", "0.75", "1",
                    "--out", str(report_path),
                ],
                queue_log,
            )
        blend_reports.append((name, report_path, json.loads(report_path.read_text())))

    uniform = json.loads(args.uniform_diagnostic.read_text())["metrics"]
    weighted = json.loads(weighted_diagnostic.read_text())["metrics"]
    half = json.loads(half_diagnostic.read_text())["metrics"]
    prior = json.loads(prior_report_path.read_text())
    candidates = [
        {"name": "uniform TTA", "eligible": True, "metrics": {k.removeprefix("val_"): v for k, v in uniform.items()}},
        {"name": "weighted TTA", "eligible": True, "metrics": {k.removeprefix("val_"): v for k, v in weighted.items()}},
        {"name": "50/50 NP-HV/TP TTA", "eligible": True, "metrics": {k.removeprefix("val_"): v for k, v in half.items()}},
        {
            "name": "weighted TTA + sampler-prior correction",
            "eligible": prior_eligible(prior),
            "metrics": prior["selected"]["mPQ+"]["metrics"],
        },
    ]
    for name, path, report in blend_reports:
        candidates.append(
            {
                "name": name,
                "eligible": type_blend_eligible(report),
                "metrics": report["selected"]["overall"],
                "report": str(path),
                "candidate_type_weight": report["selected"]["candidate_type_weight"],
            }
        )
    selected = max(
        (candidate for candidate in candidates if candidate["eligible"]),
        key=lambda candidate: candidate["metrics"]["mPQ+"],
    )
    report = {
        "protocol": "validation-only selection among uniform, weighted, 50/50 NP-HV geometry, TP blends, and sampler-prior correction; locked test untouched",
        "evaluation_set": "711-patch source-group-disjoint development validation",
        "selected": selected,
        "candidates": candidates,
        "weighted_checkpoint": str(weighted_checkpoint),
        "test_policy": "Materialize exactly the selected branch weights and correction strength on locked test once.",
    }
    (args.out_root / "final_validation_selection.json").write_text(json.dumps(report, indent=2))
    (args.out_root / "validation_complete.json").write_text(json.dumps({"selected": selected["name"]}, indent=2))


if __name__ == "__main__":
    main()
