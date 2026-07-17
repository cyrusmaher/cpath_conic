#!/usr/bin/env python3
"""Run the remaining validation-only HoVer experiments as one restartable queue.

Every child queue waits for or produces an explicit final JSON marker and skips
completed runs.  Keeping the chain in one parent process prevents independent
waiters from racing for the GPU after a prerequisite appears.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def stage_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    common = [
        "--prepared", str(args.prepared),
        "--train-ids", str(args.train_ids),
        "--val-ids", str(args.val_ids),
        "--poll-seconds", str(args.poll_seconds),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--learning-rates", *map(str, args.learning_rates),
    ]
    intervention_summary = args.intervention_root / "pilot_summary.json"
    e36_control = args.intervention_root / "e36_matched_control_summary.json"
    e37_control = args.intervention_root / "e37_matched_control_summary.json"
    backbone_summary = args.backbone_root / "e41_seresnext101_summary.json"
    instance_summary = args.instance_root / "e42_instance_loss_summary.json"
    instance_type_summary = args.instance_type_root / "e43_instance_type_summary.json"
    type_summary = args.type_root / "e44_type_focal_summary.json"

    return [
        (
            "E36 seed-matched no-H&E control",
            [
                sys.executable, "scripts/run_hovernet_matched_control_queue.py",
                "--wait-for", str(intervention_summary),
                "--backbone", str(args.resnet_backbone),
                "--out-root", str(args.intervention_root),
                "--val-batch-size", str(args.val_batch_size),
                *common,
            ],
        ),
        (
            "E37 seed-matched no-sampling control",
            [
                sys.executable, "scripts/run_hovernet_sampling_control_queue.py",
                "--wait-for", str(e36_control),
                "--backbone", str(args.resnet_backbone),
                "--out-root", str(args.intervention_root),
                "--val-batch-size", str(args.val_batch_size),
                *common,
            ],
        ),
        (
            "E41 heterogeneous SE-ResNeXt-101 backbone",
            [
                sys.executable, "scripts/run_hovernet_backbone_pilot_queue.py",
                "--wait-for", str(e37_control),
                "--backbone", str(args.seresnext_backbone),
                "--backbone-architecture", "seresnext101_32x4d",
                "--out-root", str(args.backbone_root),
                "--val-batch-size", str(args.backbone_val_batch_size),
                *common,
            ],
        ),
        (
            "E42 instance-equalized foreground loss",
            [
                sys.executable, "scripts/run_hovernet_instance_loss_pilot_queue.py",
                "--wait-for", str(backbone_summary),
                "--backbone", str(args.resnet_backbone),
                "--control-summary", str(e36_control),
                "--out-root", str(args.instance_root),
                "--val-batch-size", str(args.val_batch_size),
                *common,
            ],
        ),
        (
            "E43 one-loss-per-nucleus pooled type supervision",
            [
                sys.executable, "scripts/run_hovernet_instance_type_pilot_queue.py",
                "--wait-for", str(instance_summary),
                "--backbone", str(args.resnet_backbone),
                "--control-summary", str(e36_control),
                "--e42-summary", str(instance_summary),
                "--e37-intervention-summary", str(intervention_summary),
                "--e37-control-summary", str(e37_control),
                "--out-root", str(args.instance_type_root),
                "--val-batch-size", str(args.val_batch_size),
                *common,
            ],
        ),
        (
            "E44 type-loss imbalance ablations",
            [
                sys.executable, "scripts/run_hovernet_type_focal_pilot_queue.py",
                "--wait-for", str(instance_type_summary),
                "--backbone", str(args.resnet_backbone),
                "--control-summary", str(e36_control),
                "--out-root", str(args.type_root),
                "--val-batch-size", str(args.val_batch_size),
                *common,
            ],
        ),
        (
            "selected-checkpoint diagnostic backfill and causal audits",
            [
                sys.executable, "scripts/run_hovernet_diagnostic_backfill_queue.py",
                "--wait-for", str(type_summary),
                "--intervention-summary", str(intervention_summary),
                "--prepared", str(args.prepared),
                "--train-ids", str(args.train_ids),
                "--val-ids", str(args.val_ids),
                "--out-root", str(args.backfill_root),
                "--audit-root", str(args.audit_root),
                "--poll-seconds", str(args.poll_seconds),
                "--workers", str(args.workers),
            ],
        ),
        (
            "matched metric-driver analysis",
            [
                sys.executable, "scripts/run_hovernet_analysis_queue.py",
                "--intervention-root", str(args.intervention_root),
                "--backbone-root", str(args.backbone_root),
                "--out-root", str(args.analysis_root),
                "--poll-seconds", str(args.poll_seconds),
            ],
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, required=True)
    parser.add_argument("--resnet-backbone", type=Path, required=True)
    parser.add_argument("--seresnext-backbone", type=Path, required=True)
    parser.add_argument("--intervention-root", type=Path, default=Path("outputs/conic_intervention_pilots"))
    parser.add_argument("--backbone-root", type=Path, default=Path("outputs/conic_backbone_pilots"))
    parser.add_argument("--instance-root", type=Path, default=Path("outputs/conic_instance_loss_pilots"))
    parser.add_argument("--instance-type-root", type=Path, default=Path("outputs/conic_instance_type_pilots"))
    parser.add_argument("--type-root", type=Path, default=Path("outputs/conic_type_loss_pilots"))
    parser.add_argument("--backfill-root", type=Path, default=Path("outputs/conic_validation_diagnostic_backfill"))
    parser.add_argument("--audit-root", type=Path, default=Path("outputs/conic_experiments"))
    parser.add_argument("--analysis-root", type=Path, default=Path("outputs/conic_experiment_audits"))
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--val-batch-size", type=int, default=24)
    parser.add_argument("--backbone-val-batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[3e-5, 1e-4, 3e-4])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for label, command in stage_commands(args):
        print(f"\n=== {label} ===", flush=True)
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
