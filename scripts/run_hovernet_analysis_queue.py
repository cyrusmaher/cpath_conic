#!/usr/bin/env python3
"""Generate matched DQ/SQ-aware audits as serialized HoVer pilots finish."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def wait_for_json(path: Path, poll_seconds: int) -> None:
    while True:
        if path.exists():
            try:
                json.loads(path.read_text())
                return
            except (OSError, json.JSONDecodeError):
                pass
        print(f"waiting for completed prerequisite: {path}", flush=True)
        time.sleep(max(1, poll_seconds))


def analyze(
    candidate_root: Path,
    control_root: Path,
    candidate_label: str,
    control_label: str,
    output_stem: Path,
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/analyze_hovernet_pilot_pair.py",
        "--candidate-root", str(candidate_root),
        "--control-root", str(control_root),
        "--candidate-label", candidate_label,
        "--control-label", control_label,
        "--out-json", str(output_stem.with_suffix(".json")),
        "--out-plot", str(output_stem.with_suffix(".png")),
    ]
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervention-root", type=Path, default=Path("outputs/conic_intervention_pilots"))
    parser.add_argument("--backbone-root", type=Path, default=Path("outputs/conic_backbone_pilots"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/conic_experiment_audits"))
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    e36_summary = args.intervention_root / "e36_matched_control_summary.json"
    wait_for_json(e36_summary, args.poll_seconds)
    analyze(
        args.intervention_root / "e36_empirical_hed",
        args.intervention_root / "e36_no_hed_seed205",
        "empirical H/E",
        "no H/E, seed 205",
        args.out_root / "e36_hed_vs_matched_control",
    )

    e37_summary = args.intervention_root / "e37_matched_control_summary.json"
    wait_for_json(e37_summary, args.poll_seconds)
    for family, label in (
        ("e37_source_0.5", "source-balanced 0.5"),
        ("e37_class_0.5", "class-balanced 0.5"),
    ):
        analyze(
            args.intervention_root / family,
            args.intervention_root / "e37_no_sampling_seed206",
            label,
            "uniform sampling, seed 206",
            args.out_root / f"{family}_vs_matched_control",
        )

    e41_summary = args.backbone_root / "e41_seresnext101_summary.json"
    wait_for_json(e41_summary, args.poll_seconds)
    analyze(
        args.backbone_root / "e41_seresnext101",
        args.intervention_root / "e36_no_hed_seed205",
        "SE-ResNeXt-101",
        "ResNet-50, seed 205",
        args.out_root / "e41_seresnext101_vs_resnet50",
    )
    print("all matched pilot audits complete", flush=True)


if __name__ == "__main__":
    main()
