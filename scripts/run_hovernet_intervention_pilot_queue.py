#!/usr/bin/env python3
"""Wait for post-fold jobs, then run matched E36/E37 learning-rate pilots."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.queue_integrity import archive_incomplete_persistent_run


def run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(f"command failed with exit code {completed.returncode}: {' '.join(command)}")


def selected(run_dirs: dict[float, Path], metric: str) -> dict:
    candidates = []
    for learning_rate, run_dir in run_dirs.items():
        rows = json.loads((run_dir / "training_curve.json").read_text())
        for row in rows:
            value = row.get(metric)
            if value is not None and np.isfinite(value):
                candidates.append(
                    {
                        "learning_rate": learning_rate,
                        "epoch": int(row["epoch"]),
                        metric: float(value),
                        "val_mPQ+": float(row["val_mPQ+"]),
                        "val_R2": float(row["val_R2"]),
                        "run_dir": str(run_dir),
                    }
                )
    if not candidates:
        raise RuntimeError(f"No finite {metric} values in intervention pilots")
    return max(candidates, key=lambda row: (row[metric], row["epoch"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, required=True)
    parser.add_argument("--hed-profile", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--val-batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[3e-5, 1e-4, 3e-4])
    args = parser.parse_args()

    while not args.wait_for.exists():
        print(f"waiting for {args.wait_for}", flush=True)
        time.sleep(max(1, args.poll_seconds))

    families = {
        "e36_empirical_hed": {
            "seed": 205,
            "extra": [
                "--hed-probability",
                "0.8",
                "--hed-profile",
                str(args.hed_profile),
                "--hed-target-jitter",
                "0.05",
                "--hed-tail-expansion",
                "0.1",
                "--hed-strength-min",
                "0.25",
                "--hed-strength-max",
                "1.0",
            ],
        },
        "e37_source_0.5": {
            "seed": 206,
            "extra": ["--source-sampling-fraction", "0.5"],
        },
        "e37_class_0.5": {
            "seed": 206,
            "extra": ["--class-sampling-fraction", "0.5"],
        },
    }
    report = {
        "policy": (
            "Matched 10-epoch group-disjoint pilots with identical seeds within each family. "
            "mPQ+ and R2 select learning rate/checkpoint independently; no test inference is run. "
            "Full-horizon promotion is deliberately not automatic."
        ),
        "families": {},
    }
    for family, specification in families.items():
        run_dirs = {}
        for learning_rate in args.learning_rates:
            label = f"{learning_rate:.0e}".replace("e-0", "e-")
            outdir = args.out_root / family / f"lr_{label}"
            run_dirs[learning_rate] = outdir
            if (outdir / "summary.json").exists():
                print(f"pilot complete, skipping: {outdir}", flush=True)
                continue
            archived = archive_incomplete_persistent_run(outdir)
            if archived is not None:
                print(f"archived incomplete persistent-worker run: {archived}", flush=True)
            command = [
                sys.executable,
                "scripts/train_hovernet_our_split.py",
                "--prepared",
                str(args.prepared),
                "--backbone",
                str(args.backbone),
                "--train-ids",
                str(args.train_ids),
                "--val-ids",
                str(args.val_ids),
                "--outdir",
                str(outdir),
                "--learning-rate",
                str(learning_rate),
                "--epochs-phase1",
                str(args.epochs),
                "--epochs-phase2",
                "0",
                "--batch-size",
                str(args.batch_size),
                "--val-batch-size",
                str(args.val_batch_size),
                "--workers",
                str(args.workers),
                "--metric-every",
                "5",
                "--seed",
                str(specification["seed"]),
                "--amp",
                "none",
                "--device",
                "cuda:0",
                *specification["extra"],
            ]
            run(command)
        report["families"][family] = {
            "runs": {str(rate): str(path) for rate, path in run_dirs.items()},
            "selected_mPQ+": selected(run_dirs, "val_mPQ+"),
            "selected_R2": selected(run_dirs, "val_R2"),
        }
        args.out_root.mkdir(parents=True, exist_ok=True)
        (args.out_root / "pilot_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
