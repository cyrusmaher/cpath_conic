#!/usr/bin/env python3
"""Run the seed-206 no-sampling control required for E37 attribution.

This queue starts only after the seed-205 E36 matched control has completed, so
it cannot contend for the GPU.  It uses the exact E37 split, initialization
seed, LR grid, optimizer-step horizon, batches, and validation cadence while
leaving both source- and class-sampling fractions at zero.
"""
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


def wait_for_json(path: Path, poll_seconds: int) -> None:
    while True:
        if path.exists():
            try:
                json.loads(path.read_text())
                time.sleep(max(10, poll_seconds))
                return
            except (OSError, json.JSONDecodeError):
                pass
        print(f"waiting for completed prerequisite: {path}", flush=True)
        time.sleep(max(1, poll_seconds))


def mean_component(row: dict, direct_key: str, per_class_key: str) -> float | None:
    if row.get(direct_key) is not None:
        return float(row[direct_key])
    values = row.get(per_class_key, {})
    if not values:
        return None
    return float(np.mean(list(values.values())))


def select(run_dirs: dict[float, Path], metric: str) -> dict:
    candidates = []
    for learning_rate, run_dir in run_dirs.items():
        rows = json.loads((run_dir / "training_curve.json").read_text())
        for row in rows:
            value = row.get(metric)
            if value is None or not np.isfinite(value):
                continue
            candidates.append(
                {
                    "learning_rate": learning_rate,
                    "epoch": int(row["epoch"]),
                    metric: float(value),
                    "val_mPQ+": float(row["val_mPQ+"]),
                    "val_mDQ+": mean_component(row, "val_mDQ+", "val_per_class_DQ"),
                    "val_mSQ+": mean_component(row, "val_mSQ+", "val_per_class_SQ"),
                    "val_R2": float(row["val_R2"]),
                    "run_dir": str(run_dir),
                }
            )
    if not candidates:
        raise RuntimeError(f"no finite {metric} values in E37 matched controls")
    return max(candidates, key=lambda row: (row[metric], row["epoch"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--val-batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=206)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[3e-5, 1e-4, 3e-4])
    args = parser.parse_args()

    wait_for_json(args.wait_for, args.poll_seconds)
    run_dirs: dict[float, Path] = {}
    for learning_rate in args.learning_rates:
        label = f"{learning_rate:.0e}".replace("e-0", "e-")
        outdir = args.out_root / "e37_no_sampling_seed206" / f"lr_{label}"
        run_dirs[learning_rate] = outdir
        if (outdir / "summary.json").exists():
            print(f"control complete, skipping: {outdir}", flush=True)
            continue
        archived = archive_incomplete_persistent_run(outdir)
        if archived is not None:
            print(f"archived incomplete persistent-worker run: {archived}", flush=True)
        command = [
            sys.executable,
            "scripts/train_hovernet_our_split.py",
            "--prepared", str(args.prepared),
            "--backbone", str(args.backbone),
            "--train-ids", str(args.train_ids),
            "--val-ids", str(args.val_ids),
            "--outdir", str(outdir),
            "--learning-rate", str(learning_rate),
            "--epochs-phase1", str(args.epochs),
            "--epochs-phase2", "0",
            "--batch-size", str(args.batch_size),
            "--val-batch-size", str(args.val_batch_size),
            "--workers", str(args.workers),
            "--metric-every", "5",
            "--seed", str(args.seed),
            "--amp", "none",
            "--device", "cuda:0",
        ]
        print("running:", " ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise SystemExit(f"E37 control failed with exit code {completed.returncode}")

    report = {
        "policy": (
            "No-sampling control matched to E37 on split, seed 206, LR grid, epoch/step horizon, "
            "batch sizes, validation cadence, and ImageNet initialization. Internal test is never evaluated."
        ),
        "seed": args.seed,
        "runs": {str(rate): str(path) for rate, path in run_dirs.items()},
        "selected_mPQ+": select(run_dirs, "val_mPQ+"),
        "selected_R2": select(run_dirs, "val_R2"),
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    output = args.out_root / "e37_matched_control_summary.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
