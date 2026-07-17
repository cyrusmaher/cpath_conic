#!/usr/bin/env python3
"""Run a leakage-free heterogeneous HoVer-Net backbone LR pilot."""
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


def select(run_dirs: dict[float, Path], metric: str) -> dict:
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
        raise RuntimeError(f"no finite {metric} values in backbone pilots")
    return max(candidates, key=lambda row: (row[metric], row["epoch"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--backbone-architecture", required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--family", default="e41_seresnext101")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--val-batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=205)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[3e-5, 1e-4, 3e-4])
    args = parser.parse_args()

    wait_for_json(args.wait_for, args.poll_seconds)
    run_dirs: dict[float, Path] = {}
    for learning_rate in args.learning_rates:
        label = f"{learning_rate:.0e}".replace("e-0", "e-")
        outdir = args.out_root / args.family / f"lr_{label}"
        run_dirs[learning_rate] = outdir
        if (outdir / "summary.json").exists():
            print(f"backbone pilot complete, skipping: {outdir}", flush=True)
            continue
        archived = archive_incomplete_persistent_run(outdir)
        if archived is not None:
            print(f"archived incomplete persistent-worker run: {archived}", flush=True)
        command = [
            sys.executable,
            "scripts/train_hovernet_our_split.py",
            "--prepared", str(args.prepared),
            "--backbone", str(args.backbone),
            "--backbone-architecture", args.backbone_architecture,
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
            raise SystemExit(f"backbone pilot failed with exit code {completed.returncode}")

    report = {
        "policy": (
            "Heterogeneous-backbone pilot matched to the seed-205 ResNet control on split, LR grid, "
            "epoch horizon, train batch size, optimizer-step budget, validation cadence, and generic "
            "ImageNet-only initialization. Internal test is never evaluated."
        ),
        "backbone_architecture": args.backbone_architecture,
        "backbone": str(args.backbone),
        "seed": args.seed,
        "runs": {str(rate): str(path) for rate, path in run_dirs.items()},
        "selected_mPQ+": select(run_dirs, "val_mPQ+"),
        "selected_R2": select(run_dirs, "val_R2"),
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    output = args.out_root / f"{args.family}_summary.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
