#!/usr/bin/env python3
"""Run remaining HoVer-Net development folds sequentially and resumably."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--wait-for", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--val-batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--metric-every", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=105)
    args = parser.parse_args()

    if args.wait_for is not None:
        while not args.wait_for.exists():
            print(f"waiting for {args.wait_for}", flush=True)
            time.sleep(max(1, args.poll_seconds))

    for fold in args.folds:
        outdir = args.run_root / f"fold_{fold}_phase1_lr1e-4"
        summary = outdir / "summary.json"
        if summary.exists():
            print(f"fold {fold}: complete, skipping", flush=True)
            continue
        command = [
            sys.executable,
            "scripts/train_hovernet_our_split.py",
            "--prepared", str(args.prepared),
            "--backbone", str(args.backbone),
            "--train-ids", str(args.fold_dir / f"fold_{fold}_train_ids.npy"),
            "--val-ids", str(args.fold_dir / f"fold_{fold}_val_ids.npy"),
            "--outdir", str(outdir),
            "--learning-rate", str(args.learning_rate),
            "--epochs-phase1", str(args.epochs),
            "--epochs-phase2", "0",
            "--batch-size", str(args.batch_size),
            "--val-batch-size", str(args.val_batch_size),
            "--workers", str(args.workers),
            "--metric-every", str(args.metric_every),
            "--seed", str(args.seed_base + fold),
            "--amp", "none",
            "--device", "cuda:0",
        ]
        latest = outdir / "latest.pth"
        if latest.exists():
            command.extend(["--resume-checkpoint", str(latest)])
        print(f"fold {fold}: {' '.join(command)}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise SystemExit(f"fold {fold} failed with exit code {completed.returncode}")


if __name__ == "__main__":
    main()
