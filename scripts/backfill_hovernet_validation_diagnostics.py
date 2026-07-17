#!/usr/bin/env python3
"""Recompute deterministic development-validation diagnostics for an existing checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from scripts.train_hovernet_our_split import (
    HoVerNetExt,
    PreparedHoverDataset,
    initialization_declaration,
    validation_leaderboard_metrics,
    worker_init,
    write_curve,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--val-batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--update-curve",
        action="store_true",
        help="Merge the deterministic metrics into the matching historical curve row.",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint.get("backbone_architecture", "resnet50")
    if checkpoint.get("initialization") != initialization_declaration(architecture):
        raise RuntimeError("checkpoint does not carry the leakage-free generic-ImageNet declaration")
    checkpoint_args = checkpoint.get("args", {})
    backbone = Path(checkpoint_args.get("backbone", ""))
    if not backbone.exists():
        raise FileNotFoundError(f"checkpoint backbone is unavailable: {backbone}")

    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    val_ids = np.load(args.val_ids).astype(np.int32)
    if len(np.unique(val_ids)) != len(val_ids):
        raise RuntimeError("validation IDs contain duplicates")
    forbidden = set(metadata.loc[metadata.split.eq("test"), "patch_id"].astype(int))
    if set(map(int, val_ids)) & forbidden:
        raise RuntimeError("backfill refuses to read locked-test patches")
    val_rows = metadata.loc[metadata.patch_id.isin(val_ids)]
    if len(val_rows) != len(val_ids):
        raise RuntimeError("validation manifest contains unknown patch IDs")

    dataset = PreparedHoverDataset(
        args.prepared,
        val_rows,
        train=False,
        seed=int(checkpoint_args.get("seed", 0)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=worker_init,
        persistent_workers=args.workers > 0,
    )
    device = torch.device(args.device)
    model = HoVerNetExt(
        num_types=7,
        freeze=int(checkpoint.get("phase", 1)) == 1,
        pretrained_backbone=str(backbone),
        backbone_name=architecture,
    ).to(device)
    model.load_state_dict(checkpoint["desc"], strict=True)
    prediction_artifact = args.out_json.with_name(f"{args.out_json.stem}_predictions.npz")
    metrics = validation_leaderboard_metrics(
        model,
        loader,
        args.prepared,
        metadata,
        device,
        amp_dtype=None,
        artifact_path=prediction_artifact,
    )
    payload = {
        "protocol": "deterministic checkpoint backfill; development validation only; locked test refused",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", checkpoint.get("metrics", {}).get("epoch", 0))),
        "checkpoint_phase": int(checkpoint.get("phase", 0)),
        "evaluation_set": f"{len(val_rows)}-patch source-group-disjoint development validation",
        "prediction_artifact": str(prediction_artifact),
        "metrics": metrics,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))

    if args.update_curve:
        run_dir = args.checkpoint.parent
        curve_path = run_dir / "training_curve.json"
        rows = json.loads(curve_path.read_text())
        epoch = payload["checkpoint_epoch"]
        matches = [row for row in rows if int(row.get("epoch", -1)) == epoch]
        if len(matches) != 1:
            raise RuntimeError(f"expected one curve row for epoch {epoch}, found {len(matches)}")
        for key, value in metrics.items():
            prior = matches[0].get(key)
            if isinstance(prior, (int, float)) and isinstance(value, (int, float)):
                if not np.isclose(float(prior), float(value), rtol=1.0e-7, atol=1.0e-9, equal_nan=True):
                    raise RuntimeError(f"backfill changed existing scalar {key}: {prior} -> {value}")
            matches[0][key] = value
        write_curve(run_dir, rows)
        payload["curve_updated"] = str(curve_path)
        args.out_json.write_text(json.dumps(payload, indent=2))

    print(json.dumps({
        "checkpoint": payload["checkpoint"],
        "evaluation_set": payload["evaluation_set"],
        "R2": metrics["val_R2"],
        "mPQ+": metrics["val_mPQ+"],
        "mDQ+": metrics["val_mDQ+"],
        "mSQ+": metrics["val_mSQ+"],
        "curve_updated": payload.get("curve_updated"),
    }, indent=2))


if __name__ == "__main__":
    main()
