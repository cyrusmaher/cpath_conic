#!/usr/bin/env python3
"""Run the validation-only E42 instance-equalized-loss pilot after E41."""

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


def scored_rows(run_dir: Path, blend: float, learning_rate: float) -> list[dict]:
    path = run_dir / "training_curve.json"
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    return [
        {
            "instance_loss_blend": blend,
            "learning_rate": learning_rate,
            "epoch": int(row["epoch"]),
            "val_R2": float(row["val_R2"]),
            "val_mPQ+": float(row["val_mPQ+"]),
            "val_mDQ+": float(row["val_mDQ+"]),
            "val_mSQ+": float(row["val_mSQ+"]),
            "run_dir": str(run_dir),
        }
        for row in rows
        if row.get("val_R2") is not None and row.get("val_mPQ+") is not None
    ]


def select(candidates: list[dict], metric: str) -> dict:
    finite = [row for row in candidates if np.isfinite(row[metric])]
    if not finite:
        raise RuntimeError(f"no finite E42 candidates for {metric}")
    return max(finite, key=lambda row: (row[metric], row["epoch"]))


def enrich_selected_components(selection: dict) -> dict:
    """Backfill DQ/SQ from curves made before controls surfaced them directly."""
    if selection.get("val_mDQ+") is not None and selection.get("val_mSQ+") is not None:
        return selection
    rows = json.loads((Path(selection["run_dir"]) / "training_curve.json").read_text())
    row = next(item for item in rows if int(item["epoch"]) == int(selection["epoch"]))
    enriched = dict(selection)
    for output_key, per_class_key in (("val_mDQ+", "val_per_class_DQ"), ("val_mSQ+", "val_per_class_SQ")):
        if enriched.get(output_key) is None:
            value = row.get(output_key)
            if value is None:
                value = np.mean(list(row[per_class_key].values()))
            enriched[output_key] = float(value)
    return enriched


def run_candidate(args: argparse.Namespace, blend: float, learning_rate: float) -> Path:
    blend_label = f"{blend:g}".replace(".", "p")
    lr_label = f"{learning_rate:.0e}".replace("e-0", "e-")
    outdir = args.out_root / f"blend_{blend_label}" / f"lr_{lr_label}"
    if (outdir / "summary.json").exists():
        print(f"E42 pilot complete, skipping: {outdir}", flush=True)
        return outdir
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
        "--instance-loss-blend", str(blend),
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
        raise SystemExit(f"E42 candidate failed with exit code {completed.returncode}")
    return outdir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--val-batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=205)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[3e-5, 1e-4, 3e-4])
    args = parser.parse_args()

    wait_for_json(args.wait_for, args.poll_seconds)
    control = json.loads(args.control_summary.read_text())
    control["selected_R2"] = enrich_selected_components(control["selected_R2"])
    control["selected_mPQ+"] = enrich_selected_components(control["selected_mPQ+"])

    run_dirs: dict[tuple[float, float], Path] = {}
    # Stage A brackets learning rate at a moderate 50% equal-instance blend.
    for learning_rate in args.learning_rates:
        run_dirs[(0.5, learning_rate)] = run_candidate(args, 0.5, learning_rate)
    stage_a = [
        row
        for (blend, learning_rate), run_dir in run_dirs.items()
        for row in scored_rows(run_dir, blend, learning_rate)
    ]
    selected_lrs = {
        float(select(stage_a, "val_R2")["learning_rate"]),
        float(select(stage_a, "val_mPQ+")["learning_rate"]),
    }
    # Stage B tests weaker/full equalization only at endpoint-selected LRs.
    for blend in (0.25, 1.0):
        for learning_rate in sorted(selected_lrs):
            run_dirs[(blend, learning_rate)] = run_candidate(args, blend, learning_rate)

    candidates = [
        row
        for (blend, learning_rate), run_dir in run_dirs.items()
        for row in scored_rows(run_dir, blend, learning_rate)
    ]
    selected_r2 = select(candidates, "val_R2")
    selected_mpq = select(candidates, "val_mPQ+")
    report = {
        "protocol": (
            "Validation-only staged grid. Stage A selects LR independently for R2 and mPQ+ at blend 0.5; "
            "stage B tests blends 0.25 and 1.0 at the union of those LRs. Seed-205 blend-0 controls are "
            "reused from the exact matched no-H&E ResNet-50 pilot. No internal-test inference."
        ),
        "evaluation_set": "711-patch source-group-disjoint development validation",
        "seed": args.seed,
        "runs": {f"blend={blend:g},lr={lr:g}": str(path) for (blend, lr), path in sorted(run_dirs.items())},
        "selected_R2": selected_r2,
        "selected_mPQ+": selected_mpq,
        "matched_control": control,
        "independently_selected_delta": {
            "R2": float(selected_r2["val_R2"] - control["selected_R2"]["val_R2"]),
            "mPQ+": float(selected_mpq["val_mPQ+"] - control["selected_mPQ+"]["val_mPQ+"]),
            "mDQ+": float(selected_mpq["val_mDQ+"] - control["selected_mPQ+"].get("val_mDQ+", np.nan)),
            "mSQ+": float(selected_mpq["val_mSQ+"] - control["selected_mPQ+"].get("val_mSQ+", np.nan)),
        },
        "promotion_guard": (
            "If a selected blend or LR lies on a searched boundary, expand that boundary before a full-horizon run. "
            "Require DQ improvement without SQ, boundary-F1, signed-bias, or outlier-rate regression."
        ),
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    output = args.out_root / "e42_instance_loss_summary.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
