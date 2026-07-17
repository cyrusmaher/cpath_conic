#!/usr/bin/env python3
"""Compact a saved HoVer-Net validation run for offline fusion diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpath_conic.constants import COUNT_COLUMNS
from scripts.analyze_hovernet_type_complementarity import metric_summary


def selected_validation_ids(prepared: Path) -> tuple[pd.DataFrame, np.ndarray]:
    metadata = pd.read_csv(prepared / "metadata.csv").sort_values("patch_id")
    selected = metadata.loc[metadata.split.eq("val")].copy()
    if selected.empty:
        raise RuntimeError("prepared metadata has no development-validation rows")
    return metadata, selected.patch_id.to_numpy(dtype=np.int32)


def checkpoint_epoch(checkpoint: Path) -> int:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return int(payload.get("epoch", payload.get("metrics", {}).get("epoch", 0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-artifact", type=Path, required=True)
    parser.add_argument("--out-diagnostic", type=Path, required=True)
    args = parser.parse_args()

    metadata, patch_ids = selected_validation_ids(args.prepared)
    forbidden = set(metadata.loc[metadata.split.eq("test"), "patch_id"].astype(int))
    if set(map(int, patch_ids)) & forbidden:
        raise RuntimeError("validation artifact materialization refuses locked-test patches")

    prediction_path = args.run_dir / "predictions.npy"
    count_path = args.run_dir / "counts.npy"
    probability_path = args.run_dir / "cell_probabilities.npz"
    for path in (prediction_path, count_path, probability_path):
        if not path.exists():
            raise FileNotFoundError(path)
    predictions_full = np.load(prediction_path, mmap_mode="r")
    counts_full = np.load(count_path, mmap_mode="r")
    if int(patch_ids.max()) >= len(predictions_full) or int(patch_ids.max()) >= len(counts_full):
        raise RuntimeError("saved run does not cover every validation patch ID")
    predictions = np.asarray(predictions_full[patch_ids], dtype=np.int32)
    predicted_counts = np.asarray(counts_full[patch_ids], dtype=np.int32)

    probability = np.load(probability_path)
    probability_patch_ids = probability["patch_ids"].astype(np.int32)
    probability_instance_ids = probability["instance_ids"].astype(np.int32)
    class_probs = probability["class_probs"].astype(np.float32)
    if not set(map(int, probability_patch_ids)).issubset(set(map(int, patch_ids))):
        raise RuntimeError("saved probabilities contain patches outside development validation")

    truth = np.zeros_like(predictions, dtype=np.int32)
    for index, patch_id in enumerate(patch_ids):
        label = np.load(args.prepared / "labels" / f"{int(patch_id):05d}.npy", allow_pickle=True).item()
        truth[index, ..., 0] = label["inst_map"]
        truth[index, ..., 1] = label["class_map"]
    val_rows = metadata.set_index("patch_id").loc[patch_ids]
    true_counts = val_rows[COUNT_COLUMNS].to_numpy(dtype=np.int32)
    metrics = metric_summary(truth, predictions, true_counts)

    args.out_artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_artifact,
        patch_ids=patch_ids,
        predictions=predictions,
        predicted_counts=predicted_counts,
        probability_patch_ids=probability_patch_ids,
        probability_instance_ids=probability_instance_ids,
        class_probs=class_probs,
    )
    epoch = checkpoint_epoch(args.checkpoint)
    diagnostic = {
        "protocol": "compact saved-run artifact; development validation only; locked test refused",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": epoch,
        "evaluation_set": f"{len(patch_ids)}-patch source-group-disjoint development validation",
        "prediction_artifact": str(args.out_artifact),
        "metrics": {
            "val_R2": metrics["R2"],
            "val_mPQ+": metrics["mPQ+"],
            "val_mDQ+": metrics["mDQ+"],
            "val_mSQ+": metrics["mSQ+"],
        },
    }
    args.out_diagnostic.parent.mkdir(parents=True, exist_ok=True)
    args.out_diagnostic.write_text(json.dumps(diagnostic, indent=2))
    (args.run_dir / "metrics_val.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    main()
