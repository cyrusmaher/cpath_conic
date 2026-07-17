#!/usr/bin/env python
"""Materialize a source-routed fixed-mask prediction without duplicate intermediates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def routed_patch_ids(metadata: pd.DataFrame, route: dict[str, str], method: str) -> np.ndarray:
    return metadata.loc[metadata.source.map(route) == method, "patch_id"].to_numpy(dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--route-report", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--base-probabilities", type=Path, required=True)
    parser.add_argument("--alternate-instance-maps", type=Path, required=True)
    parser.add_argument("--alternate-features", type=Path, required=True)
    parser.add_argument("--alternate-probabilities", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    report = json.loads(args.route_report.read_text())
    route = report["selected_route"]
    alternate_ids = set(routed_patch_ids(metadata, route, "b").tolist())

    base_predictions = np.load(args.base_predictions, mmap_mode="r")
    alternate_maps = np.load(args.alternate_instance_maps, mmap_mode="r")
    args.outdir.mkdir(parents=True, exist_ok=True)
    output_path = args.outdir / "predictions.npy"
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.int32, shape=base_predictions.shape)
    for start in range(0, len(base_predictions), 128):
        stop = min(start + 128, len(base_predictions))
        output[start:stop] = base_predictions[start:stop]

    alternate_features = np.load(args.alternate_features)
    alternate_probability_data = np.load(args.alternate_probabilities)
    alternate_patch_ids = alternate_features["patch_ids"].astype(np.int32)
    alternate_instance_ids = alternate_features["instance_ids"].astype(np.int32)
    if not np.array_equal(alternate_patch_ids, alternate_probability_data["patch_ids"]) or not np.array_equal(alternate_instance_ids, alternate_probability_data["instance_ids"]):
        raise ValueError("Alternate feature and probability IDs are not aligned")
    alternate_probs = alternate_probability_data["class_probs"].astype(np.float32)
    alternate_assignments = alternate_probs.argmax(axis=1).astype(np.int8) + 1
    for patch_id in sorted(alternate_ids):
        rows = alternate_patch_ids == patch_id
        instances = np.asarray(alternate_maps[patch_id])
        class_lut = np.zeros(int(instances.max()) + 1, dtype=np.int8)
        class_lut[alternate_instance_ids[rows]] = alternate_assignments[rows]
        output[patch_id, ..., 0] = instances
        output[patch_id, ..., 1] = class_lut[instances]
    output.flush()

    base_probability_data = np.load(args.base_probabilities)
    base_patch_ids = base_probability_data["patch_ids"].astype(np.int32)
    base_probs = base_probability_data["class_probs"].astype(np.float32)
    base_assignments = (
        base_probability_data["assignments"].astype(np.int8)
        if "assignments" in base_probability_data.files
        else base_probs.argmax(axis=1).astype(np.int8) + 1
    )
    base_keep = ~np.isin(base_patch_ids, np.asarray(sorted(alternate_ids), dtype=np.int32))
    alternate_keep = np.isin(alternate_patch_ids, np.asarray(sorted(alternate_ids), dtype=np.int32))
    hybrid_patch_ids = np.concatenate([base_patch_ids[base_keep], alternate_patch_ids[alternate_keep]])
    hybrid_instance_ids = np.concatenate([
        base_probability_data["instance_ids"].astype(np.int32)[base_keep],
        alternate_instance_ids[alternate_keep],
    ])
    hybrid_probs = np.concatenate([
        base_probs[base_keep],
        alternate_probs[alternate_keep],
    ])
    hybrid_assignments = np.concatenate([
        base_assignments[base_keep],
        alternate_assignments[alternate_keep],
    ])
    order = np.lexsort((hybrid_instance_ids, hybrid_patch_ids))
    np.savez_compressed(
        args.outdir / "cell_probabilities.npz",
        patch_ids=hybrid_patch_ids[order],
        instance_ids=hybrid_instance_ids[order],
        class_probs=hybrid_probs[order],
        assignments=hybrid_assignments[order],
        class_names=(
            base_probability_data["class_names"]
            if "class_names" in base_probability_data.files
            else alternate_probability_data["class_names"]
        ),
    )
    (args.outdir / "routing.json").write_text(json.dumps({
        "route": route,
        "alternate_patch_count": len(alternate_ids),
        "base_patch_count": int(len(metadata) - len(alternate_ids)),
        "prediction_shape": list(base_predictions.shape),
        "probability_records": int(len(order)),
    }, indent=2))


if __name__ == "__main__":
    main()
