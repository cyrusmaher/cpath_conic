#!/usr/bin/env python3
"""Materialize the frozen validation-selected HoVer-Net recipe on locked test."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hovernet_type_complementarity import (
    blended_class_lookup_for_patch,
    central_counts_from_lookup,
    decoded_class_lookup,
    match_instances,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--weighted-run", type=Path, required=True)
    parser.add_argument("--uniform-run", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    selected = selection.get("selected", {})
    expected_name = "weighted_geometry_uniform_types"
    if selected.get("name") != expected_name or not selected.get("eligible"):
        raise RuntimeError(f"frozen selection must be eligible {expected_name!r}: {selected.get('name')!r}")
    weight = float(selected.get("candidate_type_weight", -1))
    if not np.isclose(weight, 0.75, rtol=0, atol=1.0e-12):
        raise RuntimeError(f"unexpected frozen uniform TP weight: {weight}")
    if "locked test untouched" not in selection.get("protocol", ""):
        raise RuntimeError("selection manifest does not declare locked-test isolation")

    output_prediction = args.outdir / "predictions.npy"
    output_counts = args.outdir / "counts.npy"
    output_probabilities = args.outdir / "cell_probabilities.npz"
    output_manifest = args.outdir / "run.json"
    if output_manifest.exists():
        raise FileExistsError("completed final locked-test materialization already exists; refusing to overwrite")

    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    test_ids = metadata.loc[metadata.split.eq("test"), "patch_id"].to_numpy(dtype=np.int32)
    if len(test_ids) != 657:
        raise RuntimeError(f"expected 657 locked-test patches, found {len(test_ids)}")

    weighted_predictions = np.load(args.weighted_run / "predictions.npy", mmap_mode="r")
    uniform_predictions = np.load(args.uniform_run / "predictions.npy", mmap_mode="r")
    if weighted_predictions.shape != uniform_predictions.shape or weighted_predictions.ndim != 4:
        raise ValueError("weighted and uniform prediction arrays do not align")
    weighted_probability_artifact = np.load(args.weighted_run / "cell_probabilities.npz")
    uniform_probability_artifact = np.load(args.uniform_run / "cell_probabilities.npz")
    weighted_probabilities = {
        (int(patch_id), int(instance_id)): np.asarray(probability, dtype=np.float32)
        for patch_id, instance_id, probability in zip(
            weighted_probability_artifact["patch_ids"],
            weighted_probability_artifact["instance_ids"],
            weighted_probability_artifact["class_probs"],
            strict=True,
        )
    }
    uniform_probabilities = {
        (int(patch_id), int(instance_id)): np.asarray(probability, dtype=np.float32)
        for patch_id, instance_id, probability in zip(
            uniform_probability_artifact["patch_ids"],
            uniform_probability_artifact["instance_ids"],
            uniform_probability_artifact["class_probs"],
            strict=True,
        )
    }
    if not set(patch_id for patch_id, _ in weighted_probabilities).issubset(set(map(int, test_ids))):
        raise RuntimeError("weighted probability artifact contains non-test patches")
    if not set(patch_id for patch_id, _ in uniform_probabilities).issubset(set(map(int, test_ids))):
        raise RuntimeError("uniform probability artifact contains non-test patches")

    args.outdir.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_prediction,
        mode="w+",
        dtype=np.int32,
        shape=weighted_predictions.shape,
    )
    counts = np.zeros((len(metadata), 6), dtype=np.int32)
    probability_patch_ids = []
    probability_instance_ids = []
    probability_values = []
    blended_instances = 0
    for index, patch_id in enumerate(test_ids, start=1):
        weighted_patch = np.asarray(weighted_predictions[int(patch_id)], dtype=np.int32)
        uniform_patch = np.asarray(uniform_predictions[int(patch_id)], dtype=np.int32)
        matches = match_instances(weighted_patch[..., 0], uniform_patch[..., 0])
        weighted_classes = decoded_class_lookup(weighted_patch)
        uniform_classes = decoded_class_lookup(uniform_patch)
        blended_probability_by_instance: dict[int, np.ndarray] = {}
        assignments, blended = blended_class_lookup_for_patch(
            int(patch_id),
            weighted_patch,
            uniform_patch,
            weighted_probabilities,
            uniform_probabilities,
            weight,
            instance_matches=matches,
            control_decoded_classes=weighted_classes,
            candidate_decoded_classes=uniform_classes,
            assignment_probabilities=blended_probability_by_instance,
        )
        blended_instances += blended
        instance_map = weighted_patch[..., 0]
        class_lut = np.zeros(int(instance_map.max()) + 1, dtype=np.uint8)
        for instance_id, class_id in assignments.items():
            class_lut[instance_id] = class_id
        output[int(patch_id), ..., 0] = instance_map
        output[int(patch_id), ..., 1] = class_lut[instance_map]
        counts[int(patch_id)] = central_counts_from_lookup(instance_map, assignments)
        instance_ids = np.asarray(sorted(assignments), dtype=np.int32)
        probability_patch_ids.append(np.full(len(instance_ids), int(patch_id), dtype=np.int32))
        probability_instance_ids.append(instance_ids)
        probability_values.append(
            np.asarray(
                [blended_probability_by_instance[int(instance_id)] for instance_id in instance_ids],
                dtype=np.float32,
            ).reshape(-1, 6)
        )
        if index % 50 == 0 or index == len(test_ids):
            print(f"Final blend {index}/{len(test_ids)}", flush=True)
    output.flush()
    del output
    np.save(output_counts, counts)
    np.savez_compressed(
        output_probabilities,
        patch_ids=np.concatenate(probability_patch_ids),
        instance_ids=np.concatenate(probability_instance_ids),
        class_probs=np.concatenate(probability_values),
    )

    weighted_checkpoint = Path(selection["weighted_checkpoint"])
    if not weighted_checkpoint.is_absolute():
        weighted_checkpoint = ROOT / weighted_checkpoint
    report_path = Path(selected["report"])
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    manifest = {
        "protocol": "single frozen-recipe materialization on 657-patch locked test",
        "selection": str(args.selection),
        "selection_sha256": sha256(args.selection),
        "validation_report": str(report_path),
        "validation_report_sha256": sha256(report_path),
        "weighted_checkpoint": str(weighted_checkpoint),
        "weighted_checkpoint_sha256": sha256(weighted_checkpoint),
        "weighted_run": str(args.weighted_run),
        "uniform_run": str(args.uniform_run),
        "geometry": "weighted six-view TTA NP/HV decoded instances",
        "typing": "uniform six-view TTA instance probabilities",
        "uniform_type_weight": weight,
        "weighted_type_weight": 1.0 - weight,
        "test_patches": int(len(test_ids)),
        "blended_matched_instances": int(blended_instances),
        "locked_test_selection_policy": "all checkpoints, weights, and eligibility gates frozen on development validation before this run",
    }
    output_manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
