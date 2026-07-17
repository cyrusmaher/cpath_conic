#!/usr/bin/env python
"""Apply a validation-selected metric-aligned token classifier."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CELLVIT_ROOT = ROOT / "third_party" / "CellViT-plus-plus"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CELLVIT_ROOT))

from cpath_conic.data import central_crop_counts
from scripts.train_metric_aligned_classifier import infer_assignments


def counts_from_assignments(
    patch_ids: np.ndarray,
    assignments: np.ndarray,
    central: np.ndarray,
    n_patches: int,
) -> np.ndarray:
    """Count non-dustbin assignments whose fixed instances touch the center."""
    counts = np.zeros((n_patches, 6), dtype=np.int32)
    keep = central.astype(bool) & (assignments > 0)
    np.add.at(counts, (patch_ids[keep], assignments[keep] - 1), 1)
    return counts


def main() -> None:
    import torch
    from cellvit.models.classifier.linear_classifier import LinearClassifier

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--instance-maps", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--probabilities-only", action="store_true", help="Skip materializing the full pixel prediction tensor")
    parser.add_argument("--fixed-mask-cache", type=Path, default=None, help="Optional aligned cache used to export counts without a pixel tensor")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    feature_data = np.load(args.features)
    features = feature_data["features"].astype(np.float32)
    patch_ids = feature_data["patch_ids"].astype(np.int32)
    instance_ids = feature_data["instance_ids"].astype(np.int32)
    checkpoint = torch.load(args.classifier, map_location="cpu", weights_only=False)
    model = LinearClassifier(
        embed_dim=int(checkpoint["embed_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_classes=int(checkpoint["num_classes"]),
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    assignments, probabilities = infer_assignments(model, features, args.device, int(checkpoint["num_classes"]), args.batch_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.fixed_mask_cache is not None:
        cache = np.load(args.fixed_mask_cache)
        central = cache["central"].astype(bool)
        if len(central) != len(assignments):
            raise ValueError("Fixed-mask cache and feature records have different lengths")
        n_patches = int(max(patch_ids.max(), cache["gt_full_counts"].shape[0] - 1)) + 1
        np.save(args.out.with_name("counts.npy"), counts_from_assignments(patch_ids, assignments, central, n_patches))
    if not args.probabilities_only:
        instance_maps = np.load(args.instance_maps, mmap_mode="r")
        output = np.zeros((*instance_maps.shape, 2), dtype=np.int32)
        for patch_id in np.unique(patch_ids):
            rows = patch_ids == patch_id
            source_instances = np.asarray(instance_maps[int(patch_id)])
            class_lut = np.zeros(int(source_instances.max()) + 1, dtype=np.int8)
            class_lut[instance_ids[rows]] = assignments[rows]
            class_map = class_lut[source_instances]
            kept_instances = source_instances.copy()
            kept_instances[class_map == 0] = 0
            output[int(patch_id), ..., 0] = kept_instances
            output[int(patch_id), ..., 1] = class_map
        np.save(args.out, output)
        if args.fixed_mask_cache is None:
            np.save(args.out.with_name("counts.npy"), np.asarray([
                central_crop_counts(patch[..., 0], patch[..., 1]) for patch in output
            ], dtype=np.int32))
    np.savez_compressed(
        args.out.with_name("cell_probabilities.npz"),
        patch_ids=patch_ids,
        instance_ids=instance_ids,
        class_probs=probabilities,
        assignments=assignments,
        class_names=np.asarray(checkpoint["class_names"]),
    )


if __name__ == "__main__":
    main()
