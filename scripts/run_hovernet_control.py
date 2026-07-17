#!/usr/bin/env python
"""Run the official public CoNIC-trained HoVer-Net checkpoint on prepared patches."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CELLVIT_ROOT = ROOT / "third_party" / "CellViT-plus-plus"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CELLVIT_ROOT))

from cellvit.models.cell_segmentation.postprocessing import DetectionCellPostProcessor
from cpath_conic.data import central_crop_counts, official_hovernet_fold
from cpath_conic.hovernet_baseline import load_official_hovernet
from cpath_conic.stain import deterministic_hed_stain_transfer
from cpath_conic.tta import (
    invert_hv_horizontal_flip,
    invert_hv_rotation,
    invert_hv_vertical_flip,
    invert_spatial_rotation,
)


def process_prediction(np_map: np.ndarray, hv_map: np.ndarray, type_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the official 0.25-mpp HoVer-Net decoder and return 0.5-mpp maps."""
    np_map = cv2.resize(np_map, (512, 512), interpolation=cv2.INTER_LINEAR)
    hv_map = cv2.resize(hv_map, (512, 512), interpolation=cv2.INTER_LINEAR)
    type_map = cv2.resize(type_map, (512, 512), interpolation=cv2.INTER_NEAREST)
    processor = DetectionCellPostProcessor(nr_types=7, magnification=40)
    combined = np.concatenate((type_map[..., None], np_map[..., None], hv_map), axis=-1)
    instances, cells = processor.post_process_single_image(combined)
    classes = np.zeros_like(instances, dtype=np.uint8)
    for instance_id, information in cells.items():
        classes[instances == int(instance_id)] = int(information["type"])
    instances = cv2.resize(instances.astype(np.int32), (256, 256), interpolation=cv2.INTER_NEAREST)
    classes = cv2.resize(classes, (256, 256), interpolation=cv2.INTER_NEAREST)
    return instances.astype(np.int32), classes.astype(np.uint8)


def instance_class_probabilities(instances: np.ndarray, type_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool six conditional cell-type probabilities over each decoded nucleus."""
    resized = cv2.resize(type_probs.astype(np.float32), (512, 512), interpolation=cv2.INTER_LINEAR)
    resized = cv2.resize(resized, (256, 256), interpolation=cv2.INTER_LINEAR)
    instance_ids = np.unique(instances)
    instance_ids = instance_ids[instance_ids > 0].astype(np.int32)
    probabilities = np.zeros((len(instance_ids), 6), dtype=np.float32)
    for index, instance_id in enumerate(instance_ids):
        values = resized[instances == instance_id, 1:7].mean(axis=0)
        total = float(values.sum())
        probabilities[index] = values / total if total > 0 else np.full(6, 1.0 / 6.0, dtype=np.float32)
    return instance_ids, probabilities


def build_stain_views(
    images: np.ndarray,
    target_concentration: np.ndarray | None,
    strength: float = 1.0,
    mode: str = "average",
) -> tuple[list[np.ndarray], int]:
    """Build native-only or native+deterministically-styled inference views."""
    values = np.asarray(images)
    if values.ndim != 4 or values.shape[-1] != 3 or values.dtype != np.uint8:
        raise ValueError("images must be uint8 BxHxWx3 RGB")
    if target_concentration is None:
        if mode != "average":
            raise ValueError("a stain target is required for styled-only inference")
        return [values], 0
    if mode not in {"average", "styled"}:
        raise ValueError("stain view mode must be 'average' or 'styled'")
    target = np.asarray(target_concentration, dtype=np.float32)
    styled = np.stack(
        [deterministic_hed_stain_transfer(image, target, strength=strength) for image in values]
    )
    changed = int(np.sum(np.any(styled != values, axis=(1, 2, 3))))
    return ([values, styled] if mode == "average" else [styled]), changed


def normalize_model_weights(model_count: int, weights: list[float] | None) -> np.ndarray:
    """Return non-negative model weights summing to one.

    Keeping this normalization separate makes the map-level ensemble invariant
    to the number of spatial or stain views. ``None`` exactly recovers the
    historical uniform ensemble.
    """
    if model_count < 1:
        raise ValueError("at least one model is required")
    if weights is None:
        return np.full(model_count, 1.0 / model_count, dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (model_count,):
        raise ValueError("--model-weights must provide one value per checkpoint")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("--model-weights must be finite and non-negative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("--model-weights must contain at least one positive value")
    return values / total


def resolve_branch_model_weights(
    model_count: int,
    model_weights: list[float] | None = None,
    np_weights: list[float] | None = None,
    hv_weights: list[float] | None = None,
    tp_weights: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Resolve optional NP/HV/TP overrides without changing legacy defaults."""
    shared = normalize_model_weights(model_count, model_weights)
    return {
        "np": shared if np_weights is None else normalize_model_weights(model_count, np_weights),
        "hv": shared if hv_weights is None else normalize_model_weights(model_count, hv_weights),
        "tp": shared if tp_weights is None else normalize_model_weights(model_count, tp_weights),
    }


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="One or more checkpoints. Multiple models are combined at NP/HV/TP-map level before one decode.",
    )
    parser.add_argument(
        "--model-weights",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional non-negative checkpoint weights, in --checkpoint order. "
            "Weights are normalized to sum to one; omission preserves uniform averaging."
        ),
    )
    for branch, description in (
        ("np", "foreground probabilities"),
        ("hv", "horizontal/vertical geometry vectors"),
        ("tp", "cell-type probabilities"),
    ):
        parser.add_argument(
            f"--{branch}-model-weights",
            type=float,
            nargs="+",
            default=None,
            help=(
                f"Optional checkpoint weights for {description}; overrides --model-weights only for {branch.upper()}. "
                "Use validation selection and provide one non-negative value per checkpoint."
            ),
        )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "official_val", "all"], default="official_val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--flip-tta", action="store_true", help="Average identity and horizontal/vertical flips before decoding")
    parser.add_argument("--rotation-tta", action="store_true", help="Also average 90/180/270 degree rotations before decoding")
    parser.add_argument(
        "--stain-target",
        type=float,
        nargs=2,
        metavar=("H", "E"),
        default=None,
        help="Add one deterministic H/E-translated view and average raw maps with the native view.",
    )
    parser.add_argument("--stain-target-label", default=None, help="Provenance label for the H/E target")
    parser.add_argument("--stain-anchor-patch-id", type=int, default=None)
    parser.add_argument("--stain-strength", type=float, default=1.0)
    parser.add_argument(
        "--stain-view-mode",
        choices=["average", "styled"],
        default="average",
        help="Average native+styled raw maps, or evaluate the styled view alone.",
    )
    args = parser.parse_args()
    if not 0 <= args.stain_strength <= 1:
        parser.error("--stain-strength must lie in [0, 1]")
    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    if args.split == "all":
        selected = metadata
    elif args.split == "official_val":
        _, official_validation_ids = official_hovernet_fold(metadata)
        selected = metadata.loc[metadata.patch_id.isin(official_validation_ids)]
    else:
        selected = metadata.loc[metadata.split.eq(args.split)]
    if args.max_patches:
        selected = selected.head(args.max_patches)
    models = [
        load_official_hovernet(checkpoint, ROOT / "third_party" / "CoNIC", args.device)
        for checkpoint in args.checkpoint
    ]
    try:
        branch_model_weights = resolve_branch_model_weights(
            len(models),
            args.model_weights,
            args.np_model_weights,
            args.hv_model_weights,
            args.tp_model_weights,
        )
    except ValueError as error:
        parser.error(str(error))
    model_weights = normalize_model_weights(len(models), args.model_weights)
    branch_specific_weights = not (
        np.allclose(branch_model_weights["np"], branch_model_weights["hv"])
        and np.allclose(branch_model_weights["np"], branch_model_weights["tp"])
    )
    checkpoint_payloads = [
        torch.load(checkpoint, map_location="cpu", weights_only=False) for checkpoint in args.checkpoint
    ]
    is_our_fold_checkpoint = all(
        isinstance(payload, dict)
        and "desc" in payload
        and str(payload.get("initialization", "")).startswith("ImageNet ")
        and str(payload.get("initialization", "")).endswith("only; no CoNIC checkpoint")
        for payload in checkpoint_payloads
    )
    n_total = int(metadata.patch_id.max()) + 1
    predictions = np.zeros((n_total, 256, 256, 2), dtype=np.int16)
    counts = np.zeros((n_total, 6), dtype=np.int32)
    probability_patch_ids = []
    probability_instance_ids = []
    probability_values = []
    stain_changed_patches = 0
    stain_total_patches = 0
    patch_ids = selected.patch_id.to_numpy(dtype=np.int32)
    with torch.inference_mode():
        for start in range(0, len(patch_ids), args.batch_size):
            batch_ids = patch_ids[start : start + args.batch_size]
            images = np.stack(
                [np.asarray(Image.open(args.prepared / "images" / f"{int(patch_id):05d}.png").convert("RGB")) for patch_id in batch_ids]
            )
            image_views, changed = build_stain_views(
                images, args.stain_target, args.stain_strength, args.stain_view_mode
            )
            if args.stain_target is not None:
                stain_changed_patches += changed
                stain_total_patches += len(images)
            np_sum = None
            hv_sum = None
            tp_sum = None
            aggregation_weights = {branch: 0.0 for branch in branch_model_weights}
            for view_images in image_views:
                tensor = torch.from_numpy(view_images).permute(0, 3, 1, 2).float().to(args.device)
                for model_index, model in enumerate(models):
                    np_weight = float(branch_model_weights["np"][model_index])
                    hv_weight = float(branch_model_weights["hv"][model_index])
                    tp_weight = float(branch_model_weights["tp"][model_index])
                    output = model(tensor)
                    member_np = torch.softmax(output["np"].float(), dim=1)
                    member_hv = output["hv"].float()
                    member_tp = torch.softmax(output["tp"].float(), dim=1)
                    weighted_np = member_np * np_weight
                    weighted_hv = member_hv * hv_weight
                    weighted_tp = member_tp * tp_weight
                    np_sum = weighted_np if np_sum is None else np_sum + weighted_np
                    hv_sum = weighted_hv if hv_sum is None else hv_sum + weighted_hv
                    tp_sum = weighted_tp if tp_sum is None else tp_sum + weighted_tp
                    aggregation_weights["np"] += np_weight
                    aggregation_weights["hv"] += hv_weight
                    aggregation_weights["tp"] += tp_weight
                    if args.flip_tta:
                        output_h = model(tensor.flip(-1))
                        np_sum += torch.softmax(output_h["np"].float(), dim=1).flip(-1) * np_weight
                        hv_sum += invert_hv_horizontal_flip(output_h["hv"].float()) * hv_weight
                        tp_sum += torch.softmax(output_h["tp"].float(), dim=1).flip(-1) * tp_weight
                        output_v = model(tensor.flip(-2))
                        np_sum += torch.softmax(output_v["np"].float(), dim=1).flip(-2) * np_weight
                        hv_sum += invert_hv_vertical_flip(output_v["hv"].float()) * hv_weight
                        tp_sum += torch.softmax(output_v["tp"].float(), dim=1).flip(-2) * tp_weight
                        for branch, weight in (("np", np_weight), ("hv", hv_weight), ("tp", tp_weight)):
                            aggregation_weights[branch] += 2.0 * weight
                    if args.rotation_tta:
                        for k in (1, 2, 3):
                            output_r = model(torch.rot90(tensor, k, dims=(-2, -1)))
                            np_sum += invert_spatial_rotation(torch.softmax(output_r["np"].float(), dim=1), k) * np_weight
                            hv_sum += invert_hv_rotation(output_r["hv"].float(), k) * hv_weight
                            tp_sum += invert_spatial_rotation(torch.softmax(output_r["tp"].float(), dim=1), k) * tp_weight
                            aggregation_weights["np"] += np_weight
                            aggregation_weights["hv"] += hv_weight
                            aggregation_weights["tp"] += tp_weight
            if np_sum is None or hv_sum is None or tp_sum is None:
                raise RuntimeError("no HoVer-Net ensemble members were loaded")
            np_probabilities = np_sum / aggregation_weights["np"]
            hv_probabilities = hv_sum / aggregation_weights["hv"]
            type_probabilities = tp_sum / aggregation_weights["tp"]
            np_prob = np_probabilities[:, 1].cpu().numpy()
            hv_maps = hv_probabilities.permute(0, 2, 3, 1).cpu().numpy()
            type_probs = type_probabilities.permute(0, 2, 3, 1).cpu().numpy()
            type_maps = type_probs.argmax(-1)
            for offset, patch_id in enumerate(batch_ids):
                instances, classes = process_prediction(np_prob[offset], hv_maps[offset], type_maps[offset])
                predictions[patch_id, ..., 0] = instances
                predictions[patch_id, ..., 1] = classes
                counts[patch_id] = central_crop_counts(instances, classes)
                instance_ids, class_probs = instance_class_probabilities(instances, type_probs[offset])
                probability_patch_ids.append(np.full(len(instance_ids), int(patch_id), dtype=np.int32))
                probability_instance_ids.append(instance_ids)
                probability_values.append(class_probs)
            print(f"HoVer-Net {min(start + len(batch_ids), len(patch_ids))}/{len(patch_ids)}", flush=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    np.save(args.outdir / "predictions.npy", predictions)
    np.save(args.outdir / "counts.npy", counts)
    np.savez_compressed(
        args.outdir / "cell_probabilities.npz",
        patch_ids=np.concatenate(probability_patch_ids) if probability_patch_ids else np.empty(0, dtype=np.int32),
        instance_ids=np.concatenate(probability_instance_ids) if probability_instance_ids else np.empty(0, dtype=np.int32),
        class_probs=np.concatenate(probability_values) if probability_values else np.empty((0, 6), dtype=np.float32),
    )
    (args.outdir / "run.json").write_text(
        json.dumps(
            {
                "checkpoints": [str(checkpoint) for checkpoint in args.checkpoint],
                "checkpoint_sha256": [hashlib.sha256(checkpoint.read_bytes()).hexdigest() for checkpoint in args.checkpoint],
                "model_count": len(models),
                "model_weights": model_weights.tolist(),
                "branch_model_weights": {
                    branch: weights.tolist() for branch, weights in branch_model_weights.items()
                },
                "source": (
                    f"{'branch-specific weighted' if branch_specific_weights else 'shared-weight'} raw-map ensemble of {len(models)} HoVer-Net fits trained from generic ImageNet initialization"
                    if is_our_fold_checkpoint
                    else "official TissueImageAnalytics/CoNIC baseline checkpoint and net_desc.py"
                ),
                "checkpoint_training_metadata": [
                    {
                        key: payload.get(key)
                        for key in (
                            "phase",
                            "phase_epoch",
                            "epoch",
                            "official_hovernet_commit",
                            "backbone_architecture",
                            "initialization",
                            "metrics",
                        )
                        if isinstance(payload, dict) and key in payload
                    }
                    for payload in checkpoint_payloads
                ],
                "split": args.split,
                "n_processed": len(patch_ids),
                "tta": {
                    "flip": args.flip_tta,
                    "rotation": args.rotation_tta,
                    "spatial_transforms_per_model": 1 + (2 if args.flip_tta else 0) + (3 if args.rotation_tta else 0),
                    "stain_views": (
                        2 if args.stain_target is not None and args.stain_view_mode == "average" else 1
                    ),
                    "total_forward_passes_per_patch": len(models)
                    * (1 + (2 if args.flip_tta else 0) + (3 if args.rotation_tta else 0))
                    * (2 if args.stain_target is not None and args.stain_view_mode == "average" else 1),
                    "averaging": "independently weighted NP/TP probabilities and exactly inverted HV vectors over models, with uniform spatial/stain views, before one decoder pass",
                },
                "stain_tta": {
                    "enabled": args.stain_target is not None,
                    "target_concentration": args.stain_target,
                    "target_label": args.stain_target_label,
                    "anchor_patch_id": args.stain_anchor_patch_id,
                    "strength": args.stain_strength,
                    "view_mode": args.stain_view_mode if args.stain_target is not None else "native",
                    "native_weight": (
                        0.5 if args.stain_target is not None and args.stain_view_mode == "average"
                        else 0.0 if args.stain_target is not None else 1.0
                    ),
                    "styled_weight": (
                        0.5 if args.stain_target is not None and args.stain_view_mode == "average"
                        else 1.0 if args.stain_target is not None else 0.0
                    ),
                    "changed_patch_fraction": (
                        stain_changed_patches / stain_total_patches if stain_total_patches else None
                    ),
                    "guard_fallback_fraction": (
                        1.0 - stain_changed_patches / stain_total_patches if stain_total_patches else None
                    ),
                },
                "postprocessing": "2x to 0.25 mpp, HoVer-Net NP/HV watershed, nearest downsample to 0.5 mpp",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
