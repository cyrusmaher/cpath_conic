#!/usr/bin/env python
"""Audit whether a LoRA segmentation change helps rare GT cell classes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CELLVIT_ROOT = ROOT / "third_party" / "CellViT-plus-plus"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CELLVIT_ROOT))

from cpath_conic.constants import CLASS_NAMES
from cpath_conic.hv import decode_hv, fast_binary_pq_stats
from cpath_conic.lora import load_lora_adapter
from scripts.run_cellvit_conic import load_model
from scripts.train_lora_segmentation import CoNICSegmentationDataset


def matched_gt_instances(true_inst: np.ndarray, pred_inst: np.ndarray, threshold: float = 0.5):
    """Return matched GT IDs, their IoUs, and binary instance totals."""
    true_flat = np.asarray(true_inst, dtype=np.int64).ravel()
    pred_flat = np.asarray(pred_inst, dtype=np.int64).ravel()
    n_true = int(true_flat.max()) + 1
    n_pred = int(pred_flat.max()) + 1
    true_area = np.bincount(true_flat, minlength=n_true)
    pred_area = np.bincount(pred_flat, minlength=n_pred)
    joint = np.bincount(true_flat * n_pred + pred_flat, minlength=n_true * n_pred).reshape(n_true, n_pred)
    gt_ids, pred_ids = np.nonzero(joint[1:, 1:])
    gt_ids += 1
    pred_ids += 1
    intersections = joint[gt_ids, pred_ids]
    unions = true_area[gt_ids] + pred_area[pred_ids] - intersections
    ious = intersections / np.maximum(unions, 1)
    keep = ious > threshold
    return gt_ids[keep], ious[keep], n_true - 1, n_pred - 1


def gt_instance_classes(true_inst: np.ndarray, class_map: np.ndarray) -> np.ndarray:
    """Map contiguous GT instance IDs to their majority CoNIC class."""
    classes = np.zeros(int(true_inst.max()) + 1, dtype=np.int8)
    for instance_id in range(1, len(classes)):
        pixels = class_map[true_inst == instance_id]
        if len(pixels):
            classes[instance_id] = np.bincount(pixels.astype(np.int64), minlength=7).argmax()
    return classes


def empty_totals() -> dict:
    return {
        "gt": np.zeros(len(CLASS_NAMES), dtype=np.int64),
        "matched": np.zeros(len(CLASS_NAMES), dtype=np.int64),
        "sum_iou": np.zeros(len(CLASS_NAMES), dtype=np.float64),
        "predicted": 0,
    }


def update_totals(totals: dict, true_inst: np.ndarray, class_map: np.ndarray, pred_inst: np.ndarray) -> None:
    classes = gt_instance_classes(true_inst, class_map)
    gt_ids, ious, n_true, n_pred = matched_gt_instances(true_inst, pred_inst)
    gt_classes = classes[1 : n_true + 1]
    totals["gt"] += np.bincount(gt_classes - 1, minlength=len(CLASS_NAMES))
    matched_classes = classes[gt_ids]
    totals["matched"] += np.bincount(matched_classes - 1, minlength=len(CLASS_NAMES))
    np.add.at(totals["sum_iou"], matched_classes - 1, ious)
    totals["predicted"] += n_pred


def summarize(totals: dict) -> dict:
    gt = totals["gt"]
    matched = totals["matched"]
    by_class = {}
    for index, name in enumerate(CLASS_NAMES):
        by_class[name] = {
            "gt": int(gt[index]),
            "matched": int(matched[index]),
            "missed": int(gt[index] - matched[index]),
            "recall_at_iou_0.5": float(matched[index] / gt[index]) if gt[index] else None,
            "matched_mean_iou": float(totals["sum_iou"][index] / matched[index]) if matched[index] else None,
        }
    total_gt = int(gt.sum())
    total_matched = int(matched.sum())
    false_positive = int(totals["predicted"] - total_matched)
    false_negative = total_gt - total_matched
    denominator = total_matched + 0.5 * false_positive + 0.5 * false_negative
    sum_iou = float(totals["sum_iou"].sum())
    return {
        "gt": total_gt,
        "matched": total_matched,
        "missed": total_gt - total_matched,
        "predicted": int(totals["predicted"]),
        "false_positive_binary": false_positive,
        "recall_at_iou_0.5": float(total_matched / total_gt),
        "bPQ": float(sum_iou / denominator) if denominator else 0.0,
        "DQ": float(total_matched / denominator) if denominator else 0.0,
        "SQ": float(sum_iou / total_matched) if total_matched else 0.0,
        "by_class": by_class,
    }


def paired_bootstrap_bpq(candidate_stats: np.ndarray, reference_stats: np.ndarray, seed: int = 20260715) -> dict:
    """Paired patch bootstrap for the difference in pooled binary PQ."""
    candidate_stats = np.asarray(candidate_stats, dtype=np.float64)
    reference_stats = np.asarray(reference_stats, dtype=np.float64)
    if candidate_stats.shape != reference_stats.shape or candidate_stats.ndim != 2 or candidate_stats.shape[1] != 4:
        raise ValueError("candidate/reference stats must be aligned patches-by-four arrays")

    def score(totals):
        tp, fp, fn, sum_iou = totals
        denominator = tp + 0.5 * fp + 0.5 * fn
        return sum_iou / denominator if denominator else 0.0

    point = score(candidate_stats.sum(axis=0)) - score(reference_stats.sum(axis=0))
    rng = np.random.default_rng(seed)
    deltas = np.empty(2000, dtype=np.float64)
    for iteration in range(len(deltas)):
        sample = rng.integers(0, len(candidate_stats), size=len(candidate_stats))
        deltas[iteration] = score(candidate_stats[sample].sum(axis=0)) - score(reference_stats[sample].sum(axis=0))
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    return {
        "candidate_minus_reference": float(point),
        "paired_patch_bootstrap_95_ci": [float(lower), float(upper)],
        "replicates": len(deltas),
        "seed": seed,
    }


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-adapter", type=Path, required=True)
    parser.add_argument("--reference-instance-maps", type=Path, required=True)
    parser.add_argument("--reference-name", default="uniform_lora")
    parser.add_argument("--candidate-name", default="minority_lora")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    metadata = pd.read_csv(args.prepared / "metadata.csv")
    val_ids = metadata.loc[metadata.split == "val", "patch_id"].to_numpy(dtype=np.int32)
    reference_maps = np.load(args.reference_instance_maps, mmap_mode="r")
    model, mean, std = load_model(args.checkpoint, args.device)
    configuration = load_lora_adapter(model, args.candidate_adapter)
    model.to(args.device).eval()
    dataset = CoNICSegmentationDataset(args.prepared, val_ids, mean, std)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=args.device.startswith("cuda"),
    )
    candidate_totals = empty_totals()
    reference_totals = empty_totals()
    sources = sorted(metadata.loc[metadata.split == "val", "source"].dropna().astype(str).unique())
    candidate_by_source = {source: empty_totals() for source in sources}
    reference_by_source = {source: empty_totals() for source in sources}
    candidate_patch_stats = []
    reference_patch_stats = []
    source_by_patch = metadata.set_index("patch_id")["source"].astype(str).to_dict()
    decoder_config = configuration["decoder_config"]
    with torch.no_grad():
        for batch in loader:
            patch_id = int(batch["patch_id"].item())
            output = model(batch["image"].float().to(args.device))
            foreground = torch.softmax(output["nuclei_binary_map"].float(), dim=1)[0, 1].cpu().numpy()
            hv_map = output["hv_map"].float()[0].permute(1, 2, 0).cpu().numpy()
            candidate_map = decode_hv(foreground, hv_map, **decoder_config)
            label = np.load(args.prepared / "labels" / f"{patch_id:05d}.npy", allow_pickle=True).item()
            true_inst = label["inst_map"].astype(np.int32)
            class_map = label["class_map"].astype(np.int8)
            update_totals(candidate_totals, true_inst, class_map, candidate_map)
            update_totals(reference_totals, true_inst, class_map, reference_maps[patch_id])
            candidate_patch_stats.append(fast_binary_pq_stats(true_inst, candidate_map))
            reference_patch_stats.append(fast_binary_pq_stats(true_inst, reference_maps[patch_id]))
            source = source_by_patch[patch_id]
            update_totals(candidate_by_source[source], true_inst, class_map, candidate_map)
            update_totals(reference_by_source[source], true_inst, class_map, reference_maps[patch_id])

    candidate = summarize(candidate_totals)
    reference = summarize(reference_totals)
    deltas = {
        name: {
            "matched": candidate["by_class"][name]["matched"] - reference["by_class"][name]["matched"],
            "recall_at_iou_0.5": candidate["by_class"][name]["recall_at_iou_0.5"]
            - reference["by_class"][name]["recall_at_iou_0.5"],
            "matched_mean_iou": candidate["by_class"][name]["matched_mean_iou"]
            - reference["by_class"][name]["matched_mean_iou"],
        }
        for name in CLASS_NAMES
    }
    report = {
        "split": "val",
        "matching": "GT-stratified binary detection; IoU > 0.5; predicted instances have no class assignment",
        "reference": {"name": args.reference_name, **reference},
        "candidate": {"name": args.candidate_name, **candidate},
        "candidate_minus_reference_by_class": deltas,
        "paired_bPQ": paired_bootstrap_bpq(candidate_patch_stats, reference_patch_stats),
        "by_source": {
            source: {
                "reference": summarize(reference_by_source[source]),
                "candidate": summarize(candidate_by_source[source]),
            }
            for source in sources
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
