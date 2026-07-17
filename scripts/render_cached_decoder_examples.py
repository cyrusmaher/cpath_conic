#!/usr/bin/env python
"""Render agent/human review panels for native-vs-TTA decoder behavior."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.data import central_crop_counts, load_patch
from cpath_conic.hv import decode_hv
from cpath_conic.visuals import render_panel


def assign_overlap_classes(pred_inst: np.ndarray, true_inst: np.ndarray, true_cls: np.ndarray) -> np.ndarray:
    """Give each prediction its majority-overlap GT class for detection-only review."""
    pred_cls = np.zeros_like(pred_inst, dtype=np.uint8)
    for pred_id in np.unique(pred_inst):
        if pred_id == 0:
            continue
        mask = pred_inst == pred_id
        gt_ids, counts = np.unique(true_inst[mask], return_counts=True)
        overlaps = [(int(count), int(gt_id)) for gt_id, count in zip(gt_ids, counts) if gt_id != 0]
        if not overlaps:
            continue
        gt_id = max(overlaps)[1]
        pixels = true_cls[true_inst == gt_id]
        class_id = int(np.bincount(pixels.astype(np.int64), minlength=7).argmax()) if len(pixels) else 0
        pred_cls[mask] = class_id
    return pred_cls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--decoder-report", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--per-tail", type=int, default=4)
    args = parser.parse_args()

    decoder = json.loads(args.decoder_report.read_text())
    config = decoder.get("best", {}).get("config", decoder.get("config", decoder))
    audit = json.loads(args.audit_report.read_text())
    patch_ids = np.load(args.cache / "patch_ids.npy")
    index_by_id = {int(patch_id): index for index, patch_id in enumerate(patch_ids)}
    native_raw = np.load(args.cache / "raw_maps.npy", mmap_mode="r")
    tta_raw = np.load(args.cache / "raw_maps_flip_tta.npy", mmap_mode="r")
    selected = [
        *(dict(row, tail="largest TTA gain") for row in audit["largest_patch_gains"][: args.per_tail]),
        *(dict(row, tail="largest TTA loss") for row in audit["largest_patch_losses"][: args.per_tail]),
    ]
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in selected:
        patch_id = int(row["patch_id"])
        index = index_by_id[patch_id]
        image, true_inst, true_cls, metadata = load_patch(args.prepared, patch_id)
        native = decode_hv(native_raw[index, ..., 0], native_raw[index, ..., 1:], **config)
        tta = decode_hv(tta_raw[index, ..., 0], tta_raw[index, ..., 1:], **config)
        for method, pred_inst in (("native", native), ("flip_tta", tta)):
            pred_cls = assign_overlap_classes(pred_inst, true_inst, true_cls)
            gt_counts = central_crop_counts(true_inst, true_cls)
            pred_counts = central_crop_counts(pred_inst, pred_cls)
            title = (
                f"E26 {method} | patch {patch_id} | source={metadata.get('source', '')} | {row['tail']} | "
                f"patch bPQ {row['reference_bPQ']:.3f} -> {row['candidate_bPQ']:.3f}"
            )
            panel = render_panel(image, true_inst, true_cls, pred_inst, pred_cls, gt_counts, pred_counts, title)
            path = args.outdir / f"{patch_id:05d}_{method}.png"
            Image.fromarray(panel).save(path)
            manifest.append(
                {
                    **row,
                    "method": method,
                    "panel": path.name,
                    "note": "Prediction colors use the majority-overlap GT class; unoverlapped false positives have white boundaries only.",
                }
            )
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"rendered {len(manifest)} panels to {args.outdir}")


if __name__ == "__main__":
    main()
