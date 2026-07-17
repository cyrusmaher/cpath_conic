#!/usr/bin/env python
"""Cache and validation-score flip plus rotation raw-map TTA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CELLVIT_ROOT = ROOT / "third_party" / "CellViT-plus-plus"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CELLVIT_ROOT))

from cpath_conic.hv import decode_hv, fast_binary_pq_stats
from cpath_conic.tta import invert_hv_rotation, invert_spatial_rotation
from scripts.run_cellvit_conic import load_model


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="Existing tune_hv_decoder validation cache")
    parser.add_argument("--decoder-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    patch_ids = np.load(args.cache / "patch_ids.npy")
    flip_maps = np.load(args.cache / "raw_maps_flip_tta.npy", mmap_mode="r")
    truth = np.load(args.cache / "true_instances.npy", mmap_mode="r")
    output_path = args.cache / "raw_maps_flip_rotation_tta.npy"
    combined_maps = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=flip_maps.shape,
    )
    model, mean, std = load_model(args.checkpoint, args.device)

    with torch.no_grad():
        for start in range(0, len(patch_ids), args.batch_size):
            batch_ids = patch_ids[start : start + args.batch_size]
            images = []
            for patch_id in batch_ids:
                image = np.asarray(
                    Image.open(args.prepared / "images" / f"{int(patch_id):05d}.png").convert("RGB"),
                    dtype=np.float32,
                ) / 255.0
                images.append(((image - mean) / std).transpose(2, 0, 1))
            tensor = torch.from_numpy(np.asarray(images)).float().to(args.device)
            rotation_sum = np.zeros((len(batch_ids), 256, 256, 3), dtype=np.float32)
            for k in (1, 2, 3):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                    output = model(torch.rot90(tensor, k, dims=(-2, -1)))
                foreground = invert_spatial_rotation(
                    torch.softmax(output["nuclei_binary_map"].float(), dim=1)[:, 1], k
                ).cpu().numpy()
                hv = invert_hv_rotation(output["hv_map"].float(), k).permute(0, 2, 3, 1).cpu().numpy()
                rotation_sum[..., 0] += foreground
                rotation_sum[..., 1:] += hv
            combined_maps[start : start + len(batch_ids)] = (
                (3.0 * np.asarray(flip_maps[start : start + len(batch_ids)], dtype=np.float32) + rotation_sum) / 6.0
            ).astype(np.float16)
            combined_maps.flush()
            print(f"cached rotation TTA {min(start + len(batch_ids), len(patch_ids))}/{len(patch_ids)}", flush=True)

    decoder_report = json.loads(args.decoder_report.read_text())
    config = decoder_report["best"]["config"]
    totals = np.zeros(4, dtype=np.float64)
    predicted_count = 0
    true_count = 0
    for index in range(len(combined_maps)):
        predicted = decode_hv(combined_maps[index, ..., 0], combined_maps[index, ..., 1:], **config)
        totals += fast_binary_pq_stats(truth[index], predicted)
        predicted_count += int(predicted.max())
        true_count += int(truth[index].max())
    tp, fp, fn, sum_iou = totals
    denominator = tp + 0.5 * fp + 0.5 * fn
    metrics = {
        "bPQ": float(sum_iou / denominator),
        "DQ": float(tp / denominator),
        "SQ": float(sum_iou / tp),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "count_ratio": float(predicted_count / true_count),
    }
    report = {
        "selection_split": "validation",
        "decoder": "locked validation-selected native-resolution HV decoder",
        "transforms": ["identity", "horizontal flip", "vertical flip", "rotation 90", "rotation 180", "rotation 270"],
        "flip_tta_reference": decoder_report["flip_tta_locked_decoder"]["metrics"],
        "flip_rotation_tta": metrics,
        "map_path": str(output_path),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
