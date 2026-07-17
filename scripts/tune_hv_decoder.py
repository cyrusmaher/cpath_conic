#!/usr/bin/env python
"""Cache CellViT maps once, then select HV decoder parameters on validation bPQ."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CELLVIT_ROOT = ROOT / "third_party" / "CellViT-plus-plus"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CELLVIT_ROOT))

from cpath_conic.hv import DEFAULT_HV_20X, decode_hv, fast_binary_pq_stats
from cpath_conic.lora import load_lora_adapter
from cpath_conic.tta import invert_hv_horizontal_flip, invert_hv_vertical_flip
from scripts.run_cellvit_conic import load_model


def cache_validation_maps(args, patch_ids: np.ndarray) -> None:
    import torch

    args.cache.mkdir(parents=True, exist_ok=True)
    raw_path = args.cache / "raw_maps.npy"
    tta_path = args.cache / "raw_maps_flip_tta.npy"
    truth_path = args.cache / "true_instances.npy"
    ids_path = args.cache / "patch_ids.npy"
    metadata_path = args.cache / "cache.json"
    if raw_path.exists() and tta_path.exists() and truth_path.exists() and ids_path.exists():
        cached_ids = np.load(ids_path)
        cached_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        requested_adapter = str(args.lora_adapter) if args.lora_adapter else None
        if np.array_equal(cached_ids, patch_ids) and cached_metadata.get("lora_adapter") == requested_adapter:
            print(f"using cached validation maps for {len(patch_ids)} patches", flush=True)
            return
        raise RuntimeError("Existing cache patch IDs or adapter differ; use a new --cache directory")
    model, mean, std = load_model(args.checkpoint, args.device)
    adapter_configuration = None
    if args.lora_adapter is not None:
        adapter_configuration = load_lora_adapter(model, args.lora_adapter)
        model.to(args.device).eval()
    map_channels = int(model.hv_map_decoder.decoder0_header[-1].out_channels)
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float16, shape=(len(patch_ids), 256, 256, 1 + map_channels))
    raw_tta = np.lib.format.open_memmap(tta_path, mode="w+", dtype=np.float16, shape=(len(patch_ids), 256, 256, 1 + map_channels))
    truth = np.lib.format.open_memmap(truth_path, mode="w+", dtype=np.int32, shape=(len(patch_ids), 256, 256))
    with torch.no_grad():
        for start in range(0, len(patch_ids), args.batch_size):
            batch_ids = patch_ids[start : start + args.batch_size]
            images = []
            for patch_id in batch_ids:
                image = np.asarray(Image.open(args.prepared / "images" / f"{int(patch_id):05d}.png").convert("RGB"), dtype=np.float32) / 255.0
                images.append(((image - mean) / std).transpose(2, 0, 1))
            tensor = torch.from_numpy(np.asarray(images)).float().to(args.device)
            def infer_maps(input_tensor):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                    output = model(input_tensor)
                foreground_map = torch.softmax(output["nuclei_binary_map"].float(), dim=1)[:, 1].cpu().numpy()
                hv_output = output["hv_map"].float().permute(0, 2, 3, 1).cpu().numpy()
                return foreground_map, hv_output

            foreground, hv = infer_maps(tensor)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                output_h = model(tensor.flip(-1))
                output_v = model(tensor.flip(-2))
            foreground_h = torch.softmax(output_h["nuclei_binary_map"].float(), dim=1)[:, 1].flip(-1).cpu().numpy()
            hv_h = invert_hv_horizontal_flip(output_h["hv_map"].float()).permute(0, 2, 3, 1).cpu().numpy()
            foreground_v = torch.softmax(output_v["nuclei_binary_map"].float(), dim=1)[:, 1].flip(-2).cpu().numpy()
            hv_v = invert_hv_vertical_flip(output_v["hv_map"].float()).permute(0, 2, 3, 1).cpu().numpy()
            foreground_tta = (foreground + foreground_h + foreground_v) / 3.0
            hv_tta = (hv + hv_h + hv_v) / 3.0
            raw[start : start + len(batch_ids), ..., 0] = foreground.astype(np.float16)
            raw[start : start + len(batch_ids), ..., 1:] = hv.astype(np.float16)
            raw_tta[start : start + len(batch_ids), ..., 0] = foreground_tta.astype(np.float16)
            raw_tta[start : start + len(batch_ids), ..., 1:] = hv_tta.astype(np.float16)
            for offset, patch_id in enumerate(batch_ids):
                label = np.load(args.prepared / "labels" / f"{int(patch_id):05d}.npy", allow_pickle=True).item()
                truth[start + offset] = label["inst_map"].astype(np.int32)
            raw.flush()
            raw_tta.flush()
            truth.flush()
            print(f"cached {min(start + len(batch_ids), len(patch_ids))}/{len(patch_ids)} validation patches", flush=True)
    np.save(ids_path, patch_ids)
    metadata_path.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "lora_adapter": str(args.lora_adapter) if args.lora_adapter else None,
                "adapter_configuration": adapter_configuration,
                "split": "val",
                "validation_split": args.validation_split,
                "validation_sources": args.validation_sources,
                "n_patches": len(patch_ids),
                "map_channels": map_channels,
            },
            indent=2,
        )
    )


def evaluate_one(payload: tuple[str, str, dict]) -> dict:
    cache, map_file, config = payload
    cache = Path(cache)
    raw = np.load(cache / map_file, mmap_mode="r")
    truth = np.load(cache / "true_instances.npy", mmap_mode="r")
    totals = np.zeros(4, dtype=np.float64)
    predicted_count = 0
    true_count = 0
    for index in range(len(raw)):
        predicted = decode_hv(raw[index, ..., 0], raw[index, ..., 1:], **config)
        totals += fast_binary_pq_stats(truth[index], predicted)
        predicted_count += int(predicted.max())
        true_count += int(truth[index].max())
    tp, fp, fn, sum_iou = totals
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        **config,
        "bPQ": float(sum_iou / denominator if denominator else 0.0),
        "DQ": float(tp / denominator if denominator else 0.0),
        "SQ": float(sum_iou / tp if tp else 0.0),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "predicted_instances": predicted_count,
        "true_instances": true_count,
        "count_ratio": float(predicted_count / true_count if true_count else 0.0),
    }


def evaluate_many(cache: Path, configs: list[dict], workers: int, map_file: str = "raw_maps.npy") -> list[dict]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(evaluate_one, [(str(cache), map_file, config) for config in configs]))
    for row in rows:
        print(
            f"bPQ={row['bPQ']:.5f} DQ={row['DQ']:.5f} SQ={row['SQ']:.5f} ratio={row['count_ratio']:.3f} "
            f"binary={row['binary_threshold']:.2f} edge={row['edge_threshold']:.2f} object={row['object_size']} "
            f"ksize={row['ksize']} open={row['opening_size']} min={row['min_nucleus_size']}",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lora-adapter", type=Path, default=None)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--validation-split", choices=["train", "val"], default="val")
    parser.add_argument("--validation-sources", default="", help="Optional comma-separated validation source filter")
    args = parser.parse_args()

    metadata = pd.read_csv(args.prepared / "metadata.csv")
    validation_sources = [source.strip().lower() for source in args.validation_sources.split(",") if source.strip()]
    validation_mask = metadata.split.eq(args.validation_split)
    if validation_sources:
        validation_mask &= metadata.source.astype(str).str.lower().isin(validation_sources)
    patch_ids = metadata.loc[validation_mask, "patch_id"].to_numpy(dtype=np.int32)
    if args.max_patches:
        patch_ids = patch_ids[: args.max_patches]
    cache_validation_maps(args, patch_ids)

    default = DEFAULT_HV_20X.copy()
    # OpenCV Sobel kernels must be positive and odd. The previous grid stopped
    # at 7 even though validation bPQ was still improving at that boundary.
    # Include every legal smaller kernel down to the absolute minimum (1), so
    # the sweep either turns over or terminates at the mathematical boundary.
    sobel_kernel_grid = [1, 3, 5, 7, 11, 15, 21]
    edge_threshold_grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    coarse = []
    for object_size in [1, 3, 5, 10]:
        for ksize in sobel_kernel_grid:
            coarse.append({**default, "object_size": object_size, "ksize": ksize, "stage": "object_kernel"})
    coarse_rows = evaluate_many(args.cache, coarse, args.workers)
    best_coarse = max(coarse_rows, key=lambda row: row["bPQ"])
    anchor = {key: best_coarse[key] for key in default}

    thresholds = []
    for binary_threshold in [0.4, 0.5, 0.6]:
        for edge_threshold in edge_threshold_grid:
            thresholds.append({**anchor, "binary_threshold": binary_threshold, "edge_threshold": edge_threshold, "stage": "thresholds"})
    threshold_rows = evaluate_many(args.cache, thresholds, args.workers)
    best_threshold = max(threshold_rows, key=lambda row: row["bPQ"])
    anchor = {key: best_threshold[key] for key in default}

    morphology = []
    for opening_size in [3, 5, 7]:
        for min_nucleus_size in [3, 5, 10]:
            morphology.append({**anchor, "opening_size": opening_size, "min_nucleus_size": min_nucleus_size, "stage": "morphology"})
    morphology_rows = evaluate_many(args.cache, morphology, args.workers)
    best_morphology = max(morphology_rows, key=lambda row: row["bPQ"])
    anchor = {key: best_morphology[key] for key in default}
    cache_metadata = json.loads((args.cache / "cache.json").read_text())
    directional_rows = []
    if int(cache_metadata.get("map_channels", 2)) == 4:
        directional = [
            {**anchor, "directional_weight": weight, "stage": "directional_weight"}
            for weight in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        directional_rows = evaluate_many(args.cache, directional, args.workers)
        best_directional = max(directional_rows, key=lambda row: row["bPQ"])
        anchor = {key: best_directional[key] for key in default}

    # Recheck the complete Sobel grid after threshold, morphology, and optional
    # directional-weight tuning. Kernel/threshold interactions are otherwise
    # capable of making the initial one-factor sweep select the wrong kernel.
    kernel_refinement = [
        {**anchor, "ksize": ksize, "stage": "kernel_refinement"}
        for ksize in sobel_kernel_grid
    ]
    kernel_refinement_rows = evaluate_many(args.cache, kernel_refinement, args.workers)
    best_kernel = max(kernel_refinement_rows, key=lambda row: row["bPQ"])
    anchor = {key: best_kernel[key] for key in default}

    edge_refinement = [
        {**anchor, "edge_threshold": threshold, "stage": "edge_threshold_refinement"}
        for threshold in edge_threshold_grid
    ]
    edge_refinement_rows = evaluate_many(args.cache, edge_refinement, args.workers)
    best_edge = max(edge_refinement_rows, key=lambda row: row["bPQ"])
    anchor = {key: best_edge[key] for key in default}

    kernel_confirmation = [
        {**anchor, "ksize": ksize, "stage": "kernel_confirmation"}
        for ksize in sobel_kernel_grid
    ]
    kernel_confirmation_rows = evaluate_many(args.cache, kernel_confirmation, args.workers)
    decoder_rows = (
        coarse_rows
        + threshold_rows
        + morphology_rows
        + directional_rows
        + kernel_refinement_rows
        + edge_refinement_rows
        + kernel_confirmation_rows
    )
    best = max(kernel_confirmation_rows, key=lambda row: row["bPQ"])
    best_config = {key: best[key] for key in default}
    tta_rows = evaluate_many(
        args.cache,
        [{**best_config, "stage": "flip_tta_locked_decoder"}],
        args.workers,
        map_file="raw_maps_flip_tta.npy",
    )
    rows = decoder_rows + tta_rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "selection_split": "validation",
        "selection_metric": "pooled binary PQ",
        "sobel_kernel_grid": sobel_kernel_grid,
        "edge_threshold_grid": edge_threshold_grid,
        "n_patches": len(patch_ids),
        "n_decoder_candidates": len(decoder_rows),
        "default_20x": next(row for row in coarse_rows if row["object_size"] == 3 and row["ksize"] == 11),
        "best": {"config": best_config, "metrics": {key: best[key] for key in ["bPQ", "DQ", "SQ", "tp", "fp", "fn", "count_ratio"]}},
        "flip_tta_locked_decoder": {"config": best_config, "metrics": {key: tta_rows[0][key] for key in ["bPQ", "DQ", "SQ", "tp", "fp", "fn", "count_ratio"]}},
        "stages": [
            "object_kernel",
            "thresholds",
            "morphology",
            *(["directional_weight"] if directional_rows else []),
            "kernel_refinement",
            "edge_threshold_refinement",
            "kernel_confirmation",
            "flip_tta_locked_decoder",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(f"selected validation bPQ={best['bPQ']:.5f}: {report['best']['config']}", flush=True)


if __name__ == "__main__":
    main()
