#!/usr/bin/env python
"""Select a 2×-resolution HV decoder using cached validation maps."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.hv import decode_hv, fast_binary_pq_stats


def evaluate_one(payload: tuple[str, str, dict]) -> dict:
    cache, map_file, config = payload
    cache_path = Path(cache)
    raw = np.load(cache_path / map_file, mmap_mode="r")
    truth = np.load(cache_path / "true_instances.npy", mmap_mode="r")
    totals = np.zeros(4, dtype=np.float64)
    predicted_count = 0
    true_count = 0
    for index in range(len(raw)):
        predicted = decode_hv(raw[index, ..., 0], raw[index, ..., 1:], scale=2.0, **config)
        totals += fast_binary_pq_stats(truth[index], predicted)
        predicted_count += int(predicted.max())
        true_count += int(truth[index].max())
    tp, fp, fn, sum_iou = totals
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        **config,
        "scale": 2.0,
        "bPQ": float(sum_iou / denominator if denominator else 0.0),
        "DQ": float(tp / denominator if denominator else 0.0),
        "SQ": float(sum_iou / tp if tp else 0.0),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
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
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    # CellViT's documented 0.25-mpp defaults establish the high-resolution
    # anchor.  Subsequent stages change one parameter family at a time.
    default = {
        "binary_threshold": 0.5,
        "edge_threshold": 0.4,
        "object_size": 10,
        "ksize": 21,
        "opening_size": 5,
        "min_nucleus_size": 10,
    }
    coarse = [
        {**default, "object_size": object_size, "ksize": ksize, "stage": "object_kernel"}
        for object_size in [1, 4, 10, 20]
        for ksize in [11, 15, 21, 31]
    ]
    coarse_rows = evaluate_many(args.cache, coarse, args.workers)
    best_coarse = max(coarse_rows, key=lambda row: row["bPQ"])
    anchor = {key: best_coarse[key] for key in default}

    thresholds = [
        {**anchor, "binary_threshold": binary, "edge_threshold": edge, "stage": "thresholds"}
        for binary in [0.4, 0.5, 0.6]
        for edge in [0.3, 0.4, 0.5]
    ]
    threshold_rows = evaluate_many(args.cache, thresholds, args.workers)
    best_threshold = max(threshold_rows, key=lambda row: row["bPQ"])
    anchor = {key: best_threshold[key] for key in default}

    morphology = [
        {**anchor, "opening_size": opening, "min_nucleus_size": minimum, "stage": "morphology"}
        for opening in [3, 5, 9]
        for minimum in [5, 10, 20]
    ]
    morphology_rows = evaluate_many(args.cache, morphology, args.workers)
    candidates = coarse_rows + threshold_rows + morphology_rows
    best = max(candidates, key=lambda row: row["bPQ"])
    best_config = {key: best[key] for key in default}
    best_config["scale"] = 2.0
    tta_row = evaluate_many(
        args.cache,
        [{key: best[key] for key in default} | {"stage": "flip_tta_locked_decoder"}],
        args.workers,
        map_file="raw_maps_flip_tta.npy",
    )[0]
    rows = candidates + [tta_row]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "selection_split": "validation",
        "selection_metric": "pooled binary PQ",
        "n_patches": int(len(np.load(args.cache / "patch_ids.npy"))),
        "n_decoder_candidates": len(candidates),
        "scale": 2.0,
        "documented_025_mpp_default": next(row for row in coarse_rows if row["object_size"] == 10 and row["ksize"] == 21),
        "best": {"config": best_config, "metrics": {key: best[key] for key in ["bPQ", "DQ", "SQ", "tp", "fp", "fn", "count_ratio"]}},
        "flip_tta_locked_decoder": {"config": best_config, "metrics": {key: tta_row[key] for key in ["bPQ", "DQ", "SQ", "tp", "fp", "fn", "count_ratio"]}},
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(f"selected 2x validation bPQ={best['bPQ']:.5f}: {best_config}", flush=True)


if __name__ == "__main__":
    main()
