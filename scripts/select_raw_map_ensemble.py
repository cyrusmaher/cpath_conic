#!/usr/bin/env python
"""Validation-select equal-weight ensembles before a shared HV decoder."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.hv import decode_hv, fast_binary_pq_stats, load_decoder_config


def summarize(totals, predicted_instances, true_instances) -> dict:
    tp, fp, fn, sum_iou = totals
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "bPQ": float(sum_iou / denominator if denominator else 0.0),
        "DQ": float(tp / denominator if denominator else 0.0),
        "SQ": float(sum_iou / tp if tp else 0.0),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "count_ratio": float(predicted_instances / true_instances if true_instances else 0.0),
    }


def evaluate_subset(maps, subset, truth, config, sources) -> tuple[dict, dict]:
    channel_count = max(maps[index].shape[-1] for index in subset)
    totals = np.zeros(4, dtype=np.float64)
    predicted_instances = 0
    true_instances = 0
    source_totals = {source: np.zeros(4, dtype=np.float64) for source in sorted(set(sources))}
    source_predicted = {source: 0 for source in source_totals}
    source_true = {source: 0 for source in source_totals}
    for patch_index, source in enumerate(sources):
        averaged = np.stack(
            [
                np.mean(
                    [
                        np.asarray(maps[index][patch_index, ..., channel], dtype=np.float32)
                        for index in subset
                        if maps[index].shape[-1] > channel
                    ],
                    axis=0,
                )
                for channel in range(channel_count)
            ],
            axis=-1,
        )
        prediction = decode_hv(averaged[..., 0], averaged[..., 1:], **config)
        stats = fast_binary_pq_stats(truth[patch_index], prediction)
        n_predicted = int(prediction.max())
        n_true = int(truth[patch_index].max())
        totals += stats
        predicted_instances += n_predicted
        true_instances += n_true
        source_totals[source] += stats
        source_predicted[source] += n_predicted
        source_true[source] += n_true
    by_source = {
        source: summarize(source_totals[source], source_predicted[source], source_true[source])
        for source in source_totals
    }
    return summarize(totals, predicted_instances, true_instances), by_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caches", type=Path, nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--decoder-config", type=Path, required=True)
    parser.add_argument("--map-file", default="raw_maps_flip_tta.npy")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.caches) != len(args.names):
        raise ValueError("--caches and --names must have the same length")
    patch_ids = np.load(args.caches[0] / "patch_ids.npy")
    truth = np.load(args.caches[0] / "true_instances.npy", mmap_mode="r")
    maps = []
    for cache in args.caches:
        if not np.array_equal(np.load(cache / "patch_ids.npy"), patch_ids):
            raise ValueError(f"Patch IDs differ in {cache}")
        candidate_truth = np.load(cache / "true_instances.npy", mmap_mode="r")
        if candidate_truth.shape != truth.shape:
            raise ValueError(f"Truth shape differs in {cache}")
        maps.append(np.load(cache / args.map_file, mmap_mode="r"))
    metadata = pd.read_csv(args.prepared / "metadata.csv").set_index("patch_id").loc[patch_ids]
    sources = metadata.source.astype(str).to_numpy()
    config = load_decoder_config(args.decoder_config)
    rows = []
    for size in range(1, len(maps) + 1):
        for subset in itertools.combinations(range(len(maps)), size):
            channel_count = max(maps[index].shape[-1] for index in subset)
            overall, by_source = evaluate_subset(maps, subset, truth, config, sources)
            rows.append(
                {
                    "members": "+".join(args.names[index] for index in subset),
                    "n_members": size,
                    "directional_channels": channel_count - 1,
                    **overall,
                    "worst_source_bPQ": min(metrics["bPQ"] for metrics in by_source.values()),
                    "by_source": by_source,
                }
            )
            print(f"{rows[-1]['members']}: bPQ={overall['bPQ']:.5f} DQ={overall['DQ']:.5f} SQ={overall['SQ']:.5f}", flush=True)
    selected = max(rows, key=lambda row: (row["bPQ"], row["worst_source_bPQ"], -row["n_members"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"selection_metric": "validation bPQ", "selected": selected, "candidates": rows}, indent=2))
    flat_rows = [{key: value for key, value in row.items() if key != "by_source"} for row in rows]
    with args.out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"selected {selected['members']} at validation bPQ={selected['bPQ']:.5f}", flush=True)


if __name__ == "__main__":
    main()
