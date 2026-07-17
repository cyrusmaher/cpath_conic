#!/usr/bin/env python
"""Paired class/source audit for decoded validation-map caches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES
from cpath_conic.hv import decode_hv, fast_binary_pq_stats
from scripts.audit_lora_detection_by_class import (
    empty_totals,
    paired_bootstrap_bpq,
    summarize,
    update_totals,
)


def patch_bpq(stats: np.ndarray) -> float:
    tp, fp, fn, sum_iou = np.asarray(stats, dtype=np.float64)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return float(sum_iou / denominator) if denominator else 0.0


def decoder_config(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return payload.get("best", {}).get("config", payload.get("config", payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--decoder-report", type=Path, required=True)
    parser.add_argument("--candidate-map-file", default="raw_maps_flip_tta.npy")
    parser.add_argument("--reference-cache", type=Path, default=None)
    parser.add_argument("--reference-decoder-report", type=Path, default=None)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference-map-file")
    reference.add_argument("--reference-instance-maps", type=Path)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--validation-split", choices=["train", "val"], default="val")
    parser.add_argument("--validation-sources", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    metadata = pd.read_csv(args.prepared / "metadata.csv")
    metadata_by_id = metadata.set_index("patch_id")
    patch_ids = np.load(args.cache / "patch_ids.npy")
    validation_sources = [source.strip().lower() for source in args.validation_sources.split(",") if source.strip()]
    validation_mask = metadata.split.eq(args.validation_split)
    if validation_sources:
        validation_mask &= metadata.source.astype(str).str.lower().isin(validation_sources)
    expected_ids = metadata.loc[validation_mask, "patch_id"].to_numpy(dtype=np.int32)
    if not np.array_equal(patch_ids, expected_ids):
        raise ValueError("Cache IDs do not exactly match the validation split")

    config = decoder_config(args.decoder_report)
    reference_config = decoder_config(args.reference_decoder_report) if args.reference_decoder_report else config
    candidate_raw = np.load(args.cache / args.candidate_map_file, mmap_mode="r")
    reference_cache = args.reference_cache or args.cache
    reference_raw = np.load(reference_cache / args.reference_map_file, mmap_mode="r") if args.reference_map_file else None
    reference_instances = np.load(args.reference_instance_maps, mmap_mode="r") if args.reference_instance_maps else None
    truth = np.load(args.cache / "true_instances.npy", mmap_mode="r")
    if len(candidate_raw) != len(patch_ids) or len(truth) != len(patch_ids):
        raise ValueError("Candidate cache and truth are not aligned")
    if reference_raw is not None and len(reference_raw) != len(patch_ids):
        raise ValueError("Reference cache and candidate cache are not aligned")
    if reference_raw is not None:
        reference_ids = np.load(reference_cache / "patch_ids.npy")
        if not np.array_equal(reference_ids, patch_ids):
            raise ValueError("Reference and candidate cache patch IDs differ")

    sources = sorted(metadata.loc[validation_mask, "source"].dropna().astype(str).unique())
    candidate_totals = empty_totals()
    reference_totals = empty_totals()
    candidate_by_source = {source: empty_totals() for source in sources}
    reference_by_source = {source: empty_totals() for source in sources}
    candidate_stats = []
    reference_stats = []
    patch_rows = []

    for index, patch_id in enumerate(patch_ids):
        patch_id = int(patch_id)
        label = np.load(args.prepared / "labels" / f"{patch_id:05d}.npy", allow_pickle=True).item()
        true_inst = np.asarray(truth[index], dtype=np.int32)
        class_map = label["class_map"].astype(np.int8)
        candidate = decode_hv(candidate_raw[index, ..., 0], candidate_raw[index, ..., 1:], **config)
        if reference_raw is not None:
            reference_map = decode_hv(reference_raw[index, ..., 0], reference_raw[index, ..., 1:], **reference_config)
        else:
            reference_map = np.asarray(reference_instances[patch_id], dtype=np.int32)
        candidate_patch = np.asarray(fast_binary_pq_stats(true_inst, candidate), dtype=np.float64)
        reference_patch = np.asarray(fast_binary_pq_stats(true_inst, reference_map), dtype=np.float64)
        candidate_stats.append(candidate_patch)
        reference_stats.append(reference_patch)
        source = str(metadata_by_id.loc[patch_id, "source"])
        update_totals(candidate_totals, true_inst, class_map, candidate)
        update_totals(reference_totals, true_inst, class_map, reference_map)
        update_totals(candidate_by_source[source], true_inst, class_map, candidate)
        update_totals(reference_by_source[source], true_inst, class_map, reference_map)
        candidate_score = patch_bpq(candidate_patch)
        reference_score = patch_bpq(reference_patch)
        patch_rows.append(
            {
                "patch_id": patch_id,
                "source": source,
                "candidate_bPQ": candidate_score,
                "reference_bPQ": reference_score,
                "delta_bPQ": candidate_score - reference_score,
                "true_instances": int(true_inst.max()),
                "candidate_instances": int(candidate.max()),
                "reference_instances": int(reference_map.max()),
            }
        )

    candidate_summary = summarize(candidate_totals)
    reference_summary = summarize(reference_totals)
    class_deltas = {
        name: {
            key: candidate_summary["by_class"][name][key] - reference_summary["by_class"][name][key]
            for key in ("matched", "recall_at_iou_0.5", "matched_mean_iou")
        }
        for name in CLASS_NAMES
    }
    ordered = sorted(patch_rows, key=lambda row: row["delta_bPQ"])
    report = {
        "split": args.validation_split,
        "validation_sources": validation_sources,
        "decoder_config": config,
        "reference_decoder_config": reference_config,
        "candidate": {"name": args.candidate_name, **candidate_summary},
        "reference": {"name": args.reference_name, **reference_summary},
        "paired_bPQ": paired_bootstrap_bpq(candidate_stats, reference_stats),
        "candidate_minus_reference_by_class": class_deltas,
        "by_source": {
            source: {
                "reference": summarize(reference_by_source[source]),
                "candidate": summarize(candidate_by_source[source]),
            }
            for source in sources
        },
        "largest_patch_losses": ordered[:12],
        "largest_patch_gains": list(reversed(ordered[-12:])),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "candidate": {key: candidate_summary[key] for key in ("bPQ", "DQ", "SQ", "matched", "predicted")},
                "reference": {key: reference_summary[key] for key in ("bPQ", "DQ", "SQ", "matched", "predicted")},
                "paired_bPQ": report["paired_bPQ"],
                "out": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
