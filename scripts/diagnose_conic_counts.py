#!/usr/bin/env python
"""Audit CoNIC count alignment and per-class regression behavior."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS
from cpath_conic.data import central_crop_counts, load_metadata, patch_count_from_maps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    metadata = load_metadata(args.prepared)
    subset = metadata.loc[metadata.split == args.split].sort_values("patch_id")
    counts = np.load(args.counts)
    gt = subset[COUNT_COLUMNS].to_numpy(dtype=float)
    pred_by_id = counts[subset.patch_id.to_numpy()]
    pred_by_index = counts[subset.index.to_numpy()]

    reconstruction_mismatches = 0
    max_reconstruction_error = 0
    full_count_mismatches = 0
    crop_presence_mismatches = 0
    first_mismatch = None
    for patch_id in metadata.patch_id:
        label = np.load(args.prepared / "labels" / f"{int(patch_id):05d}.npy", allow_pickle=True).item()
        derived = central_crop_counts(label["inst_map"], label["class_map"])
        full_counts = patch_count_from_maps(label["inst_map"], label["class_map"])
        crop_presence_counts = patch_count_from_maps(label["inst_map"][16:-16, 16:-16], label["class_map"][16:-16, 16:-16])
        stored = metadata.loc[metadata.patch_id == patch_id, COUNT_COLUMNS].iloc[0].to_numpy()
        reconstruction_mismatches += int(not np.array_equal(derived, stored))
        full_count_mismatches += int(not np.array_equal(full_counts, stored))
        crop_presence_mismatches += int(not np.array_equal(crop_presence_counts, stored))
        max_reconstruction_error = max(max_reconstruction_error, int(np.abs(derived - stored).max()))
        if first_mismatch is None and not np.array_equal(derived, stored):
            first_mismatch = (int(patch_id), stored.tolist(), derived.tolist(), full_counts.tolist())

    print(f"rows={len(metadata)} split={args.split} split_rows={len(subset)}")
    print(f"index_equals_patch={bool(np.array_equal(metadata.index.to_numpy(), metadata.patch_id.to_numpy()))}")
    print(f"saved_count_alignment_equal={bool(np.array_equal(pred_by_id, pred_by_index))}")
    print(f"gt_count_reconstruction_mismatches={reconstruction_mismatches} max_error={max_reconstruction_error}")
    print(f"gt_full_patch_count_mismatches={full_count_mismatches}")
    print(f"gt_crop_presence_count_mismatches={crop_presence_mismatches}")
    print(f"first_mismatch={first_mismatch}")
    print("class,gt_mean,pred_mean,gt_std,pred_std,r2_by_id,pearson_by_id,pred_zero_fraction")
    for index, name in enumerate(CLASS_NAMES):
        true = gt[:, index]
        pred = pred_by_id[:, index]
        correlation = float(pearsonr(true, pred).statistic) if np.std(true) and np.std(pred) else float("nan")
        print(",".join([
            name,
            f"{true.mean():.4f}",
            f"{pred.mean():.4f}",
            f"{true.std():.4f}",
            f"{pred.std():.4f}",
            f"{r2_score(true, pred):.4f}",
            f"{correlation:.4f}",
            f"{np.mean(pred == 0):.4f}",
        ]))


if __name__ == "__main__":
    main()
