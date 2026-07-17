#!/usr/bin/env python
"""Compare one raw CoNIC parquet row with its prepared masks and counts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS
from cpath_conic.data import _decode_mask, central_crop_counts, patch_count_from_maps


def instance_class(inst_map: np.ndarray, class_map: np.ndarray, instance_id: int) -> int:
    pixels = class_map[inst_map == instance_id].astype(np.int64)
    return int(np.bincount(pixels, minlength=7).argmax())


def counts_by_rule(inst_map: np.ndarray, class_map: np.ndarray, rule: str, margin: int = 16) -> np.ndarray:
    h, w = inst_map.shape
    y0, y1, x0, x1 = margin, h - margin, margin, w - margin
    counts = np.zeros(6, dtype=np.int64)
    for instance_id in np.unique(inst_map):
        if instance_id == 0:
            continue
        ys, xs = np.where(inst_map == instance_id)
        inside = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)
        if rule == "any_inside":
            keep = bool(inside.any())
        elif rule == "majority_inside":
            keep = bool(inside.mean() > 0.5)
        elif rule == "all_inside":
            keep = bool(inside.all())
        elif rule == "bbox_center":
            keep = bool(y0 <= (ys.min() + ys.max()) / 2 < y1 and x0 <= (xs.min() + xs.max()) / 2 < x1)
        else:
            raise ValueError(rule)
        cls = instance_class(inst_map, class_map, int(instance_id))
        if keep and 1 <= cls <= 6:
            counts[cls - 1] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--patch-id", type=int, default=0)
    args = parser.parse_args()

    row = None
    source_file = None
    columns = ["patch_id", "inst_map", "class_map", *COUNT_COLUMNS]
    for parquet_path in sorted(args.raw.glob("*.parquet")):
        frame = pq.read_table(parquet_path, columns=columns, filters=[("patch_id", "=", args.patch_id)]).to_pandas()
        if len(frame):
            row = frame.iloc[0]
            source_file = parquet_path
            break
    if row is None:
        raise KeyError(args.patch_id)

    raw_inst = _decode_mask(row.inst_map, np.uint32)
    raw_class = _decode_mask(row.class_map, np.uint8)
    prepared = np.load(args.prepared / "labels" / f"{args.patch_id:05d}.npy", allow_pickle=True).item()
    stored = row[COUNT_COLUMNS].to_numpy(dtype=np.int64)

    print(f"source={source_file}")
    print(f"raw_inst_type={type(row.inst_map).__name__} raw_class_type={type(row.class_map).__name__}")
    print(f"inst_equal={np.array_equal(raw_inst, prepared['inst_map'])} class_equal={np.array_equal(raw_class, prepared['class_map'])}")
    print(f"inst_shape={raw_inst.shape} inst_dtype={raw_inst.dtype} instances={len(np.unique(raw_inst)) - 1}")
    print(f"class_shape={raw_class.shape} class_dtype={raw_class.dtype} classes={np.unique(raw_class).tolist()}")
    print(f"stored={dict(zip(CLASS_NAMES, stored.tolist()))}")
    rules = {
        "full": patch_count_from_maps(raw_inst, raw_class),
        "pixel_centroid": central_crop_counts(raw_inst, raw_class),
        "bbox_center": counts_by_rule(raw_inst, raw_class, "bbox_center"),
        "any_inside": counts_by_rule(raw_inst, raw_class, "any_inside"),
        "majority_inside": counts_by_rule(raw_inst, raw_class, "majority_inside"),
        "all_inside": counts_by_rule(raw_inst, raw_class, "all_inside"),
    }
    for name, values in rules.items():
        print(f"{name}={dict(zip(CLASS_NAMES, values.tolist()))} absolute_error={int(np.abs(values - stored).sum())}")


if __name__ == "__main__":
    main()
