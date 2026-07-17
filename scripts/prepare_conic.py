#!/usr/bin/env python
"""Convert CoNIC Parquet rows to patch files and source-level splits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS
from cpath_conic.data import _decode_image, _decode_mask, source_group, stable_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    files = sorted(args.raw.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {args.raw}; run download_conic.py first")
    (args.outdir / "images").mkdir(parents=True, exist_ok=True)
    (args.outdir / "labels").mkdir(parents=True, exist_ok=True)
    (args.outdir / "splits").mkdir(parents=True, exist_ok=True)
    rows = []
    seen = 0
    for parquet_path in files:
        table = pq.read_table(parquet_path)
        frame = table.to_pandas()
        for record in frame.to_dict(orient="records"):
            if args.limit is not None and seen >= args.limit:
                break
            patch_id = int(record["patch_id"])
            image = _decode_image(record["image"])
            inst_map = _decode_mask(record["inst_map"], np.uint32)
            class_map = _decode_mask(record["class_map"], np.uint8)
            if image.shape != (256, 256, 3) or inst_map.shape != (256, 256) or class_map.shape != (256, 256):
                raise ValueError(f"Unexpected shape for patch {patch_id}: {image.shape}, {inst_map.shape}, {class_map.shape}")
            Image.fromarray(image).save(args.outdir / "images" / f"{patch_id:05d}.png")
            np.save(args.outdir / "labels" / f"{patch_id:05d}.npy", {"inst_map": inst_map, "label_map": class_map, "class_map": class_map})
            source = str(record.get("source", ""))
            patch_info = str(record.get("patch_info", patch_id))
            group = source_group(patch_info, source)
            row = {"patch_id": patch_id, "patch_info": patch_info, "source": source, "source_group": group, "split": stable_split(group, args.seed)}
            for name in CLASS_NAMES:
                key = f"count_{name}"
                row[key] = int(record.get(key, 0))
            rows.append(row)
            seen += 1
        if args.limit is not None and seen >= args.limit:
            break
    metadata = pd.DataFrame(rows).sort_values("patch_id")
    metadata.to_csv(args.outdir / "metadata.csv", index=False)
    for split in ["train", "val", "test"]:
        subset = metadata.loc[metadata.split == split].copy()
        subset.to_csv(args.outdir / "splits" / f"{split}.csv", index=False)
    (args.outdir / "label_map.yaml").write_text("background: 0\n" + "\n".join(f"{name}: {i}" for i, name in enumerate(CLASS_NAMES, 1)) + "\n")
    (args.outdir / "preparation.json").write_text(json.dumps({"seed": args.seed, "patches": len(metadata), "class_names": CLASS_NAMES, "count_columns": COUNT_COLUMNS, "source_level_split": True}, indent=2))
    print(metadata.groupby("split").size().to_string())


if __name__ == "__main__":
    main()
