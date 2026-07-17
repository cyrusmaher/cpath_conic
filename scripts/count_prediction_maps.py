#!/usr/bin/env python3
"""Materialize central-crop class counts from full CoNIC prediction maps."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.data import central_crop_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    predictions = np.load(args.predictions, mmap_mode="r")
    if predictions.ndim != 4 or predictions.shape[-1] != 2:
        raise ValueError("predictions must have shape N,H,W,2")
    counts = np.zeros((len(predictions), 6), dtype=np.int32)
    for patch_id, patch in enumerate(predictions):
        counts[patch_id] = central_crop_counts(patch[..., 0], patch[..., 1])
        if (patch_id + 1) % 500 == 0 or patch_id + 1 == len(predictions):
            print(f"counted {patch_id + 1}/{len(predictions)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, counts)


if __name__ == "__main__":
    main()
