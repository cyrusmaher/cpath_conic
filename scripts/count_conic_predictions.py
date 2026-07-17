#!/usr/bin/env python
"""Recompute official central-crop counts from prediction maps."""
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
    counts = np.asarray(
        [central_crop_counts(patch[..., 0], patch[..., 1]) for patch in predictions],
        dtype=np.int32,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, counts)
    print(f"wrote {counts.shape} counts to {args.out}")


if __name__ == "__main__":
    main()
